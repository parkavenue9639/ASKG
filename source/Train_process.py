import os
import sys
import torch
import cv2
import json
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm, trange
from skimage.measure import label
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score

import eval_utils

# 将项目根目录加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pretrain.misc.utils import NoamOpt


class TextTrainProcess:
    def __init__(self, device, opt, dataloader_list, model):
        self.device = device
        self.opt = opt
        self.dataloader = {'train': dataloader_list[0], 'valid': dataloader_list[1], 'test': dataloader_list[2]}
        self.model = model
        self.checkpoint_path = '../checkpoint/abstract'
        self.best_model_path = '../best_model/abstract'
        self.start_epoch = 0
        self.best_val_score = None
        self.med_crit = None
        self.outputs_crit = None
        self.rnn_NoamOpt = None

    def check_checkpoint(self):
        # 获取所有 checkpoint 文件
        checkpoints = sorted(
            [f for f in os.listdir(self.checkpoint_path) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])  # 提取 epoch 号并排序
        )

        if checkpoints:
            latest_checkpoint_path = os.path.join(self.checkpoint_path, checkpoints[-1])
            checkpoint = torch.load(latest_checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state'])
            self.rnn_NoamOpt.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.start_epoch = checkpoint['epoch']
            self.best_val_score = checkpoint.get('best_val_score', None)
            print("Loaded checkpoint from {}, starting from epoch {}".format(latest_checkpoint_path, self.start_epoch))
        else:
            self.start_epoch = 0
            self.best_val_score = None
            print("No valid checkpoint found. Starting training from scratch.")

    def init_loss_func(self):
        # 主要用于二分类
        self.med_crit = nn.BCELoss().to(self.device)
        # 主要用于多分类
        self.outputs_crit = nn.CrossEntropyLoss(ignore_index=-1).to(self.device)

        self.rnn_NoamOpt = NoamOpt(self.opt.d_model, self.opt.factor, self.opt.warmup,
                                   torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    def abstract_train(self):
        self.model.train()
        tqdm_bar = tqdm(self.dataloader['train'], desc='Training')

        for iteration, (ids, abstracts, abstracts_labels, medterm_labels) in enumerate(tqdm_bar):
            abstracts = abstracts.to(self.device)
            abstracts_labels = abstracts_labels.to(self.device)
            medterm_labels = medterm_labels.to(self.device)

            self.rnn_NoamOpt.zero_grad()

            med_probs, abstracts_outputs = self.model(att_feats=abstracts, input_ids=abstracts, input_type='sentence')
            med_loss = self.med_crit(med_probs, medterm_labels)

            caption_loss = self.outputs_crit(abstracts_outputs.view(-1, abstracts_outputs.size(-1)),
                                             abstracts_labels.view(-1))
            loss = 2.0 * med_loss + caption_loss
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.grad_clip)
            self.rnn_NoamOpt.step()

            train_loss = loss.item()

            tqdm_bar.desc = "train_loss = {:.5f}, med_loss = {:.5f}, caption_loss = {:.5f}," \
                            "lr = {:.5f}, cnn_lr = {:.5f}" \
                .format(train_loss, med_loss, caption_loss,
                        self.rnn_NoamOpt.optimizer.param_groups[0]['lr'], self.opt.cnn_learning_rate)

    def eval(self):
        tqdm_bar = tqdm(self.dataloader['valid'], desc="Evaluating")
        self.model.eval()

        eval_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        all_labels = []
        all_preds = []
        with torch.no_grad():
            for iteration, (ids, abstracts, abstracts_labels, medterm_labels) in enumerate(tqdm_bar):
                abstracts = abstracts.to(self.device)
                abstracts_labels = abstracts_labels.to(self.device)
                # print("abstracts_labels shape: {}".format(abstracts_labels.shape))  # [16, 300]
                # print("abstracts_labels type: {}".format(abstracts_labels.type))  # Tensor object

                # 前向传播
                med_probs, abstracts_outputs = self.model(att_feats=abstracts, input_ids=abstracts,
                                                          input_type='sentence')

                loss = self.outputs_crit(abstracts_outputs.view(-1, abstracts_outputs.size(-1)),
                                         abstracts_labels.view(-1))
                preds = torch.argmax(med_probs, dim=-1)  # 多分类预测类别

                eval_loss += loss.item()
                total_samples += abstracts.size(0)

                # 收集预测与标签
                '''all_labels.extend(abstracts_labels.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())'''

                # 获取预测类别标签
                _, predicted = torch.max(abstracts_outputs, dim=-1)
                # print("predicted shape: {}".format(predicted.shape))  # [16, 300]
                # print("predicted type: {}".format(predicted.type))  # Tensor object
                if not isinstance(predicted, torch.Tensor):
                    predicted = torch.tensor(predicted)

                if not isinstance(abstracts_labels, torch.Tensor):
                    abstracts_labels = torch.tensor(abstracts_labels)

                # 计算准确率
                correct_predictions += (predicted == abstracts_labels).sum().item()
                total_samples += abstracts_labels.size(0)

            # 计算平均损失
            avg_loss = eval_loss / total_samples
            # accuracy = accuracy_score(all_labels, all_preds)
            # macro_f1 = f1_score(all_labels, all_preds, average='macro')
            accuracy = correct_predictions / total_samples
            print("Eval Loss: {} | Accuracy: {} | Macro F1: NA".format(avg_loss, accuracy))
            return accuracy

    def free_cache(self):
        if self.device.type == 'mps':
            print("free mps memory")
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()

    def train(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.best_model_path, exist_ok=True)
        self.init_loss_func()
        self.check_checkpoint()

        for epoch in trange(self.start_epoch, int(self.opt.max_epochs), desc="Epoch"):
            try:
                self.abstract_train()
            except Exception as e:
                print("Error during training: {}".format(e))

            if epoch % self.opt.val_every_epoch == 0:
                current_score = self.eval()
                self.free_cache()  # 释放内存

                best_flag = False
                if self.best_val_score is None or current_score > self.best_val_score:
                    self.best_val_score = current_score
                    best_flag = True

                # 保存 checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.rnn_NoamOpt.optimizer.state_dict(),
                    'best_val_score': self.best_val_score
                }
                checkpoint_path = os.path.join(self.checkpoint_path, "checkpoint_epoch_{}.pth".format(epoch + 1))
                torch.save(checkpoint, checkpoint_path)
                print("Checkpoint saved to {}".format(checkpoint_path))

                # 获取所有 checkpoint 文件并按照 epoch 排序
                checkpoints = sorted(
                    [f for f in os.listdir(self.checkpoint_path) if
                     f.startswith("checkpoint_epoch_") and f.endswith(".pth")],
                    key=lambda x: int(x.split("_")[-1].split(".")[0])  # 提取 epoch 号并排序
                )

                # 若超过 max_checkpoints，则删除最早的 checkpoint
                if len(checkpoints) > 3:
                    oldest_checkpoint = checkpoints[0]
                    os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                    print("Removed oldest checkpoint: {}".format(oldest_checkpoint))

                if best_flag:
                    best_model_path = os.path.join(self.best_model_path, 'best_model.pth')
                    torch.save(self.model.state_dict(), best_model_path)
                    print("Best model saved to {}".format(best_model_path))
        self.free_cache()


class ImageTrain:
    def __init__(self, device, opt, dataloader_list, image_dataset_list, model, other_model):
        self.device = device
        self.opt = opt
        self.dataloader = {'train': dataloader_list[0], 'valid': dataloader_list[1], 'test': dataloader_list[2]}
        self.dataset = {'train': image_dataset_list[0], 'valid': image_dataset_list[1], 'test': image_dataset_list[2]}
        self.model = model
        self.cnn_model = other_model[0]
        self.aux_model = other_model[1]
        self.fusion_model = other_model[2]
        self.best_abstract_model_path = "../best_model/abstract"
        self.checkpoint_path = '../checkpoint/image'
        self.best_model_path = '../best_model/image'
        self.start_epoch = 0
        self.best_val_score = None
        self.med_crit = None
        self.outputs_crit = None
        self.rnn_NoamOpt = None
        self.cnn_optimizer = None
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])

    def check_abstract_model(self):
        abstract_model_path = os.path.join(self.best_abstract_model_path, "best_model.pth")
        if not os.path.exists(abstract_model_path):
            print("abstract model does not exist")
            return
        else:
            abstract_model = torch.load(abstract_model_path)
            self.model.load_state_dict(abstract_model)
            print("Loaded abstract from {}".format(abstract_model_path))

    def check_checkpoint(self):
        # 获取所有 checkpoint 文件
        checkpoints = sorted(
            [f for f in os.listdir(self.checkpoint_path) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])  # 提取 epoch 号并排序
        )

        if checkpoints:
            latest_checkpoint_path = os.path.join(self.checkpoint_path, checkpoints[-1])
            checkpoint = torch.load(latest_checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state'])
            self.rnn_NoamOpt.optimizer.load_state_dict(checkpoint['rnn_optimizer_state'])
            self.cnn_optimizer.optimizer.load_state_dict(checkpoint['cnn_optimizer_state'])
            self.cnn_model.load_state_dict(checkpoint['cnn_model_state'])
            self.aux_model.load_state_dict(checkpoint['aux_model_state'])
            self.fusion_model.load_state_dict(checkpoint['fusion_model_state'])
            self.start_epoch = checkpoint['epoch']
            self.best_val_score = checkpoint.get('best_val_score', None)
            print("Loaded checkpoint from {}, starting from epoch {}".format(latest_checkpoint_path, self.start_epoch))
        else:
            self.start_epoch = 0
            self.best_val_score = None
            print("No valid checkpoint found. Starting training from scratch.")

    def init_loss_function(self):
        # 主要用于二分类
        self.med_crit = nn.BCELoss().to(self.device)
        # 主要用于多分类
        self.outputs_crit = nn.CrossEntropyLoss(ignore_index=-1).to(self.device)

        self.rnn_NoamOpt = NoamOpt(self.opt.d_model, self.opt.factor, self.opt.warmup,
                                   torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
        self.cnn_optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.opt.cnn_learning_rate, weight_decay=self.opt.weight_decay)

    def decode_transformer_findings(self, idw2word, sampled_findings):
        decode_list = []
        print("Max ID in idw2word:", max(idw2word.keys()))
        n_samples, n_words = sampled_findings.size()
        # print("n_samples: {}  n_words: {}".format(n_samples, n_words)) [16, 300]

        for n in range(n_samples):
            decoded = []
            words = []
            print(sampled_findings[n])
            for i in range(n_words):
                token_id = int(sampled_findings[n][i])
                token = self.dataset['train'].idw2word.get(token_id, '<UNK>')  # 避免 KeyError
                if token == '<BOS>':
                    continue
                if token == '<EOS>':
                    break
                if token != '<UNK>' and token != '<BLANK>':
                    words.append(token)
            if len(words) != 0:
                decoded.append(' '.join(words))
            decode_list.append(' '.join(decoded))
        return decode_list  # [batch_size, length]

    def eval(self):
        ## TODO

        # 创建一个进度条，用于显示评估进度。dataloader为验证集的loader
        tqdm_bar = tqdm(self.dataloader['valid'], desc="Evaluating")

        # 设置模型为eval模式，关闭drouptout
        self.cnn_model.eval()  # cnn提取全局图像特征
        self.aux_model.eval()  # 辅助模型，用于局部特征提取
        self.fusion_model.eval()  # 融合全局和局部特征
        self.model.eval()  # 最终模型，生成医学报告

        num_show = 0  # 用于控制显示数量

        id2findings_t = {}  # 存储真实finding
        id2findings_g = {}  # 存储生成的finding

        id2captions_t = {}  # 真实的文本描述
        id2captions_g = {}  # 生成的文本描述

        gt = torch.FloatTensor().to(self.device)  # 真实标签
        pred = torch.FloatTensor().to(self.device)  # 预测结果

        count = 0  # 用于统计评估的样本数
        with torch.no_grad():  # 禁用梯度计算，提高评估效率
            for iteration, (ids, image_ids, imgs, input_ids, lm_labels, medterm_labels) in enumerate(tqdm_bar):
                imgs = imgs.to(self.device)
                medterm_labels = medterm_labels.to(self.device)

                output_global, fm_global, pool_global = self.cnn_model(imgs)  # 全局输出、全局特征图、全局池化结果
                # print("output_global shape:{}".format(output_global.shape))  # [16, 229]
                # print("fm_global shape:{}".format(fm_global.shape))  # [16, 1024, 7, 7]
                # print("pool_global shape:{}".format(pool_global.shape))  # [16, 1024]

                patchs_var = self.Attention_gen_patchs(imgs, fm_global, self.device)  # 局部特征提取
                # print("patchs_var shape:{}".format(patchs_var.shape))  # [16, 3, 224, 224]

                output_local, _, pool_local = self.aux_model(patchs_var)  # 辅助模型提取局部特征：局部输出、局部特征池化
                # print("pool_local shape:{}".format(pool_local.shape))  # [16, 1024]
                # print(fusion_var.shape)
                output_fusion = self.fusion_model(pool_global, pool_local)  # 局部特征和全局特征融合
                # print("output_fusion shape: {}".format(output_fusion.shape))  # [16, 229]

                # 医学术语预测概率、预测的文本序列
                med_porbs, findings_seq = self.model(att_feats=output_fusion, mode='inference', input_type='img')
                if len(findings_seq) > 512:
                    findings_seq = findings_seq[:511]

                # 解码文本
                findings_samples = self.decode_transformer_findings(self.dataset['train'].idw2word, findings_seq)
                findings_truths = self.decode_transformer_findings(self.dataset['train'].idw2word, lm_labels)
                # print("findings_samples :{}".format(findings_samples))
                # print("findings_truths :{}".format(findings_truths))

                # 存储预测结果
                for i, ix in enumerate(image_ids):
                    if ix not in id2findings_t:
                        id2findings_t[ix] = []
                        id2findings_g[ix] = [findings_samples[i]]

                        id2captions_t[ix] = []
                        if len(findings_samples[i]) > 512:
                            findings_samples[i] = findings_samples[i][:511]
                        id2captions_g[ix] = [findings_samples[i]]

                    id2findings_t[ix].append(findings_truths[i])
                    id2captions_t[ix] = [findings_truths[i]]

                    # 打印前20个样本的预测结果，帮助观察模型效果
                    if num_show < 20:
                        print(json.dumps(id2captions_g[ix], ensure_ascii=False))
                        # print(json.dumps(id2captions_t[ix], ensure_ascii=False))
                        num_show += 1

                if count % 100 == 0:
                    print(count)
                count += 1

                # 将med_porbs预测概率和medterm_labels真实标签累积，用于计算后续指标
                pred = torch.cat((pred, med_porbs.data), 0)
                gt = torch.cat((gt, medterm_labels), 0)

        print('Total image to be evaluated %d' % (len(id2captions_t)))

        lang_stats = None
        # 如果开启语言评估，计算BLEU；CIDEr等指标
        if self.opt.language_eval == 1:
            lang_stats = eval_utils.evaluate(id2captions_t, id2captions_g, save_to='./results/',
                                             split='test_graph_pretrain')

        return lang_stats

    def free_cache(self):
        if self.device.type == 'mps':
            print("free mps memory")
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()

    def train(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.best_model_path, exist_ok=True)
        self.init_loss_function()
        self.check_abstract_model()
        self.check_checkpoint()
        if not self.best_val_score:
            self.best_val_score = self.eval()['CIDEr']
        for epoch in trange(self.start_epoch, int(self.opt.max_epochs), desc="Epoch"):
            try:
                self.image_train()
            except Exception as e:
                print("Error during training: {}".format(e))

            if epoch % self.opt.val_every_epoch == 0:
                current_score = self.eval()['CIDEr']
                self.free_cache()  # 释放内存

                best_flag = False
                if self.best_val_score is None or current_score > self.best_val_score:
                    self.best_val_score = current_score
                    best_flag = True

                # 保存 checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state': self.model.state_dict(),
                    'rnn_optimizer_state': self.rnn_NoamOpt.optimizer.state_dict(),
                    'cnn_optimizer_state': self.cnn_optimizer.state_dict(),
                    'cnn_model_state': self.cnn_model.state_dict(),
                    'aux_model_state': self.aux_model.state_dict(),
                    'fusion_model_state': self.fusion_model.state_dict(),
                    'best_val_score': self.best_val_score
                }
                checkpoint_path = os.path.join(self.checkpoint_path, "checkpoint_epoch_{}.pth".format(epoch + 1))
                torch.save(checkpoint, checkpoint_path)
                print("Checkpoint saved to {}".format(checkpoint_path))

                # 获取所有 checkpoint 文件并按照 epoch 排序
                checkpoints = sorted(
                    [f for f in os.listdir(self.checkpoint_path) if
                     f.startswith("checkpoint_epoch_") and f.endswith(".pth")],
                    key=lambda x: int(x.split("_")[-1].split(".")[0])  # 提取 epoch 号并排序
                )

                # 若超过 max_checkpoints，则删除最早的 checkpoint
                if len(checkpoints) > 3:
                    oldest_checkpoint = checkpoints[0]
                    os.remove(os.path.join(self.checkpoint_path, oldest_checkpoint))
                    print("Removed oldest checkpoint: {}".format(oldest_checkpoint))

                if best_flag:
                    best_model_path = os.path.join(self.best_model_path, 'best_model.pth')
                    torch.save(self.model.state_dict(), best_model_path)
                    print("Best model saved to {}".format(best_model_path))
        self.free_cache()

    def binImage(self, heatmap):
        _, heatmap_bin = cv2.threshold(heatmap, 178, 255, cv2.THRESH_BINARY)
        return heatmap_bin

    def selectMaxConnect(self, heatmap):
        labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)
        max_label = 0
        max_num = 0
        for i in range(1, num + 1):
            if np.sum(labeled_img == i) > max_num:
                max_num = np.sum(labeled_img == i)
                max_label = i
        lcc = (labeled_img == max_label)
        if max_num == 0:
            lcc = (labeled_img == -1)
        lcc = lcc + 0
        return lcc

    def Attention_gen_patchs(self, ori_image, fm_cuda, device):
        # 基于CNN特征图提取关键区域，裁剪并返回Patch作为后续输入
        # ori_image：原始输入图像，形状为[bz, C, H, W]
        #  fm_cuda: 由CNN提取的全局特征图，形状为[BZ, NC, H, W], 其中bz：batch_size, nc:通道数（feature map数量）， h w： feature map尺寸
        # fm => mask =>(+ ori-img) => crop = patchs
        feature_conv = fm_cuda.data.cpu().numpy()  # 将全局特征图转换为np
        size_upsample = (224, 224)  # 目标尺寸，用于后续的图像变换
        bz, nc, h, w = feature_conv.shape  # 获取特征图的形状

        patchs_cuda = torch.FloatTensor().to(device)  # 初始化，用于存储最终的patch数据

        # 遍历batch_size，处理每张特征图
        for i in range(0, bz):
            feature = feature_conv[i]
            # 计算Class Activation Map
            cam = feature.reshape((nc, h * w))  # 将 nc × h × w 变为 nc × (h*w)
            cam = cam.sum(axis=0)  # 对nc个通道求和，生成一个单通道的特征图
            cam = cam.reshape(h, w)  # 恢复形状
            # 归一化至0-256，并转换为uint8
            cam = cam - np.min(cam)  # 让最小值变为0
            cam_img = cam / np.max(cam)  # 让最大值变为1
            cam_img = np.uint8(255 * cam_img)  # 让其变成 0-255 的 uint8 格式，便于后续 OpenCV 处理。

            # 生成二值热力图
            heatmap_bin = self.binImage(cv2.resize(cam_img, size_upsample))  # 将cam放大后进行二值化处理
            heatmap_maxconn = self.selectMaxConnect(heatmap_bin)  # 提取最大连通区域
            heatmap_mask = heatmap_bin * heatmap_maxconn  # 最终得到的区域掩码（0/1）

            # 计算掩码区域的最小边界框
            ind = np.argwhere(heatmap_mask != 0)
            minh = min(ind[:, 0])
            minw = min(ind[:, 1])
            maxh = max(ind[:, 0])
            maxw = max(ind[:, 1])

            # 裁剪原始图像
            image = ori_image[i].cpu().numpy().reshape(224, 224, 3)  # Numpy不支持直接处理gpu上的tensor
            image = image[int(224 * 0.334):int(224 * 0.667), int(224 * 0.334):int(224 * 0.667), :]  # 取图像的中心区域

            # 进一步采集并预处理
            image = cv2.resize(image, size_upsample)
            image_crop = image[minh:maxh, minw:maxw, :] * 256  # 裁剪出感兴趣的区域，然后反归一化
            image_crop = self.preprocess(
                Image.fromarray(image_crop.astype('uint8')).convert('RGB'))  # 转换为 PIL.Image 格式，以便 preprocess() 处理

            img_variable = torch.autograd.Variable(image_crop.reshape(3, 224, 224).unsqueeze(0).to(device))

            # 凭借patchs
            patchs_cuda = torch.cat((patchs_cuda, img_variable), 0)

        return patchs_cuda

    def image_train(self):
        self.cnn_model.train()
        self.aux_model.train()
        self.fusion_model.train()
        self.model.train()
        total_step = len(self.dataloader['train'])

        tqdm_bar = tqdm(self.dataloader['train'], desc="Training")

        for iteration, (ids, image_id, imgs, input_ids, lm_labels, medterm_labels) in enumerate(tqdm_bar):
            imgs = imgs.to(self.device)
            input_ids = input_ids.to(self.device)
            lm_labels = lm_labels.to(self.device)
            medterm_labels = medterm_labels.to(self.device)

            self.rnn_NoamOpt.zero_grad()
            self.cnn_optimizer.zero_grad()

            # compute output
            output_global, fm_global, pool_global = self.cnn_model(imgs)

            patchs_var = self.Attention_gen_patchs(imgs, fm_global, self.device)

            output_local, _, pool_local = self.aux_model(patchs_var)
            # print(fusion_var.shape)
            output_fusion = self.fusion_model(pool_global, pool_local)

            med_porbs, findings_outputs = self.model(att_feats=output_fusion, input_ids=input_ids, input_type='img')
            # print("med_porbs shape: {}".format(med_porbs.shape))  # [16, 229]
            # print("medterm_labels shape: {}".format(medterm_labels.shape))  # [16, 50176]

            med_loss = self.med_crit(med_porbs, medterm_labels)
            caption_loss = self.outputs_crit(findings_outputs.view(-1, findings_outputs.size(-1)), lm_labels.view(-1))
            loss = 2.0 * med_loss + caption_loss

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.grad_clip)
            nn.utils.clip_grad_norm_(self.cnn_model.parameters(), self.opt.grad_clip)

            self.rnn_NoamOpt.step()
            self.cnn_optimizer.step()

            train_loss = loss.item()

            tqdm_bar.desc = "train_loss = {:.5f}, med_loss = {:.5f}, caption_loss = {:.5f}," \
                            "lr = {:.5f}, cnn_lr = {:.5f}" \
                .format(train_loss, med_loss, caption_loss,
                        self.rnn_NoamOpt.optimizer.param_groups[0]['lr'], self.opt.cnn_learning_rate)
