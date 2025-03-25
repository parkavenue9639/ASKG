import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score

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
        if os.path.exists(os.path.join(self.checkpoint_path, 'check_point.pth')):
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state'])
            self.rnn_NoamOpt.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.start_epoch = checkpoint['epoch']
            self.best_val_score = checkpoint.get('best_val_score', None)
            print(f"Loaded checkpoint from {self.checkpoint_path}, starting from epoch {self.start_epoch}")
        else:
            self.start_epoch = 0
            self.best_val_score = None

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

    def train(self):
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.init_loss_func()
        self.check_checkpoint()

        for epoch in trange(self.start_epoch, int(self.opt.max_epochs), desc="Epoch"):
            try:
                self.abstract_train()
            except Exception as e:
                print("Error during training: {}".format(e))

            if epoch % self.opt.val_every_epoch == 0:
                current_score = self.eval()
                if self.device.type == 'mps':
                    print("free mps memory")
                    torch.mps.empty_cache()
                else:
                    torch.cuda.empty_cache()

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
