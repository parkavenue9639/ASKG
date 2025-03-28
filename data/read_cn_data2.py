import torch
from torch.utils.data import Dataset
from PIL import Image

import os
import pickle as pkl
import numpy as np


class CHXrayDataSet2(Dataset):
    def __init__(self, opt, split, transform=None):
        self.transform = transform

        self.data_dir = opt.data_dir
        # TODO
        self.pkl_dir = '../data/processed_chxray_precise_tag/'
        self.img_dir = os.path.join(self.data_dir, 'NLMCXR_png')

        self.num_medterm = opt.num_medterm  # 指定医学术语的数量

        with open(os.path.join(self.pkl_dir, 'align2.' + split + '.pkl'), 'rb') as f:
            self.findings = pkl.load(f)  # 影像对应的文本描述
            self.findings_labels = pkl.load(f)  # 文本描述的标签
            self.image = pkl.load(f)  # 影像文件名
            self.medterms = pkl.load(f)  # 影像对应的医学术语  length:1470

        f.close()

        with open(os.path.join(self.pkl_dir, 'word2idw.pkl'), 'rb') as f:
            self.word2idw = pkl.load(f)  # 单词到id的映射，用于文本处理
            # print(len(self.word2idw))  1779
            # print(self.word2idw)  {'word', id}
        f.close()

        with open(os.path.join(self.pkl_dir, 'idw2word.pkl'), 'rb') as f:
            self.idw2word = pkl.load(f)  # id到单词的映射，用于解码文本
            # print(len(self.idw2word)) 1779
            # print(self.idw2word)  {id: 'word}
        f.close()

        self.ids = list(self.image.keys())  # 获取所有的影像id
        self.vocab_size = len(self.word2idw)  # 词汇表大小

        print('CHXrayDataSet2 {} init complete， len : {}'.format(split, len(self.image)))
        self.checkdata()

    def __getitem__(self, index):
        ix = self.ids[index]  # 获取影像id
        image_id = self.image[ix]  # 获取影像文件名
        image_name = os.path.join(self.img_dir, image_id)  # 拼接完整路径
        img = Image.open(image_name).convert('RGB')  # 打开图像，转换为RGB（避免PIL处理灰度图时出错）
        if self.transform is not None:
            img = self.transform(img)  # 数据增强，对图像进行变换

        #print(img.size(), image_id)
        medterm_labels = np.zeros(229)
        # medterm_labels = np.zeros(self.num_medterm)
        medterms = self.medterms[ix]
        for medterm in medterms:
            # medterm_labels[medterm] = 1
            if medterm < self.num_medterm:
                medterm_labels[medterm] = 1
        # print("medterm_labels shape{}".format(medterm_labels.shape))  # (50176,)

        findings = self.findings[ix]
        findings_labels = self.findings_labels[ix]
        findings = np.array(findings)
        findings_labels = np.array(findings_labels)

        findings = torch.from_numpy(findings).long()
        findings_labels = torch.from_numpy(findings_labels).long()


        return ix, image_id, img, findings, findings_labels, torch.FloatTensor(medterm_labels)

    def __len__(self):
        return len(self.ids)

    def checkdata(self):
        """打印各种数据的前三个元素的类型、长度和元素内容"""
        data_attrs = {
            'findings': self.findings,
            'findings_labels': self.findings_labels,
            'image': self.image,
            'medterms': self.medterms,
            'word2idw': self.word2idw,
            'idw2word': self.idw2word,
            'ids': self.ids
        }

        for attr_name, attr_value in data_attrs.items():
            print(f"\nChecking {attr_name}:")
            if isinstance(attr_value, dict):
                sample_keys = list(attr_value.keys())[:5]
                for key in sample_keys:
                    print(
                        f"  Key: {key} | Type: {type(attr_value[key])} | Length: {len(attr_value[key]) if hasattr(attr_value[key], '__len__') else 'N/A'} | Value: {attr_value[key]}")
            elif isinstance(attr_value, list):
                for i, elem in enumerate(attr_value[:3]):
                    print(
                        f"  Index {i} | Type: {type(elem)} | Length: {len(elem) if hasattr(elem, '__len__') else 'N/A'} | Value: {elem}")
            else:
                print(
                    f"  Type: {type(attr_value)} | Length: {len(attr_value) if hasattr(attr_value, '__len__') else 'N/A'} | Sample Value: {attr_value}")
