import torch
from torch.utils.data import Dataset

import os
import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split


class Fh21Dataset(Dataset):
    def __init__(self, opt, split='train'):
        
        # 使用绝对路径
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    'data/processed_fh21_precise_tag')
        self.num_medterm = opt.num_medterm  # 指定医学术语的数量

        # 读取pkl文件，该文件是一个pickel序列化的文件，存储了数据集的主要内容
        with open(os.path.join(self.data_dir, 'fh21.pkl'), 'rb') as f:
            self.data = pkl.load(f)

        # 该文件是一个word-to-id词典，将单词映射到唯一的整数id
        with open(os.path.join(self.data_dir, 'word2idw.pkl'), 'rb') as f:
            self.word2idw = pkl.load(f)

        # 创建反向词典，用于将id转换为单词
        self.idw2word = {v: k for k, v in self.word2idw.items()}

        # 获取词汇大小
        self.vocab_size = len(self.word2idw)

        train_data, test_data = train_test_split(self.data, test_size=0.3, random_state=42)
        valid_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
        if split == 'train':
            self.data = train_data
        elif split == 'valid':
            self.data = valid_data
        elif split == 'test':
            self.data = test_data
        else:
            raise ValueError('split must be train or valid or test')

        self.check_data()

        print("Fh21Dataset {} init complete, len : {}".format(split, len(self.data)))

    def __getitem__(self, index):
        # 用于获取index位置的数据，返回索引、摘要数据、摘要标签、医学术语标签
        ix = self.data[index][0]
        abstracts = self.data[index][1]
        abstracts = np.array(abstracts)

        abstracts_labels = self.data[index][2]
        abstracts_labels = np.array(abstracts_labels)

        # 转换为torch张量
        abstracts = torch.from_numpy(abstracts).long()
        abstracts_labels = torch.from_numpy(abstracts_labels).long()

        # 处理医学术语标签
        med_terms_labels = np.zeros(229)
        med_terms = self.data[index][3]
        for med_term in med_terms:
            # med_term_labels[med_term] = 1
            if med_term < self.num_medterm:
                # # 独热向量编码向量，表示该摘要包含哪些医学术语。
                med_terms_labels[med_term] = 1  # 表示该摘要设计med_term这个医学术语

        # 数据索引、转换为torch张量的摘要、摘要的标签、医学术语的独热编码量
        return ix, abstracts, abstracts_labels, torch.FloatTensor(med_terms_labels)

    def __len__(self):
        #  数据集长度
        return len(self.data)

    def check_data(self):
        print("Fh21Dataset Data 类型：", type(self.data))  # list
        if isinstance(self.data, list):
            print("Fh21Dataset Data 长度：", len(self.data))  # 29058
            print("Fh21Dataset Data 示例（前 1 个元素）：", self.data[0])

        print("Fh21Dataset word2idw 长度：", len(self.word2idw))
        print("Fh21Dataset idw2word 长度：", len(self.idw2word))
        print("词表示例：", dict(list(self.idw2word.items())[:5]))

        print("\n🔍 解码前 3 条数据（abstracts）：")
        for i, sample in enumerate(self.data[:3]):
            image_id, abstract_ids, label_ids, medterm_ids = sample
            decoded = self.decode_ids_with_transformer_logic([abstract_ids])[0]
            print(f"\n Sample {i + 1}: Image ID = {image_id}")
            print("  Raw IDs      :", abstract_ids)
            print("  Decoded Text :", decoded)

    def decode_ids_with_transformer_logic(self, batch_abstract_ids):
        """
        模拟模型解码逻辑，将一批 abstract ids（List[List[int]]）转换为字符串
        """
        decode_list = []
        for abstract_ids in batch_abstract_ids:
            words = []
            for token_id in abstract_ids:
                if token_id < 0:
                    continue
                token = self.idw2word.get(token_id, '<UNK>')
                if token in ['<BOS>', '<BLANK>', '<UNK>']:
                    continue
                if token == '<EOS>':
                    break
                words.append(token)
            decode_list.append(' '.join(words))
        return decode_list


