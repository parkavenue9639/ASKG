from data.read_fh21_data import Fh21Dataset
from data.read_cn_data2 import CHXrayDataSet2
from torch.utils.data import DataLoader

import torchvision.transforms as transforms


class DataProcess:
    def __init__(self, opt):
        self.opt = opt

    def get_text_loader(self, split):
        dataset = Fh21Dataset(self.opt, split=split)
        loader = DataLoader(dataset, batch_size=self.opt.train_batch_size, shuffle=True, num_workers=16)
        return dataset, loader

    def get_image_loader(self, split):
        # 采用ImageNet的均值和标准差，这样可以加速模型收敛，因为常见的预训练模型都是用ImageNet数据集训练的，保持相同的归一化方式有助于更好的迁移学习
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])

        # 数据集封装
        dataset = CHXrayDataSet2(self.opt, split=split,  # split指定数据的类别
                                 transform=transforms.Compose([
                                     transforms.Resize((224, 224)),  # 将图像调整到224x224，适配CNN结构
                                     transforms.ToTensor(),  # 将图像转换为[C H W]格式的tensor，并归一化到[0,1]
                                     normalize  # 归一化方式
                                 ]))
        if split == 'train':
            loader = DataLoader(dataset=dataset, batch_size=self.opt.train_batch_size,
                                shuffle=True, num_workers=16)
        elif split == 'valid':
            loader = DataLoader(dataset=dataset, batch_size=self.opt.eval_batch_size,
                                shuffle=True, num_workers=16)
        elif split == 'test':
            loader = DataLoader(dataset=dataset, batch_size=self.opt.eval_batch_size,
                                shuffle=False, num_workers=16)
        else:
            raise Exception('DataLoader split must be train or val.')
        return dataset, loader
