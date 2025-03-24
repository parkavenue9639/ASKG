from data.read_fh21_data import Fh21Dataset
from torch.utils.data import DataLoader


class DataProcess:
    def __init__(self, opt):
        self.opt = opt

    def get_text_loader(self, split):
        dataset = Fh21Dataset(self.opt, split=split)
        loader = DataLoader(dataset, batch_size=self.opt.train_batch_size, shuffle=True, num_workers=16)
        return dataset, loader

