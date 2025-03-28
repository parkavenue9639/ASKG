import opts
import pickle
from Train_process import *

# import eval_utils

import sys
import os

# 将项目根目录加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pretrain.models.cnn import Densenet121_AG, Fusion_Branch
from pretrain.models.SentenceImageModel import *
from data.data_process import DataProcess

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
'''from misc.utils import NoamOpt
from read_cn_data2 import get_loader_cn
from read_fh21_data import get_loader2
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score
from skimage.measure import label
import torchvision.transforms as transforms'''


def set_device():
    # choose device
    if torch.backends.mps.is_available() and os.name == 'posix' and 'darwin' in os.sys.platform:
        dev = torch.device('mps')
    else:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return dev


def parse_args():
    options = opts.parse_opt()
    print(options)
    config_json = OpenAIGPTConfig('../pretrain/config/cn_precise_config.json')
    options.vocab_size = config_json.vocab_size  # 26916
    return options, config_json


def init_tag_decoder(conf):
    with open(conf.tag_decoderdir, 'rb') as f:
        # data/processed_fh21_precise_tag/tagdecoder.pkl
        decoder = pickle.load(f)
    return decoder


def init_model(tag_decoder, device, opt, config):

    cnn_model = Densenet121_AG(pretrained=False, num_classes=opt.num_medterm).to(device)
    aux_model = Densenet121_AG(pretrained=False, num_classes=opt.num_medterm).to(device)
    fusion_model = Fusion_Branch(input_size=1024, output_size=opt.num_medterm, device=device).to(device)
    model = SentenceLMHeadModel(tag_decoder, config, device=device).to(device)
    return cnn_model, aux_model, fusion_model, model


def init_text_dataset(dataprocessor, split):
    return dataprocessor.get_text_loader(split)


def init_image_dataset(dataprocessor, split):
    return dataprocessor.get_image_loader(split)


def copy_every_model_to_avail_cuda(model_list, device):
    if device != torch.device('mps'):
        for i, mod in enumerate(model_list):
            mod = mod.to(device)  # 将模型转移到可用设备
            if torch.cuda.device_count() > 1:  # 如果有多个 GPU
                model_list[i] = nn.DataParallel(mod)  # 直接更新列表中的模型
            else:
                model_list[i] = mod  # 只有一个 GPU，无需并行
    return


def init_abstract_loader(dataprocessor):
    text_train_dataset, text_train_loader = init_text_dataset(dataprocessor, 'train')
    text_val_dataset, text_val_loader = init_text_dataset(dataprocessor, 'valid')
    text_test_dataset, text_test_loader = init_text_dataset(dataprocessor, 'train')
    return [text_train_loader, text_val_loader, text_test_loader]

def init_image_loader(dataprocessor):
    img_train_dataset, img_train_loader = init_image_dataset(dataprocessor, 'train')
    img_valid_dataset, img_valid_loader = init_image_dataset(dataprocessor, 'valid')
    img_test_dataset, img_test_loader = init_image_dataset(dataprocessor, 'train')
    return ([img_train_loader, img_valid_loader, img_test_loader],
            [img_train_dataset, img_valid_dataset, img_test_dataset])


if __name__ == '__main__':
    device = set_device()
    opt, config = parse_args()
    tag_decoder = init_tag_decoder(config)
    cnn_model, aux_model, fusion_model, model = init_model(tag_decoder, device, opt, config)
    copy_every_model_to_avail_cuda([cnn_model, aux_model, fusion_model, model], device)

    # 初始化数据处理类
    dataprocessor = DataProcess(opt)
    # 创建辅助信号数据集
    text_loader_list = init_abstract_loader(dataprocessor)

    # 创建训练处理类
    abstract_train = TextTrainProcess(device, opt, text_loader_list, model)
    abstract_train.train()

    # 创建图像数据集
    image_loader_list, image_dataset_list = init_image_loader(dataprocessor)

    # 创建训练处理类
    other_model = [cnn_model, aux_model, fusion_model]
    image_train = ImageTrain(device, opt, image_loader_list, image_dataset_list, model, other_model)
    image_train.train()

