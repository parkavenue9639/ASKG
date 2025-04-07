import torch
import os

from collections import OrderedDict


class TestProcess:

    def __init__(self, model, cnn_model):
        self.model = model
        self.cnn_model = cnn_model

    def modify_state_dict(self, state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module.' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v
        return new_state_dict

    def load_best_model(self):
        try:
            best_model_path = os.path.join('../best_model/image', 'best_model.pth')
            best_cnn_model_path = os.path.join('../best_model/image', 'best_cnn_model.pth')

            model = torch.load(best_model_path)
            new_model_state = self.modify_state_dict(model)
            cnn_model = torch.load(best_cnn_model_path)
            new_cnn_model_state = self.modify_state_dict(cnn_model)
            self.model.load_state_dict(new_model_state)
            self.cnn_model.load_state_dict(new_cnn_model_state)
            print("load best model successfully")
        except Exception as e:
            print("fail to load best model")
