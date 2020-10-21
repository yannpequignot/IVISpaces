import torch
from torch import nn
import matplotlib.pyplot as plt

from Experiments import AbstractRegressionSetup

from Models import get_mlp

import numpy as np

from Tools import log_diagonal_mvn_pdf, NormalLogLikelihood

data_path = 'Experiments/foong/data/'

input_dim = 1
nblayers = 1
activation = nn.Tanh()  #
layerwidth = 50
sigma_noise = 0.1
seed = 42


class Setup(AbstractRegressionSetup):
    def __init__(self, device):
        super(Setup, self).__init__()
        self.sigma_noise = sigma_noise

        self.device = device
        self._preparare_data()
        self._flip_data_to_torch()

    def _preparare_data(self):
        train = torch.load(data_path + 'foong_train_out.pt')
        valid = torch.load(data_path + 'foong_validation_out.pt')
        test = torch.load(data_path + 'foong_test.pt')

        self._X_train, self._y_train = train[0].cpu(), train[1].unsqueeze(-1).cpu()
        self._X_validation, self._y_validation = valid[0].cpu(), valid[1].unsqueeze(-1).cpu()
        self._X_test, self._y_test = test[0].cpu(), test[1].unsqueeze(-1).cpu()
        self.n_train_samples = self._X_train.shape[0]

    # def _flip_data_to_torch(self):
    #     self._X_train = self._X_train.to(self.device).float()
    #     self._y_train = self._y_train.to(self.device).float()
    #     self._X_validation = self._X_validation.to(self.device).float()
    #     self._y_validation = self._y_validation.to(self.device).float()
    #     self._X_test = self._X_test.to(self.device).float()
    #     self._y_test = self._y_test.to(self.device).float()
    #     self.n_train_samples = self._X_train.shape[0]
