import numpy as np
import torch

from Data import AbstractRegressionSetup


sigma_noise = .5
seed = 42


class Setup(AbstractRegressionSetup): 
    def __init__(self, device, seed=seed):
        super(Setup, self).__init__()
        self.sigma_noise = sigma_noise
        self.seed=seed
        self.device = device

        self._preparare_data()
        self._normalize_data()
        self._flip_data_to_torch()

    def _preparare_data(self):
        self._X_train, y_train = torch.load ('Data/winequality2/train.pt')
        self._y_train = np.expand_dims(y_train, axis=1)
        self._X_test, y_test = torch.load ('Data/winequality2/test.pt')
        self._y_test = np.expand_dims(y_test, axis=1)