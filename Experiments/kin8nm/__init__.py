import numpy as np
import torch

from Experiments import AbstractRegressionSetup


experiment_name='Kin8nm'

input_dim = 8
sigma_noise = 0.1
seed = 42



class Setup(AbstractRegressionSetup): 
    def __init__(self, device, seed=seed):
        super(Setup, self).__init__()
        self.sigma_noise = sigma_noise
        self.seed=seed
        self.device = device

        self._preparare_data()
        self._split_holdout_data()
        self._normalize_data()
        self._flip_data_to_torch()

    def _preparare_data(self):
        self._X, _y = torch.load ('Experiments/kin8nm/data.pt')
        self._y = np.expand_dims(_y, axis=1)




        