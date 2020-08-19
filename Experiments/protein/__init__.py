import numpy as np
import torch
from torch import nn

from Experiments import AbstractRegressionSetup

from Models import get_mlp

experiment_name='protein'

#UCI repo
#https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure

input_dim = 9
nblayers = 1
activation = nn.ReLU()
layerwidth = 100
sigma_noise = 4.4 #Yarin Gal McDropOut https://github.com/yaringal/DropoutUncertaintyExps
seed = 42
sigma_prior=0.5


class Setup(AbstractRegressionSetup): 
    def __init__(self, device, seed=seed):
        super(Setup, self).__init__()

        self.experiment_name = experiment_name
        
        self.sigma_noise = sigma_noise
        self.sigma_prior=sigma_prior
        self.seed=seed

        self.input_dim=input_dim
        
        self.plot = False

        self.device = device
        self.param_count, self._model = get_mlp(input_dim, layerwidth, nblayers, activation)

        self._preparare_data()
        self._split_holdout_data()
        self._normalize_data()
        self._flip_data_to_torch()

    def _preparare_data(self):
        self._X, _y = torch.load ('Experiments/protein/data.pt')
        self._y = np.expand_dims(_y, axis=1)

