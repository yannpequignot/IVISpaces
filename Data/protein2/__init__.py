import numpy as np
import torch

from Data import AbstractRegressionSetup

experiment_name='protein2'

#UCI repo
#https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure

input_dim = 9
sigma_noise = 4.4 #Yarin Gal McDropOut https://github.com/yaringal/DropoutUncertaintyExps


seed=42

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
        
    def ood_data(self):
        return self._X_ood, self._y_ood
    
    def _preparare_data(self):
        self._X, _y = torch.load ('Data/protein2/train.pt')
        self._y = np.expand_dims(_y, axis=1)
        self._X_ood, y_ood =  torch.load ('Data/protein2/test.pt')
        self._y_ood = np.expand_dims(y_ood, axis=1)
        self._X_ood = torch.as_tensor(self._X_ood).to(self.device).float()
        self._y_ood = torch.as_tensor(self._y_ood).to(self.device).float()
        
