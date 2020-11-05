import numpy as np

from Data import AbstractRegressionSetup
from sklearn.datasets import load_boston


experiment_name='Boston'

sigma_noise = 2.5
seed = 42

class Setup(AbstractRegressionSetup): 
    def __init__(self,  device, seed=seed):
        super(Setup, self).__init__()
        self.sigma_noise = sigma_noise
        self.seed=seed
        self.device = device

        self._preparare_data()
        self._split_holdout_data()
        self._normalize_data()
        self._flip_data_to_torch()
        

    def _preparare_data(self):
        self._X, _y = load_boston(return_X_y=True)
        self._y = np.expand_dims(_y, axis=1)
        

        