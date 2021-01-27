import torch
from torch import nn
from .predictiveMLP import get_mlp
from .GenerativeNetwork import BigGenerator
from .MeanField import MeanFieldVariationalDistribution
from abc import ABC, abstractmethod


class VI(nn.Module, ABC):
    def __init__(self, input_dim, layerwidth, nblayers, activation, init_sigma_noise, learn_noise):
        super().__init__()
        self.param_count, self.predictor = get_mlp(input_dim, layerwidth, nblayers, activation)
        self._sigma_noise = nn.Parameter(torch.log(torch.tensor(init_sigma_noise).exp() - 1.),
                                         requires_grad=learn_noise)

        self._gen_init()

    @abstractmethod
    def _gen_init(self):
        pass

    @property
    def sigma_noise(self):
        return torch.log(torch.exp(self._sigma_noise) + 1.)


    def forward(self, x, nb_predictors):
        theta = self.gen(nb_predictors)
        y_pred = self.predictor(x, theta)
        return y_pred


class HyVI(VI):
    def __init__(self, input_dim, layerwidth, nblayers, activation, init_sigma_noise, learn_noise, lat_dim):
        self.lat_dim = lat_dim
        super().__init__(input_dim, layerwidth, nblayers, activation, init_sigma_noise, learn_noise)

    def _gen_init(self):
        self.gen = BigGenerator(self.lat_dim, self.param_count)

    @property
    def name(self):
        return 'HyVI'


class MFVI(VI):
    def __init__(self, input_dim, layerwidth, nblayers, activation, init_sigma_noise, learn_noise, lat_dim):
        self.lat_dim = lat_dim
        super().__init__(input_dim, layerwidth, nblayers, activation, init_sigma_noise, learn_noise)

    def _gen_init(self):
        self.gen = MeanFieldVariationalDistribution(self.param_count, std_init=1., sigma=0.001)
    @property
    def name(self):
        return 'MFVI'