import torch
from torch import nn
from Models import get_mlp, BigGenerator

class HyVI(nn.Module):
    def __init__(self, input_dim, layerwidth, nblayers, activation, init_sigma_noise, learn_noise, lat_dim):
        super(HyVI, self).__init__()
        self.param_count, self.predictor = get_mlp(input_dim, layerwidth, nblayers, activation)
        self.lat_dim=lat_dim
        self._sigma_noise = nn.Parameter(torch.log(torch.tensor(init_sigma_noise).exp() - 1.), requires_grad=learn_noise)

        self.gen_init()

    def gen_init(self):
        self.gen = BigGenerator(self.lat_dim, self.param_count)

    @property
    def sigma_noise(self):
        return torch.log(torch.exp(self._sigma_noise) + 1.)

    def forward(self, x, nb_predictors):
        theta = self.gen(nb_predictors)
        y_pred = self.predictor(x, theta)
        return y_pred