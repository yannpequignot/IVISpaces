import torch
import torch.nn.functional as F
from torch import nn


def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma)
    return -(log_coeff + exponent).sum()


class MC_Dropout_Model(nn.Module):
    def __init__(self, input_dim, output_dim, no_units, init_sigma_noise, drop_prob, learn_noise, activation):
        super(MC_Dropout_Model, self).__init__()
        self.drop_prob=drop_prob
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = nn.Linear(input_dim, no_units)
        self.layer2 = nn.Linear(no_units, output_dim)

        self.activation = activation
        self._sigma_noise = nn.Parameter(torch.log(torch.tensor(init_sigma_noise).exp() - 1.), requires_grad=learn_noise)


    @property
    def sigma_noise(self):
        return torch.log(torch.exp(self._sigma_noise) + 1.)

    def forward(self, x):

        x = x.view(-1, self.input_dim)
        x = self.layer1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.drop_prob, training=True)
        x = self.layer2(x)
        return x


