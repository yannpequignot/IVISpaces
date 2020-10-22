import torch
from torch import nn
import math

class MeanFieldVariationalDistribution(nn.Module):
    def __init__(self, nb_dim, std_init=1., sigma=1.0, device='cpu'):
        super(MeanFieldVariationalDistribution, self).__init__()
        self.device = device
        self.nb_dim = nb_dim
        self.rho = nn.Parameter(torch.log(torch.exp(sigma * torch.ones(nb_dim, device=device)) - 1), requires_grad=True)
        self.mu = nn.Parameter(std_init * torch.randn(nb_dim, device=device), requires_grad=True)

    @property
    def sigma(self):
        return self._rho_to_sigma(self.rho)

    def forward(self, n=1):
        epsilon = torch.randn(size=(n, self.nb_dim)).to(self.device)
        return self.mu + self.sigma * epsilon

    def _rho_to_sigma(self, rho):
        sigma = torch.log(torch.exp(rho) + 1.)
        return sigma

    def log_prob(self, x):
        S = self.sigma
        mu = self.mu
        dim = self.nb_dim
        n_x = x.shape[0]
        H = S.view(1, 1, dim).pow(-1)
        d = ((x - mu.view(1, dim)) ** 2).view(n_x, dim)
        const = 0.5 * S.log().sum() + 0.5 * dim * torch.tensor(2 * math.pi).log()
        return -0.5 * (H * d).sum(2).squeeze() - const