import torch
from torch import nn


class BigGenerator(nn.Module):
    def __init__(self, lat_dim, output_dim, device):
        super(BigGenerator, self).__init__()
        self.lat_dim = lat_dim
        self.device = device
        self.output_dim = output_dim

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(
                nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(lat_dim, 4 * lat_dim),
            *block(4 * lat_dim, 8 * lat_dim),
            nn.Linear(8 * lat_dim, output_dim)
        )

    def forward(self, n=1):
        epsilon = torch.randn(size=(n, self.lat_dim), device=self.device)
        return self.model(epsilon)
