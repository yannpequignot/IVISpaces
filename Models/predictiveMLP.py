import torch
from torch import nn

class mlp(nn.Module):
    def __init__(self, input_dim, layerwidth, nb_layers, activation):
        super(mlp, self).__init__()
        self.input_dim = input_dim
        self.layerwidth = layerwidth
        self.nb_layers = nb_layers
        self.activation = activation

    """
    Feedforward neural network used as the observation model for the likelihood

    Parameters:
        x (Tensor): Input of the network of size NbExemples X NbDimensions
        theta (Tensor):  M set of parameters of the network NbModels X NbParam
        input_dim (Int): dimensions of NN's inputs (=NbDimensions)
        layerwidth (Int): Number of hidden units per layer
        nb_layers (Int): Number of layers
        activation (Module/Function): activation function of the neural network

    Returns:
        Predictions (Tensor) with dimensions NbModels X NbExemples X NbDimensions

    Example:

    input_dim=11
    nblayers = 2
    activation=nn.Tanh()
    layerwidth = 20
    param_count = (input_dim+1)*layerwidth+(nblayers-1)*(layerwidth**2+layerwidth)+layerwidth+1

    x=torch.rand(3,input_dim)
    theta=torch.rand(5,param_count)
    model=mlp(input_dim=input_dim,layerwidth=layerwidth,nb_layers=nblayers,activation=activation)
    model(x,theta)
    """

    def forward(self, x, theta):
        nb_theta = theta.shape[0]
        nb_x = x.shape[0]
        split_sizes = [self.input_dim * self.layerwidth] + [self.layerwidth] + \
                      [self.layerwidth ** 2, self.layerwidth] * (self.nb_layers - 1) + [self.layerwidth, 1]
        theta = theta.split(split_sizes, dim=1)
        input_x = x.view(nb_x, self.input_dim, 1)
        m = torch.matmul(theta[0].view(
            nb_theta, 1, self.layerwidth, self.input_dim), input_x)
        m = m.add(theta[1].reshape(nb_theta, 1, self.layerwidth, 1))
        m = self.activation(m)
        for i in range(self.nb_layers - 1):
            m = torch.matmul(
                theta[2 * i + 2].view(-1, 1, self.layerwidth, self.layerwidth), m)
            m = m.add(theta[2 * i + 3].reshape(-1, 1, self.layerwidth, 1))
            m = self.activation(m)
        m = torch.matmul(
            theta[2 * (self.nb_layers - 1) + 2].view(nb_theta, 1, 1, self.layerwidth), m)
        m = m.add(theta[2 * (self.nb_layers - 1) + 3].reshape(nb_theta, 1, 1, 1))
        return m.squeeze(-1)

def get_mlp(input_dim, layerwidth, nb_layers, activation):
    param_count = (input_dim+1)*layerwidth+(nb_layers-1) * \
        (layerwidth**2+layerwidth)+layerwidth+1
    return param_count, mlp(input_dim, layerwidth, nb_layers, activation)
