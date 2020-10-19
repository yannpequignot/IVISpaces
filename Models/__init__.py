import torch
from torch import nn

from .McDropOut import MC_Dropout_Wrapper

import numpy as np
import math
import torch
from torch import nn
from torch import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussianProcess(nn.Module):
    def __init__(self, mean, lengthscale, noise=0.05):
        super(GaussianProcess, self).__init__()
        self.ls=lengthscale
        self.mean=mean
        self.var=noise**2

    def covar_matrix(self,x):
        K=torch.cdist(x.div(self.ls),x.div(self.ls),p=2).pow_(2).div_(-2).exp_()
        return K+self.var*torch.eye(x.shape[0], device=x.device)##
    
    def log_prob(self,inputs,f):
        GP_inputs=MultivariateNormal(loc=self.mean*torch.ones(inputs.shape[0],device=f.device),\
                                     covariance_matrix=self.covar_matrix(inputs))
        return GP_inputs.log_prob(f).view(-1)
    
    def forward(self, inputs, n=1): 
        GP_inputs=MultivariateNormal(loc=self.mean*torch.ones(inputs.shape[0], device=inputs.device), \
                                     covariance_matrix=self.covar_matrix(inputs))
        return GP_inputs.sample((n,)).squeeze()


class MeanFieldVariationalDistribution(nn.Module):
    def __init__(self, nb_dim, std_init=1., sigma=1.0, device='cpu'):
        super(MeanFieldVariationalDistribution, self).__init__()
        self.device = device
        self.nb_dim = nb_dim
        self.rho = nn.Parameter(torch.log(torch.exp(sigma*torch.ones(nb_dim, device=device)) - 1), requires_grad=True)
        self.mu = nn.Parameter(std_init*torch.randn(nb_dim, device=device), requires_grad=True)

    @property
    def sigma(self):
        return self._rho_to_sigma(self.rho)
        
    def forward(self, n=1):
        epsilon = torch.randn(size=(n,self.nb_dim)).to(self.device)
        return self.mu + self.sigma *epsilon
    
    def _rho_to_sigma(self, rho):
        sigma = torch.log(torch.exp(rho)+1.)
        return sigma

    def log_prob(self, x):
        S = self.sigma
        mu = self.mu
        dim=self.nb_dim
        n_x=x.shape[0]
        H=S.view(1,1,dim).pow(-1)
        d=((x-mu.view(1,dim))**2).view(n_x,dim)
        const=0.5*S.log().sum()+0.5*dim*torch.tensor(2*math.pi).log()
        return -0.5*(H*d).sum(2).squeeze()-const

def get_mlp(input_dim, layerwidth, nblayers, activation):
    param_count = (input_dim+1)*layerwidth+(nblayers-1) * \
        (layerwidth**2+layerwidth)+layerwidth+1

    def mlp(x, theta, input_dim=input_dim, layerwidth=layerwidth, nb_layers=nblayers, activation=activation):
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
        mlp(x,theta,input_dim=input_dim,layerwidth=layerwidth,nb_layers=nblayers,activation=activation)

        """

        nb_theta = theta.shape[0]
        nb_x = x.shape[0]
        split_sizes = [input_dim*layerwidth]+[layerwidth] + \
            [layerwidth**2, layerwidth]*(nb_layers-1)+[layerwidth, 1]
        theta = theta.split(split_sizes, dim=1)
        input_x = x.view(nb_x, input_dim, 1)
        m = torch.matmul(theta[0].view(
            nb_theta, 1, layerwidth, input_dim), input_x)
        m = m.add(theta[1].reshape(nb_theta, 1, layerwidth, 1))
        m = activation(m)
        for i in range(nb_layers-1):
            m = torch.matmul(
                theta[2*i+2].view(-1, 1, layerwidth, layerwidth), m)
            m = m.add(theta[2*i+3].reshape(-1, 1, layerwidth, 1))
            m = activation(m)
        m = torch.matmul(
            theta[2*(nb_layers-1)+2].view(nb_theta, 1, 1, layerwidth), m)
        m = m.add(theta[2*(nb_layers-1)+3].reshape(nb_theta, 1, 1, 1))
        return m.squeeze(-1)
    return param_count, mlp


class GeNet(nn.Module):
            def __init__(self, lat_dim, nb_neur, output_dim,  activation, init_w, init_b, device):
                super(GeNet, self).__init__()
                self.lat_dim = lat_dim
                self.device=device
                self.output_dim=output_dim
                self.hnet=nn.Sequential(
                        nn.Linear(lat_dim,nb_neur),
                        activation,
                        #nn.Linear(nb_neur,nb_neur),
                        #activation,
                        nn.Linear(nb_neur,output_dim)
                        ).to(device)
                
                torch.nn.init.normal_(self.hnet[2].weight,mean=0., std=init_w)
                torch.nn.init.normal_(self.hnet[2].bias,mean=0., std=init_b)
    
            def forward(self, n=1):
                epsilon = torch.randn(size=(n,self.lat_dim), device=self.device)
                return self.hnet(epsilon)           

class GeNetEns(nn.Module):
    def __init__(self, nb_comp, lat_dim, layer_width, output_dim, activation, init_w, init_b, device):
        super(GeNetEns, self).__init__()
        self.device = device
        self.nb_comp=nb_comp
        self.output_dim=output_dim
        self.components= nn.ModuleList([GeNet(lat_dim,layer_width,output_dim,activation,init_w,init_b,device) for i in range(nb_comp)]).to(device)

        self._best_score = float('inf')
        self.d = torch.distributions.multinomial.Multinomial(probs=torch.ones(nb_comp))

    def sample(self, n=1):
        return torch.stack([self.components[c](n) for c in range(self.nb_comp)])

    def _save_best_model(self, score,epoch,ED,LP):
        if score < self._best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': self.state_dict(),
                'ELBO': score,
                'ED':ED,
                'LP':LP
            }, 'best.pt')
            self._best_score=score

    def _get_best_model(self):
        best= torch.load('best.pt')
        self.load_state_dict(best['state_dict'])
        return best['epoch'], best['ELBO'], best['ED'], best['LP']
    
    def forward(self, n=1):
        m = self.d.sample((n,))
        return torch.cat([self.components[c](int(m[c])) for c in range(len(self.components))])
    

class Generator(nn.Module):
    def __init__(self, lat_dim, output_dim, device):
        super(Generator, self).__init__()
        
        self.lat_dim = lat_dim
        self.device=device
        self.output_dim=output_dim
        
        self._best_score = float('inf')
        
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.ReLU(inplace=True))#nn.LeakyReLU(inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(lat_dim, 2*lat_dim),
            *block(2*lat_dim, 4*lat_dim),
            nn.Linear(4*lat_dim, output_dim)
        )
      
        
    def _save_best_model(self, score,epoch,ED,LP):
        if score < self._best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': self.state_dict(),
            }, 'best.pt')
            self._best_score=score

    def _get_best_model(self):
        best= torch.load('best.pt')
        self.load_state_dict(best['state_dict'])
        return best['epoch'], best['ELBO']

    def forward(self, n=1):
        epsilon = torch.randn(size=(n,self.lat_dim), device=self.device)
        return self.model(epsilon) 
    
class BigGenerator(nn.Module):
    def __init__(self, lat_dim, output_dim, device):
        super(BigGenerator, self).__init__()
        
        self.lat_dim = lat_dim
        self.device=device
        self.output_dim=output_dim
        
        self._best_score = float('inf')
        
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.ReLU(inplace=True)) #changed from inplace=True on October 8 2020 !!!! #nn.LeakyReLU(inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(lat_dim, 4*lat_dim),
            *block(4*lat_dim, 8*lat_dim),
            nn.Linear(8*lat_dim, output_dim)
        )
      

    def forward(self, n=1):
        epsilon = torch.randn(size=(n,self.lat_dim), device=self.device)
        return self.model(epsilon) 
    
class SLPGenerator(nn.Module):
    def __init__(self, lat_dim, output_dim, device):
        super(SLPGenerator, self).__init__()
        
        self.lat_dim = lat_dim
        self.device=device
        self.output_dim=output_dim
        
        self._best_score = float('inf')
        
        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.ReLU(inplace=True))#nn.LeakyReLU(inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(lat_dim, 50),
            nn.Linear(50, output_dim)
        )

        
    def _save_best_model(self, score,epoch,ED,LP):
        if score < self._best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': self.state_dict(),
                'ELBO': score,
            }, 'best.pt')
            self._best_score=score

    def _get_best_model(self):
        best= torch.load('best.pt')
        self.load_state_dict(best['state_dict'])
        return best['epoch'], best['ELBO']

    def forward(self, n=1):
        epsilon = torch.randn(size=(n,self.lat_dim), device=self.device)
        return self.model(epsilon)  

    

