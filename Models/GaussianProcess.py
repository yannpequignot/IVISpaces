import torch
from torch import nn
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
