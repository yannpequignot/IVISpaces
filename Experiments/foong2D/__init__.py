import torch
from torch import nn
import matplotlib.pyplot as plt

from Experiments import AbstractRegressionSetup

from Models import get_mlp

import numpy as np

from Tools import log_diagonal_mvn_pdf, NormalLogLikelihood

experiment_name = 'Foong2D'
data_path='Experiments/foong2D/Data/'

input_dim = 2
nblayers = 1
activation = nn.Tanh()#
layerwidth = 50
sigma_noise=0.1
seed = 42
sigma_prior=.5

class Setup(AbstractRegressionSetup):  
    def __init__(self, device, layerwidth=layerwidth, nblayers=nblayers):
        super(Setup, self).__init__()
        self.experiment_name = experiment_name
        self.sigma_noise = sigma_noise
        self.sigma_prior=sigma_prior
        self.plot = True
        
        self.input_dim=input_dim
        self.device = device
        self.param_count, self._model = get_mlp(input_dim, layerwidth, nblayers, activation)
        self._preparare_data()
        #self._normalize_data()
        self._flip_data_to_torch()
        

    def _preparare_data(self):
        train = torch.load(data_path + 'foong_2D_train.pt')
        test = torch.load(data_path + 'foong_2D_test.pt')
        
        self._X_train, self._y_train = train[0].cpu(), train[1].unsqueeze(-1).cpu()
        self._X_test, self._y_test = test[0].cpu(), test[1].unsqueeze(-1).cpu()
        self.n_train_samples=self._X_train.shape[0]

        
        

    
    def loss(self,theta, R):
        y_pred = self._normalized_prediction(self._X_train, theta, self.device)  # MxNx1 tensor
        assert y_pred.shape[1] == self._y_train.shape[0]
        assert y_pred.shape[2] == self._y_train.shape[1]
        assert self._y_train.shape[1] == 1
        B = y_pred.shape[0]
        S = y_pred.shape[1]
        d = torch.tanh(R*(y_pred.view(B, S, 1) - self._y_train.view(1, S, 1)) ** 2)
        return d.mean(1)
    
    def sqloss(self,theta):
        y_pred = self._normalized_prediction(self._X_train, theta, self.device)  # MxNx1 tensor
        assert y_pred.shape[1] == self._y_train.shape[0]
        assert y_pred.shape[2] == self._y_train.shape[1]
        assert self._y_train.shape[1] == 1
        B = y_pred.shape[0]
        S = y_pred.shape[1]
        d = (y_pred.view(B, S, 1) - self._y_train.view(1, S, 1)) ** 2
        return d.mean(1)

    def logprior(self, theta):
        return  self._logprior(theta)
    
    def projection(self,theta0,theta1, n_samples, ratio_ood):
        #compute size of both samples
        #n_samples=self.n_train_samples
        n_ood=int(ratio_ood*n_samples)
        n_id=n_samples-n_ood
        
        #batch sample from train
        index=torch.randperm(self._X_train.shape[0])
        X_id=self._X_train[index][0:n_id]
          
        #batch sample OOD    
        M=2.
        m=-2.
        X_ood = torch.rand(n_ood,self.input_dim).to(self.device) * (M-m) + m    

        # here is using a normal instead   
        #ood_samples=torch.Tensor(n_ood,input_dim).normal_(0.,3.).to(self.device)
        X=torch.cat([X_id, X_ood])
        
        #compute projection on both paramters with model
        theta0_proj=self._model(X, theta0).squeeze(2)
        theta1_proj=self._model(X, theta1).squeeze(2)
        return theta0_proj, theta1_proj
    
    
    def prediction(self,X,theta):
        y_pred=self._normalized_prediction(X, theta,self.device).squeeze(2)
        #theta_proj=self._normalized_prediction(self._X_train, theta, self.device).squeeze(2)
        return y_pred
    

        
    def _flip_data_to_torch(self): 
        self._X_train = self._X_train.to(self.device).float()
        self._y_train = self._y_train.to(self.device).float()
        self._X_test = self._X_test.to(self.device).float()
        self._y_test = self._y_test.to(self.device).float()
        self.n_train_samples=self._X_train.shape[0]



        
