import numpy as np
import math
import torch
from torch import nn

import pandas as pd

from datetime import datetime

from torch.utils.data import Dataset

from Models import get_mlp, BigGenerator, MeanFieldVariationalDistribution, GaussianProcess, MC_Dropout_Wrapper
from Tools import average_normal_loglikelihood, log_diagonal_mvn_pdf
from Metrics import kl_nne, evaluate_metrics, entropy_nne, batch_entropy_nne

from Experiments import get_setup

from Inference.VI_trainer import IVI

from tqdm import trange

import timeit
import os

def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))



## Hyperparameters ##

#predictive model
layerwidth=50
nblayers=1
activation=nn.ReLU()

#generative model
lat_dim=5

#optimizer
learning_rate=0.005

#scheduler
patience=30
lr_decay=.7#.7
min_lr= 0.0001
n_epochs=2000#5000#2000


#loss hyperparameters
n_samples_LL=100 #nb of predictor samples for average LogLikelihood

n_samples_KL=500 #nb of predictor samples for KL divergence
kNNE=1 #k-nearest neighbour

batch_size=50

sigma_prior=.5# TO DO check with other experiments setup.sigma_prior    


input_sampling='uniform' #'uniform', 'uniform+data'


def OOD_sampler(x_train,n_ood):
    M = x_train.max(0, keepdim=True)[0]
    m = x_train.min(0, keepdim=True)[0]
    X = torch.rand(n_ood,x_train.shape[1]).to(device) * (M-m) + m                           
    return X


def FuNNeVI(dataset,device):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
    def prior(n):
        return sigma_prior*torch.randn(size=(n,param_count), device=device)
    

    
    if input_sampling=='uniform':
        def input_sampler(x_data):
            n_ood=200
            M = x_train.max(0, keepdim=True)[0]
            m = x_train.min(0, keepdim=True)[0]
            X = torch.rand(n_ood,input_dim).to(device) * (M-m) + m                           
            return X
    
 
    
    def projection(theta0,theta1, x_data):
        X=input_sampler(x_data)
        #compute projection on both paramters with model
        theta0_proj=model(X, theta0).squeeze(2)
        theta1_proj=model(X, theta1).squeeze(2)
        return theta0_proj, theta1_proj

    def kl(x_data, GeN):

        theta=GeN(n_samples_KL) #variationnel
        theta_prior=prior(n_samples_KL) #prior

        theta_proj, theta_prior_proj = projection(theta, theta_prior,x_data)

        K= kl_nne(theta_proj, theta_prior_proj, k=kNNE)
        return K
    
    
    
    def ELBO(x_data, y_data, GeN, _sigma_noise):
        y_pred=model(x_data,GeN(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood=average_normal_loglikelihood(y_pred, y_data, sigma_noise)
        the_KL=kl(x_data, GeN)
        the_ELBO= - Average_LogLikelihood+ (len(x_data)/size_data)* the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise

    #generative model
    GeN = BigGenerator(lat_dim,param_count,device).to(device)

    ## Parametrize noise for learning aleatoric uncertainty
    
    _sigma_noise=torch.log(torch.tensor(setup.sigma_noise).exp()-1.).clone().to(device).detach().requires_grad_(False)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    optimizer = torch.optim.Adam(GeN.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, min_lr=min_lr)

    Run=IVI(train_loader, ELBO, optimizer)
    
    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc=dataset+'/FuNNeVI', refresh=False)
        for t in tr:
            
            
            scores=Run.one_epoch(GeN, _sigma_noise)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'], sigma=scores['sigma'])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start
    
    theta=GeN(1000).detach()
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach()

    X=[x_train,x_test,OOD_sampler(x_train,1000)]
    target=[y_train,y_test]
    Y_=[model(X[i],theta) for i in range(len(X))]
    Y=[y+sigma_noise*torch.randn_like(y) for y in Y_]
    H=[batch_entropy_nne(y.transpose(0, 1), k=30) for y in Y]
    return H, theta


def GeNNeVI(dataset,device):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()
    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
    def prior(n):
        return sigma_prior*torch.randn(size=(n,param_count), device=device)
    
    def kl(x_data, GeN):

        theta=GeN(n_samples_KL) #variationnel
        theta_prior=prior(n_samples_KL) #prior

        K= kl_nne(theta, theta_prior, k=kNNE)
        return K
    
    def ELBO(x_data, y_data, GeN, _sigma_noise):
        y_pred=model(x_data,GeN(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood=average_normal_loglikelihood(y_pred, y_data, sigma_noise)
        the_KL=kl(x_data, GeN)
        the_ELBO= - Average_LogLikelihood+ (len(x_data)/size_data)* the_KL#(len(x_data)/size_data)*the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise

    #generative model
    GeN = BigGenerator(lat_dim,param_count,device).to(device)

    ## Parametrize noise for learning aleatoric uncertainty
    
    _sigma_noise=torch.log(torch.tensor(setup.sigma_noise).exp()-1.).clone().to(device).detach().requires_grad_(False)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    optimizer = torch.optim.Adam(GeN.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, min_lr=min_lr)

    Run=IVI(train_loader, ELBO, optimizer)
    
    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc=dataset+'/GeNNeVI', refresh=False)
        for t in tr:

            
            scores=Run.one_epoch(GeN, _sigma_noise)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'], sigma=scores['sigma'])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start
    
    theta=GeN(1000).detach()
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach()

    X=[x_train,x_test,OOD_sampler(x_train,1000)]
    target=[y_train,y_test]
    Y_=[model(X[i],theta) for i in range(len(X))]
    Y=[y+sigma_noise*torch.randn_like(y) for y in Y_]
    H=[batch_entropy_nne(y.transpose(0, 1), k=30) for y in Y]
    return H, theta

models_HMC = torch.load('Results/HMC_models.pt')

def HMC(dataset,device):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()
    
 
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
 
    HMC_=models_HMC[dataset]
    indices = torch.randperm(len(HMC_))[:1000]
    theta=HMC_[indices].to(device)
    sigma_noise = torch.tensor(setup.sigma_noise)
    
    X=[x_train,x_test,OOD_sampler(x_train,1000)]
    target=[y_train,y_test]
    Y_=[model(X[i],theta) for i in range(len(X))]
    Y=[y+sigma_noise*torch.randn_like(y) for y in Y_]
    H=[batch_entropy_nne(y.transpose(0, 1), k=30) for y in Y]
    return H, theta


if __name__ == "__main__":
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")
    file_name='Results/NEW/UCI_ENTROPY_Fixed_Noise'+date_string
    makedirs(file_name)

    with open(file_name, 'w') as f:
        script=open(__file__)
        f.write(script.read())  

    datasets= ['boston','concrete', 'energy', 'wine', 'yacht']#'powerplant',
    ENTROPY={dataset:{} for dataset in datasets}#torch.load('Results/NEW/UCI_ENTROPY2020-10-08-12:13.pt')#{dataset:{} for dataset in datasets}#torch.load('Results/NEW/UCI_ENTROPY2020-10-07-15:13.pt')#

    
    for dataset in datasets:
        print(dataset)     
 
        entropies={}
    
                       
        H, theta=GeNNeVI(dataset,device)
        entropies.update({'GeNNeVI':(H, theta)})
        
        H, theta=FuNNeVI(dataset,device)
        entropies.update({'FuNNeVI':(H, theta)})
        
        H, theta=HMC(dataset,device)
        entropies.update({'HMC':(H, theta)})
        
        ENTROPY[dataset].update(entropies)
        
        torch.save(ENTROPY,file_name+'.pt')
