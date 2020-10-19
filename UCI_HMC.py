import numpy as np
import math
import torch
from torch import nn

import pandas as pd

from datetime import datetime

from torch.utils.data import Dataset

from Models import get_mlp, BigGenerator, MeanFieldVariationalDistribution, GaussianProcess, MC_Dropout_Wrapper
from Tools import AverageNormalLogLikelihood, logmvn01pdf
from Metrics import KL, evaluate_metrics, Entropy

from Experiments import get_setup

from Inference.IVI_noise import IVI

from tqdm import trange

import itertools

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


sigma_prior=.5# TO DO check with other experiments setup.sigma_prior    

models_HMC = torch.load('Results/HMC_models.pt')

input_sampling='uniform' #'uniform', 'uniform+data'

def OOD_sampler(x_train,n_ood):
    M = x_train.max(0, keepdim=True)[0]
    m = x_train.min(0, keepdim=True)[0]
    X = torch.rand(n_ood,x_train.shape[1]).to(device) * (M-m) + m                           
    return X


def MFVI(dataset,device):
    
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation) 
    
    MFVI=MeanFieldVariationalDistribution(param_count, std_init=0.1 ,sigma=0.001, device=device)    

    _sigma_noise=torch.log(torch.tensor(setup.sigma_noise).exp()-1.).clone().to(device).detach().requires_grad_(False)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    def ELBO(x_data, y_data, MFVI, _sigma_noise):
        y_pred=model(x_data,MFVI(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood=AverageNormalLogLikelihood(y_pred, y_data, sigma_noise)
        theta=MFVI(n_samples_KL)
        the_KL=MFVI.log_prob(theta).mean()-logmvn01pdf(theta,sigma_prior).mean()
        the_ELBO= - Average_LogLikelihood+ (len(x_data)/size_data)* the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise
    
    optimizer = torch.optim.Adam(MFVI.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2*patience, factor=lr_decay, min_lr=min_lr)
    Run=IVI(train_loader, ELBO, optimizer)

    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc=dataset+'/MFVI', refresh=False)
        for t in tr:
            scores=Run.run(MFVI,_sigma_noise)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'], sigma=scores['sigma'])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start

    theta=MFVI(1000).detach()
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach().cpu()
    y_pred=model(x_test,theta)
    metrics=get_metrics(y_pred, sigma_noise, y_test, std_y_train, 'MFVI', time)
    return theta, metrics

def FuNNeMFVI(dataset,device):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
    def prior(n):
        return sigma_prior*torch.randn(size=(n,param_count), device=device)
    

    if input_sampling=='mixture':
        def input_sampler(x_data):
            n_ood=1
            M = x_train.max(0, keepdim=True)[0]
            m = x_train.min(0, keepdim=True)[0]
            X_rand = torch.rand(n_ood,input_dim).to(device) * (M-m) + m                            
            X=torch.cat([x_data,\
                         x_data+0.1*torch.randn_like(x_data),\
                         x_data+0.1*torch.randn_like(x_data),\
                         x_data+0.1*torch.randn_like(x_data),
                         X_rand])   
            return X
    
    if input_sampling=='uniform':
        def input_sampler(x_data):
            n_ood=200
            M = x_train.max(0, keepdim=True)[0]
            m = x_train.min(0, keepdim=True)[0]
            X = torch.rand(n_ood,input_dim).to(device) * (M-m) + m                           
            return X
    
    if input_sampling=='uniform+data':
        def input_sampler(x_data,n_ood=150):
            M = x_train.max(0, keepdim=True)[0]
            m = x_train.min(0, keepdim=True)[0]
            X_rand = torch.rand(n_ood,input_dim).to(device) * (M-m) + m                           
            return torch.cat([x_data,X_rand])
    
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

        K=KL(theta_proj, theta_prior_proj,k=kNNE,device=device)
        return K
    
    
    
    def ELBO(x_data, y_data, GeN, _sigma_noise):
        y_pred=model(x_data,GeN(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood=AverageNormalLogLikelihood(y_pred, y_data, sigma_noise)
        the_KL=kl(x_data, GeN)
        the_ELBO= - Average_LogLikelihood+ (len(x_data)/size_data)* the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise

    #generative model
    MFVI=MeanFieldVariationalDistribution(param_count, std_init=0. ,sigma=0.001, device=device)    

    ## Parametrize noise for learning aleatoric uncertainty
    
    _sigma_noise=torch.log(torch.tensor(setup.sigma_noise).exp()-1.).clone().to(device).detach().requires_grad_(False)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    optimizer = torch.optim.Adam(MFVI.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, min_lr=min_lr)

    Run=IVI(train_loader, ELBO, optimizer)
    
    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc=dataset+'/FuNNeMFVI', refresh=False)
        for t in tr:
            
            
            scores=Run.run(MFVI,_sigma_noise)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'], sigma=scores['sigma'])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start
    
    theta=MFVI(1000).detach()
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach().cpu()
    y_pred=model(x_test,theta)
    metrics=get_metrics(y_pred, sigma_noise, y_test, std_y_train, 'FuNNeMFVI', time)
    return theta, metrics


def FuNNeVI(dataset,device):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
    def prior(n):
        return sigma_prior*torch.randn(size=(n,param_count), device=device)
    

    if input_sampling=='mixture':
        def input_sampler(x_data):
            n_ood=1
            M = x_train.max(0, keepdim=True)[0]
            m = x_train.min(0, keepdim=True)[0]
            X_rand = torch.rand(n_ood,input_dim).to(device) * (M-m) + m                            
            X=torch.cat([x_data,\
                         x_data+0.1*torch.randn_like(x_data),\
                         x_data+0.1*torch.randn_like(x_data),\
                         x_data+0.1*torch.randn_like(x_data),
                         X_rand])   
            return X
    
    if input_sampling=='uniform':
        def input_sampler(x_data):
            n_ood=200
            M = x_train.max(0, keepdim=True)[0]
            m = x_train.min(0, keepdim=True)[0]
            X = torch.rand(n_ood,input_dim).to(device) * (M-m) + m                           
            return X
    
    if input_sampling=='uniform+data':
        def input_sampler(x_data,n_ood=150):
            M = x_train.max(0, keepdim=True)[0]
            m = x_train.min(0, keepdim=True)[0]
            X_rand = torch.rand(n_ood,input_dim).to(device) * (M-m) + m                           
            return torch.cat([x_data,X_rand])
    
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

        K=KL(theta_proj, theta_prior_proj,k=kNNE,device=device)
        return K
    
    
    
    def ELBO(x_data, y_data, GeN, _sigma_noise):
        y_pred=model(x_data,GeN(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood=AverageNormalLogLikelihood(y_pred, y_data, sigma_noise)
        the_KL=kl(x_data, GeN)
        the_ELBO= - Average_LogLikelihood+ (len(x_data)/size_data)* the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise

    #generative model
    GeN = BigGenerator(lat_dim,param_count,device).to(device)

    ## Parametrize noise for learning aleatoric uncertainty
    
    _sigma_noise=torch.log(torch.tensor(setup.sigma_noise).exp()-1.).clone().to(device).detach().requires_grad_(False)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    optimizer = torch.optim.Adam(list(GeN.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, min_lr=min_lr)

    Run=IVI(train_loader, ELBO, optimizer)
    
    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc=dataset+'/FuNNeVI', refresh=False)
        for t in tr:
            
            
            scores=Run.run(GeN,_sigma_noise)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'], sigma=scores['sigma'])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start
    
    
    
    theta=GeN(1000).detach()
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach().cpu()
    y_pred=model(x_test,theta)
    metrics=get_metrics(y_pred, sigma_noise, y_test, std_y_train, 'FuNNeVI', time)
    return theta, metrics


def GeNNeVI(dataset,device):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()
    
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

        K=KL(theta, theta_prior,k=kNNE,device=device)
        return K
    
    def ELBO(x_data, y_data, GeN, _sigma_noise):
        y_pred=model(x_data,GeN(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood=AverageNormalLogLikelihood(y_pred, y_data, sigma_noise)
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
            
            scores=Run.run(GeN,_sigma_noise)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'], sigma=scores['sigma'])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start
    
    theta=GeN(1000).detach()
    y_pred=model(x_test,theta)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach().cpu()
    metrics=get_metrics(y_pred, sigma_noise, y_test, std_y_train, 'GeNNeVI', time)
    return theta, metrics

def get_metrics(y_pred, sigma_noise, y_test, std_y_train, method, time, noise=True):
    metrics=evaluate_metrics(y_pred, sigma_noise.view(1,1,1), y_test,  std_y_train, device='cpu', std=False, noise=noise)
    metrics.update({'time [s]': time})
    metrics.update({'std noise': sigma_noise.item()})
    return metrics

def MeanStd(metric_list, method):
    df=pd.DataFrame(metric_list)
    mean=df.mean().to_dict()
    std=df.std().to_dict()
    metrics=list(mean.keys())
    for j in metrics:
        mean[(method,j)] = mean.pop(j)
        std[(method,j)] = std.pop(j)
    return mean, std

def FunKL(s, t, model, sampler, n=100):
    assert t.shape == s.shape
    KLs = torch.Tensor(n)
    for i in range(n):
        rand_input=sampler()
        t_=model(rand_input,t).squeeze(2)
        s_=model(rand_input,s).squeeze(2)
        k=1
        K= KL(t_, s_, k=k, device=device)     
        while torch.isinf(K):
            k+=1
            K= KL(t_, s_, k=k, device=device)
        KLs[i]=K
    return K.mean()  # , K.std()

def FunH(s, model, sampler, n=100):
    Hs = torch.Tensor(n)
    for i in range(n):
        rand_input=sampler()
        s_=model(rand_input,s).squeeze(2)
        k=1
        H= Entropy(s_,k=1,k_MC=200,device=s.device)     
        while torch.isinf(H):
            k+=1
            H= Entropy(s_,k=1,k_MC=200,device=s.device)     
        Hs[i]=H
    return H.mean()  # , K.std()

def ComputeEntropy(thetas, dataset, method):
    setup_ = get_setup(dataset)
    device=thetas[0].device
    setup=setup_.Setup(device) 

    entropies={}
    entropies_std={}
    x_train, y_train=setup.train_data()

    sampler= lambda :OOD_sampler(x_train=x_train,n_ood=200)

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
    HMC_=models_HMC[dataset]
    indices = torch.randperm(len(HMC_))[:1000]
    HMC=HMC_[indices].to(device)#models_HMC[dataset][:thetas[0].shape[0],:].to(device)     
    
    metric=(method,'paramH')
    print(dataset+': '+'paramH')
    Hs=[]
    for theta in thetas:
        H= Entropy(theta,k=1,k_MC=1,device=theta.device)     
        print(H.item())
        Hs.append(H.item())
    
    entropies.update({metric:np.mean(Hs)})
    entropies_std.update({metric:np.std(Hs)})
    
    metric=(method,'funH')
    print(dataset+': '+'funH')
    Hs=[]
    for theta in thetas:
        H= FunH(theta,model,sampler)     
        print(H.item())
        Hs.append(H.item())

    entropies.update({metric:np.mean(Hs)})
    entropies_std.update({metric:np.std(Hs)})
    
    return entropies, entropies_std

        
def paramCompareWithHMC(thetas, dataset, method):
    divergences={}
    divergences_std={}
    setup_ = get_setup(dataset)
    device=thetas[0].device
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
    HMC_=models_HMC[dataset]
    indices = torch.randperm(len(HMC_))[:1000]
    HMC=HMC_[indices].to(device)#models_HMC[dataset][:thetas[0].shape[0],:].to(device)        

    
    metric='KL(-,HMC)'
    print(dataset+': '+metric)
    KLs=[]
    for theta in thetas:
        K=KL(theta,HMC, k=kNNE,device=device)
        print(K.item())
        KLs.append(K.item())
    
    divergences.update({metric:np.mean(KLs)})
    divergences_std.update({metric:np.std(KLs)})
    
    metric='KL(HMC,-)'
    print(dataset+': '+metric)
    KLs=[]
    for theta in thetas:
        K=KL(HMC,theta, k=kNNE,device=device)
        print(K.item())
        KLs.append(K.item())
    
    divergences.update({metric:np.mean(KLs)})
    divergences_std.update({metric:np.std(KLs)})
    
    metric='KL(-,-)'
    print(dataset+': '+metric)
    KLs=[]
        
    models_pairs=list(itertools.combinations(thetas,2))
    KLs=[]
    for theta_0,theta_1 in models_pairs:
        K=KL(theta_0,theta_1,k=kNNE,device=device)
        print(K.item())
        KLs.append(K.item())    
   
    divergences.update({metric:np.mean(KLs)})
    divergences_std.update({metric:np.std(KLs)})
        
    metrics=list(divergences.keys())
    for j in metrics:
        divergences[(method,j)] = divergences.pop(j)
        divergences_std[(method,j)] = divergences_std.pop(j)
    return divergences, divergences_std


def CompareWithHMC(thetas, dataset, method):
    divergences={}
    divergences_std={}
    setup_ = get_setup(dataset)
    setup=setup_.Setup(thetas[0].device) 

    x_train, y_train=setup.train_data()
    sampler= lambda :OOD_sampler(x_train=x_train,n_ood=200)
    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
    HMC_=models_HMC[dataset]
    indices = torch.randperm(len(HMC_))[:1000]
    HMC=HMC_[indices].to(thetas[0].device)#models_HMC[dataset][:thetas[0].shape[0],:].to(device)        

    
    metric='KL(-,HMC)'
    print(dataset+': '+metric)
    KLs=[]
    for theta in thetas:
        K=FunKL(theta,HMC,model,sampler)
        print(K.item())
        KLs.append(K.item())
    
    divergences.update({metric:np.mean(KLs)})
    divergences_std.update({metric:np.std(KLs)})
    
    metric='KL(HMC,-)'
    print(dataset+': '+metric)
    KLs=[]
    for theta in thetas:
        K=FunKL(HMC,theta,model,sampler)
        print(K.item())
        KLs.append(K.item())
    
    divergences.update({metric:np.mean(KLs)})
    divergences_std.update({metric:np.std(KLs)})
    
    metric='KL(-,-)'
    print(dataset+': '+metric)
    KLs=[]
        
    models_pairs=list(itertools.combinations(thetas,2))
    KLs=[]
    for theta_0,theta_1 in models_pairs:
        K=FunKL(theta_0,theta_1,model,sampler)
        print(K.item())
        KLs.append(K.item())    
   
    divergences.update({metric:np.mean(KLs)})
    divergences_std.update({metric:np.std(KLs)})
    
    metrics=list(divergences.keys())
    for j in metrics:
        divergences[(method,j)] = divergences.pop(j)
        divergences_std[(method,j)] = divergences_std.pop(j)
    return divergences, divergences_std

def HMC_metrics(dataset,device):
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 
    
    x_test, y_test=setup.test_data()

    ## predictive model
    input_dim=x_test.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()
    
    HMC_=models_HMC[dataset]
    indices = torch.randperm(len(HMC_))[:1000]
    HMC=HMC_[indices].to(device)
    sigma_noise=torch.tensor(setup.sigma_noise)
    y_pred=model(x_test,HMC)
    
    metrics=get_metrics(y_pred, sigma_noise, y_test, std_y_train, 'HMC', 0.)
    metrics_keys=list(metrics.keys())
    for j in metrics_keys:
        metrics[('HMC',j)] = metrics.pop(j)
    return metrics
    
#metrics[(method,metric)].update({dataset:(np.mean(KLs).round(decimals=3), np.std(KLs).round(decimals=3))})

if __name__ == "__main__":
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")
    file_name='Results/NEW/UCI_HMC'+date_string
    makedirs(file_name)

    with open(file_name, 'w') as f:
        script=open(__file__)
        f.write(script.read())  

    ## small 
    batch_size=50
    datasets=['boston','concrete', 'energy', 'wine', 'yacht']
        
#     ## large
#     batch_size=500
#     datasets=['kin8nm','powerplant','navalC','protein']
    (RESULTS, STDS), (DIV, DIV_std), (pDIV, pDIV_std),(ENT, ENT_std) = torch.load('Results/NEW/UCI_HMC2020-10-11-17:10.pt')
#     RESULTS, STDS={dataset:{} for dataset in datasets}, {dataset:{} for dataset in datasets}#torch.load('Results/NEW/UCI_splits2020-10-08-13:48.pt')##{dataset:{} for dataset in datasets}, {dataset:{} for dataset in datasets}
#     DIV, DIV_std={dataset:{} for dataset in datasets}, {dataset:{} for dataset in datasets}
#     pDIV, pDIV_std={dataset:{} for dataset in datasets}, {dataset:{} for dataset in datasets}
#     ENT, ENT_std={dataset:{} for dataset in datasets}, {dataset:{} for dataset in datasets}

    repeat=range(3)
    for dataset in datasets:
        print(dataset)     
 
        metrics={}
        stds={}
        div, div_std= {},{}
        Pdiv, Pdiv_std= {},{}
        H, H_std= {},{}
        
#         #MFVI
#         Thetas, Metrics=[],[]
#         for _ in repeat:
#             theta, metric= MFVI(dataset,device)
#             Thetas.append(theta), Metrics.append(metric)
#         Pdivergences, Pdivergences_std=paramCompareWithHMC(Thetas, dataset, 'MFVI')
#         divergences, divergences_std=CompareWithHMC(Thetas, dataset, 'MFVI') 

#         div.update(divergences),div_std.update(divergences_std)
#         Pdiv.update(Pdivergences), Pdiv_std.update(Pdivergences_std)
        
#         mean, std= MeanStd(Metrics, 'MFVI')
#         metrics.update(mean)
#         stds.update(std)
        
        
#         #FuNNeMFVI
#         Thetas, Metrics=[],[]
#         for _ in repeat:
#             theta, metric= FuNNeMFVI(dataset,device)
#             Thetas.append(theta), Metrics.append(metric)
#         Pdivergences, Pdivergences_std=paramCompareWithHMC(Thetas, dataset, 'FuNNeMFVI')
#         divergences, divergences_std=CompareWithHMC(Thetas, dataset, 'FuNNeMFVI') 

#         div.update(divergences),div_std.update(divergences_std)
#         Pdiv.update(Pdivergences), Pdiv_std.update(Pdivergences_std)
        
#         mean, std= MeanStd(Metrics, 'FuNNeMFVI')
#         metrics.update(mean)
#         stds.update(std)
        
        #FuNNeVI
        Thetas, Metrics=[],[]
        for _ in repeat:
            theta, metric= FuNNeVI(dataset,device)
            Thetas.append(theta), Metrics.append(metric)
        Pdivergences, Pdivergences_std=paramCompareWithHMC(Thetas, dataset, 'FuNNeVI')
        divergences, divergences_std=CompareWithHMC(Thetas, dataset, 'FuNNeVI') 
        entropies, entropies_std= ComputeEntropy(Thetas, dataset,'FuNNeVI')
        
        div.update(divergences),div_std.update(divergences_std)
        Pdiv.update(Pdivergences), Pdiv_std.update(Pdivergences_std)
        H.update(entropies), H_std.update(entropies_std)
        
        mean, std= MeanStd(Metrics, 'FuNNeVI')
        metrics.update(mean)
        stds.update(std)

        
        #GeNNeVI
        Thetas, Metrics=[],[]
        for _ in repeat:
            theta, metric= GeNNeVI(dataset,device)
            Thetas.append(theta), Metrics.append(metric)
        Pdivergences, Pdivergences_std=paramCompareWithHMC(Thetas, dataset, 'GeNNeVI')
        divergences, divergences_std=CompareWithHMC(Thetas, dataset, 'GeNNeVI')
        entropies, entropies_std= ComputeEntropy(Thetas, dataset,'GeNNeVI')

        div.update(divergences),div_std.update(divergences_std)
        Pdiv.update(Pdivergences), Pdiv_std.update(Pdivergences_std)
        H.update(entropies), H_std.update(entropies_std)

      
        mean, std= MeanStd(Metrics, 'GeNNeVI')
        metrics.update(mean)
        stds.update(std)
        
  
        
        #HMC
        metrics_HMC=HMC_metrics(dataset,device)
        metrics.update(metrics_HMC)
        HMC_=models_HMC[dataset]
        indices = torch.randperm(len(HMC_))[:1000]
        HMC=HMC_[indices].to(device)
        entropies, entropies_std= ComputeEntropy([HMC], dataset,'HMC')
        H.update(entropies), H_std.update(entropies_std)
        
        
        
        RESULTS[dataset].update(metrics), STDS[dataset].update(stds)
        DIV[dataset].update(div), DIV_std[dataset].update(div_std)
        pDIV[dataset].update(Pdiv),pDIV_std[dataset].update(Pdiv_std)
        ENT[dataset].update(H),ENT_std[dataset].update(H_std)

 #       print(DIV,DIV_std)
        torch.save([(RESULTS,STDS),(DIV,DIV_std),(pDIV,pDIV_std), (ENT,ENT_std)],file_name+'.pt')
        
