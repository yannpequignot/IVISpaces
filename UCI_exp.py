import numpy as np
import math
import torch
from torch import nn


from datetime import datetime

from torch.utils.data import Dataset

from Models import get_mlp, BigGenerator, MeanFieldVariationalDistribution
from Tools import AverageNormalLogLikelihood, logmvn01pdf
from Metrics import KL, evaluate_metrics, Entropy

from Experiments import get_setup

from Inference.IVI_noise import IVI

from tqdm import trange

import timeit

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
patience=40
lr_decay=.5#.7
min_lr= 0.0001
n_epochs=1#5000#2000


#loss hyperparameters
n_samples_LL=100 #nb of predictor samples for average LogLikelihood

n_samples_KL=500 #nb of predictor samples for KL divergence
kNNE=1 #k-nearest neighbour

n_samples_FU=200 #nb of ood inputs for predictive KL NN estimation


sigma_prior=.5# TO DO check with other experiments setup.sigma_prior    

def MFVI_noise(dataset,device):
    
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device, seed=42) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    batch_size=int(np.min([size_data // 6, 500])) #50 works fine too!
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation) 
    
    MFVI=MeanFieldVariationalDistribution(param_count, std_init=0.,sigma=0.2, device=device)    

    _sigma_noise=torch.log(torch.tensor(1.).exp()-1.).clone().to(device).detach().requires_grad_(True)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    def ELBO(x_data, y_data, MFVI, _sigma_noise):
        alpha=(len(x_data)/size_data) #TODO check with alpah=1.

        y_pred=model(x_data,MFVI(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood=AverageNormalLogLikelihood(y_pred, y_data, sigma_noise)
        theta=MFVI(n_samples_KL)
        the_KL=MFVI.log_prob(theta).mean()-logmvn01pdf(theta,sigma_prior).mean()
        the_ELBO= - Average_LogLikelihood+ alpha* the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise
    
    optimizer = torch.optim.Adam(list(MFVI.parameters())+[_sigma_noise], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, verbose=True, min_lr=min_lr)
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
    return metrics


def FuNNeVI_noise(dataset,device):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device, seed=42) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    batch_size=int(np.min([size_data // 6, 500])) #50 works fine too!
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
    def prior(n):
        return sigma_prior*torch.randn(size=(n,param_count), device=device)
    
    def projection(theta0,theta1, x_data):
        #batch sample OOD   
        n_ood=n_samples_FU
        epsilon=0.1
        M = x_train.max(0, keepdim=True)[0]+epsilon
        m = x_train.min(0, keepdim=True)[0]-epsilon
        X_ood = torch.rand(n_ood,input_dim).to(device) * (M-m) + m    
        #X_ood = x_data+torch.randn_like(x_data)
        
        #compute projection on both paramters with model
        theta0_proj=model(X_ood, theta0).squeeze(2)
        theta1_proj=model(X_ood, theta1).squeeze(2)
        return theta0_proj, theta1_proj

    def kl(x_data, GeN):

        theta=GeN(n_samples_KL) #variationnel
        theta_prior=prior(n_samples_KL) #prior

        theta_proj, theta_prior_proj = projection(theta, theta_prior,x_data)

        K=KL(theta_proj, theta_prior_proj,k=kNNE,device=device)
        return K
    
    def ELBO(x_data, y_data, GeN, _sigma_noise):
        alpha=(len(x_data)/size_data) #TODO check with alpha=1.
        y_pred=model(x_data,GeN(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood=AverageNormalLogLikelihood(y_pred, y_data, sigma_noise)
        the_KL=kl(x_data, GeN)
        the_ELBO= - Average_LogLikelihood+ alpha* the_KL#(len(x_data)/size_data)*the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise

    #generative model
    GeN = BigGenerator(lat_dim,param_count,device).to(device)

    ## Parametrize noise for learning aleatoric uncertainty
    
    _sigma_noise=torch.log(torch.tensor(1.0).exp()-1.).clone().to(device).detach().requires_grad_(True)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    optimizer = torch.optim.Adam(list(GeN.parameters())+[_sigma_noise], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, verbose=True, min_lr=min_lr)

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
    return metrics


def GeNNeVI_noise(setup,device):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device, seed=42) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()
    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    batch_size=int(np.min([size_data // 6, 500])) #50 works fine too!
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
        alpha=(len(x_data)/size_data) #TODO check with alpah=1.
        y_pred=model(x_data,GeN(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood=AverageNormalLogLikelihood(y_pred, y_data, sigma_noise)
        the_KL=kl(x_data, GeN)
        the_ELBO= - Average_LogLikelihood+ alpha* the_KL#(len(x_data)/size_data)*the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise

    #generative model
    GeN = BigGenerator(lat_dim,param_count,device).to(device)

    ## Parametrize noise for learning aleatoric uncertainty
    
    _sigma_noise=torch.log(torch.tensor(1.0).exp()-1.).clone().to(device).detach().requires_grad_(True)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    optimizer = torch.optim.Adam(list(GeN.parameters())+[_sigma_noise], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, verbose=True, min_lr=min_lr)

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
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach().cpu()
    y_pred=model(x_test,theta)
    metrics=get_metrics(y_pred, sigma_noise, y_test, std_y_train, 'GeNNeVI', time)
    return metrics

def get_metrics(y_pred, sigma_noise, y_test, std_y_train, method, time):
    metrics=evaluate_metrics(y_pred, sigma_noise.view(1,1,1), y_test,  std_y_train, device='cpu', std=False)
    metrics.update({'time [s]': time})
    metrics.update({'std noise': sigma_noise.item()})
    metrics_list=list(metrics.keys())
    for j in metrics_list:
        metrics[(method,j)] = metrics.pop(j)
    return metrics


if __name__ == "__main__":
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")
    datasets=['boston','concrete', 'energy', 'powerplant',  'wine', 'yacht']
    RESULTS={dataset:{} for dataset in datasets}#
    for dataset in datasets:
        print(dataset)     
 
        metrics={}
        metrics.update(MFVI_noise(dataset,device))
        metrics.update(GeNNeVI_noise(dataset,device))
        metrics.update(FuNNeVI_noise(dataset,device))
            
        RESULTS[dataset].update(metrics)
        torch.save(RESULTS,'Results/NEW/UCI'+date_string+'.pt')
        #RESULTS.append(GeNNeVI_noise(dataset,device))
        #torch.save(RESULTS,'Results/NEW/GeNoise'+date_string+'.pt')