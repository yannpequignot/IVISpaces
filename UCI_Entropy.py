import numpy as np
import math
import torch
from torch import nn

import pandas as pd

from datetime import datetime

from torch.utils.data import Dataset

from Models import get_mlp, BigGenerator, MeanFieldVariationalDistribution, GaussianProcess, MC_Dropout_Wrapper
from Tools import AverageNormalLogLikelihood, logmvn01pdf
from Metrics import KL, evaluate_metrics, Entropy, BatchEntropy

from Experiments import get_setup

from Inference.IVI_noise import IVI

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
kNNE=50 #k-nearest neighbour

batch_size=50

sigma_prior=.5# TO DO check with other experiments setup.sigma_prior    


input_sampling='uniform' #'uniform', 'uniform+data'


def OOD_sampler(x_train,n_ood):
    M = x_train.max(0, keepdim=True)[0]
    m = x_train.min(0, keepdim=True)[0]
    X = torch.rand(n_ood,x_train.shape[1]).to(device) * (M-m) + m                           
    return X

def ensemble_bootstrap(dataset,device,seed):
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device, seed=seed) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()
    
     
    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    N = x_train.shape[0]
    input_dim=x_train.shape[1]
    model_list = []
      
    num_models=5 #10
    num_epochs=n_epochs
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    start = timeit.default_timer()

    for m_i in range(num_models):
        
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, layerwidth),
            activation,
            torch.nn.Linear(layerwidth, 1))
        model.to(device)

        loss = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        with trange(num_epochs) as tr:
            tr.set_description(desc=dataset+'/EnsembleB', refresh=False)
            for t in tr:
                cost = 0.
                count_batch=0
                for x,y in train_loader:
                    optimizer.zero_grad()
                    fx = model(x)
                    output = loss(fx, y)
                    output.backward()
                    optimizer.step()

                    cost += output.item() *len(x)
                    count_batch+=1
                tr.set_postfix(loss=cost/count_batch)              
        model_list.append(model)
    
    stop = timeit.default_timer()
    time = stop - start

    
    def EnsPredict(x,N):
        y_ts = [] 
    
        for m_i in range(len(model_list)):
            #Evaluate the model
            model_list[m_i].eval()
            y_ts.append(model_list[m_i](x).detach())

        y_t=torch.stack(y_ts, dim=0)
        y_t_mean = y_t.mean(axis=0)
        y_t_sigma = y_t.std(axis=0)
        y_pred=y_t_mean+y_t_sigma* torch.randn(1000,len(x),1).to(device)
        return y_pred
    
    X=[x_train,x_test, OOD_sampler(x_train,1000)]
    target=[y_train,y_test]
    
    Y=[EnsPredict(x,1000) for x in X]
#    H=[BatchEntropy(y.transpose(0,1),k=30) for y in Y]
    return Y
    

def Mc_dropout(dataset,device,seed):
    
    #MC_Dropout
    drop_prob=0.05
    num_epochs=2000 #4x500 = 20000 yarin gal
    learn_rate=1e-3
    
    #batch_size=128
    #TODO batch_size???
        
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device, seed=seed) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()
    
    batch_size=len(x_train) #128?

    
    weight_decay= 1e-1/(len(x_train)+len(x_test))**0.5

    
    
    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    
        
    def train_mc_dropout(x_train, y_train, batch_size, drop_prob, num_epochs, num_units, learn_rate, weight_decay):
        
        in_dim = x_train.shape[1]
        out_dim = y_train.shape[1]
        
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        net = MC_Dropout_Wrapper(input_dim=in_dim, output_dim=out_dim, no_units=num_units, \
                                 learn_rate=learn_rate, train_loader=train_loader, init_log_noise=0, \
                                 weight_decay=weight_decay, drop_prob=drop_prob,  device=device)

        with trange(num_epochs) as tr:
            tr.set_description(desc=dataset+'/McDropout', refresh=False)
            for t in tr:            
                loss = net.fit()
                tr.set_postfix(loss=loss)              
        return net
    
    
    start = timeit.default_timer()
    net  = train_mc_dropout(x_train=x_train,y_train=y_train, batch_size=batch_size, \
                            drop_prob=drop_prob, num_epochs=num_epochs,  num_units=layerwidth, learn_rate=learn_rate, weight_decay=weight_decay)
    stop = timeit.default_timer()
    time = stop - start
    
    def DropOutPredict(x,N):
        samples=[]
        for i in range(N):
            preds = net.network(x).detach() # T x 1
            samples.append(preds)

        samples = torch.stack(samples) #N x T x 1
        means = samples.mean(axis = 0).view(1,-1,1) #1 x T x 1
        aleatoric = torch.exp(net.network.log_noise).detach() #1
        epistemic = samples.std(axis = 0).view(-1,1) #  T x 1
        sigma_noise = aleatoric.view(1,1,1) 
        y_pred=means + epistemic * torch.randn(N,len(x),1).to(device)
        # 1 x T x 1 + (T x 1)*(N x T x 1) = N x T x 1
        return y_pred
 

    X=[x_train,x_test, OOD_sampler(x_train,1000)]
    Y=[DropOutPredict(x,1000) for x in X]
#    H=[BatchEntropy(y.transpose(0,1),k=30) for y in Y]
    return Y 


def MFVI(dataset,device, seed):
    
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device, seed=seed) 

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
    
    MFVI=MeanFieldVariationalDistribution(param_count, std_init=0. ,sigma=0.001, device=device)    

    _sigma_noise=torch.log(torch.tensor(1.).exp()-1.).clone().to(device).detach().requires_grad_(True)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    def ELBO(x_data, y_data, MFVI, _sigma_noise):
        y_pred=model(x_data,MFVI(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood=AverageNormalLogLikelihood(y_pred, y_data, sigma_noise)
        theta=MFVI(n_samples_KL)
        the_KL=MFVI.log_prob(theta).mean()-logmvn01pdf(theta,sigma_prior).mean()
        the_ELBO= - Average_LogLikelihood+ (len(x_data)/size_data)* the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise
    
    optimizer = torch.optim.Adam(list(MFVI.parameters())+[_sigma_noise], lr=learning_rate)
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
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach()

    X=[x_train,x_test,OOD_sampler(x_train,1000)]
    target=[y_train,y_test]
    Y=[model(X[i],theta) for i in range(len(X))]
    #H=[BatchEntropy(y.transpose(0,1),k=30) for y in Y]
    return Y


def FuNNeMFVI(dataset,device, seed):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device, seed=seed) 

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
    
    _sigma_noise=torch.log(torch.tensor(1.).exp()-1.).clone().to(device).detach().requires_grad_(True)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    optimizer = torch.optim.Adam(list(MFVI.parameters())+[_sigma_noise], lr=learning_rate)
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
    X=[x_train,x_test,OOD_sampler(x_train,1000)]
    Y=[model(X[i],theta) for i in range(len(X))]
    #H=[BatchEntropy(y.transpose(0,1),k=kNNE) for y in Y]
    return Y




def FuNNeVI(dataset,device, seed):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device, seed=seed) 

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
    
    _sigma_noise=torch.log(torch.tensor(1.).exp()-1.).clone().to(device).detach().requires_grad_(True)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    optimizer = torch.optim.Adam(list(GeN.parameters())+[_sigma_noise], lr=learning_rate)
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
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach()

    X=[x_train,x_test,OOD_sampler(x_train,1000)]
    Y=[model(X[i],theta) for i in range(len(X))]
    #H=[BatchEntropy(y.transpose(0,1),k=30) for y in Y]
    return Y


def GeNNeVI(dataset,device, seed):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device, seed=seed) 

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
    
    _sigma_noise=torch.log(torch.tensor(1.0).exp()-1.).clone().to(device).detach().requires_grad_(True)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    optimizer = torch.optim.Adam(list(GeN.parameters())+[_sigma_noise], lr=learning_rate)
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
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach()

    X=[x_train,x_test,OOD_sampler(x_train,1000)]
    Y=[model(X[i],theta) for i in range(len(X))]
    #H=[BatchEntropy(y.transpose(0,1),k=30) for y in Y]
    return Y


if __name__ == "__main__":
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")
    file_name='Results/NEW/UCI_ENTROPY'+date_string
    makedirs(file_name)

    with open(file_name, 'w') as f:
        script=open(__file__)
        f.write(script.read())  

    datasets=['wine', 'yacht']# ['boston','concrete', 'energy', 'wine', 'yacht']#'powerplant',
    ENTROPY=torch.load('Results/NEW/UCI_ENTROPY2020-10-13-21:22.pt')#{dataset:{} for dataset in datasets}#torch.load('Results/NEW/UCI_ENTROPY2020-10-08-12:13.pt')#{dataset:{} for dataset in datasets}#torch.load('Results/NEW/UCI_ENTROPY2020-10-07-15:13.pt')#

    seed=117
    
    for dataset in datasets:
        print(dataset)     
 
        entropies={}
    
        H=FuNNeMFVI(dataset,device, seed)
        entropies.update({'FuNNeMFVI': H})       
        ENTROPY[dataset].update(entropies)

        H=ensemble_bootstrap(dataset,device, seed)
        entropies.update({'EnsembleB': H})    
        ENTROPY[dataset].update(entropies)

        H=Mc_dropout(dataset,device, seed)
        entropies.update({'McDropOut': H})
        ENTROPY[dataset].update(entropies)

        
        H=MFVI(dataset,device, seed)
        entropies.update({'MFVI': H})
        ENTROPY[dataset].update(entropies)
               
            
        H=GeNNeVI(dataset,device, seed)
        entropies.update({'GeNNeVI':H})
        ENTROPY[dataset].update(entropies)

        H=FuNNeVI(dataset,device, seed)
        entropies.update({'FuNNeVI':H})
        
#         H,MSE=FuNNeVI_GPprior(dataset,device, seed)
#         entropies.update({'FuNNeVI-GP':(H,MSE)})
            
        ENTROPY[dataset].update(entropies)
        
        torch.save(ENTROPY,file_name+'.pt')
