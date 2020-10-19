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

import timeit
import os

def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


# def PlotFoong(y_pred, x_pred=x_pred,  x=x_train, y=y_train, device=device):
    
#     parameters = {'xtick.labelsize':8,
#                   'ytick.labelsize':8}
#     plt.rcParams.update(parameters)
    
#     N=y_pred.shape[0]-1
#     print(N)
#     m_3=int(0.001*N)
#     M_3=N-m_3
#     m_2=int(0.021*N)
#     M_2=N-m_2
#     m_1=int(0.136*N)
#     M_1=N-m_1

#     x_pred=x_pred.squeeze()

#     pred,_=y_pred.sort(dim=0)
#     y_mean=y_pred.mean(dim=0).squeeze().cpu()
#     y_3=pred[m_3,:].squeeze().cpu()
#     Y_3=pred[M_3,:].squeeze().cpu()
#     y_2=pred[m_2,:].squeeze().cpu()
#     Y_2=pred[M_2,:].squeeze().cpu()    
#     y_1=pred[m_1,:].squeeze().cpu()
#     Y_1=pred[M_1,:].squeeze().cpu()

    

#     fig, ax=plt.subplots(figsize=(5,3))
#     plt.plot(x_pred.cpu(), y_mean, color='springgreen')
#     plt.plot(x_pred.cpu(), torch.cos(4.0*(x_pred+0.2)).cpu(),'--',color='green')

#     ax.fill_between(x_pred.cpu(), y_3, Y_3, facecolor='springgreen', alpha=0.1)
#     ax.fill_between(x_pred.cpu(), y_2, Y_2, facecolor='springgreen', alpha=0.1)
#     ax.fill_between(x_pred.cpu(), y_1, Y_1, facecolor='springgreen', alpha=0.1)

#     ax.set_yticks([-3,0,3])
#     ax.set_xticks([-1,0,1])
#     plt.grid(True, which='major', linewidth=0.5)
#     plt.ylim(-4, 4)
#     plt.xlim(-2.,2.)
#     plt.scatter(x.cpu(), y.cpu() , marker='.',color='black',zorder=4)
#     return fig

def OOD_sampler(n_ood=50):
    M = -4.
    m = 2.
    X = torch.rand(n_ood,1).to(device) * (M-m) + m                           
    return X
        
        
## Hyperparameters ##

#predictive model
layerwidth=50
nblayers=1
activation=nn.Tanh()#ReLU()

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


input_sampling='uniform' #'uniform', 'uniform+data'


def ensemble_bootstrap(dataset,device):
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()
     
    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    N = x_train.shape[0]
    input_dim=x_train.shape[1]
    model_list = []
      
    num_models=10 #10
    num_epochs=3000
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
            tr.set_description(desc=dataset+'/EnsembleB-{}'.format(m_i), refresh=False)
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

    y_ts = [] 

    for m_i in range(len(model_list)):
        #Evaluate the model
        model_list[m_i].eval()
        y_ts.append(model_list[m_i](x_pred).detach())

    y_t=torch.stack(y_ts, dim=0)
    y_t_mean = y_t.mean(axis=0)
    y_t_sigma = y_t.std(axis=0)
    
    y_pred = y_t_mean+y_t_sigma* torch.randn(1000,len(x_pred),1).to(device)+\
        setup.sigma_noise*torch.randn(1000,len(x_pred),1).to(device)
    return y_pred



def Mc_dropout(dataset,device):
    
    #MC_Dropout
    drop_prob=0.05
    num_epochs=2000 #4x500 = 20000 yarin gal
    learn_rate=1e-3
    
    #batch_size=128
    #TODO batch_size???
        
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    
    batch_size=len(x_train) #128?

    
    weight_decay= 1e-1/(len(x_train)+len(x_test))**0.5

    
    
    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    
        
    def train_mc_dropout(x_train, y_train, batch_size, drop_prob, num_epochs, num_units, learn_rate, weight_decay, log_noise):
        
        in_dim = x_train.shape[1]
        out_dim = y_train.shape[1]
        
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
        net = MC_Dropout_Wrapper(input_dim=in_dim, output_dim=out_dim, no_units=num_units, \
                                 learn_rate=learn_rate, train_loader=train_loader, init_log_noise=log_noise, \
                                 weight_decay=weight_decay, drop_prob=drop_prob,  device=device, learn_noise=False, activation=nn.Tanh())


        with trange(num_epochs) as tr:
            tr.set_description(desc=dataset+'/McDropout', refresh=False)
            for t in tr:            
                loss = net.fit()
                tr.set_postfix(loss=loss)              
        return net
    
    
    start = timeit.default_timer()
    net  = train_mc_dropout(x_train=x_train,y_train=y_train, batch_size=batch_size,\
                            drop_prob=drop_prob, num_epochs=num_epochs,  num_units=layerwidth, \
                            learn_rate=learn_rate, weight_decay=weight_decay, log_noise=np.log(setup.sigma_noise))
    stop = timeit.default_timer()
    time = stop - start
    
    samples=[]
    nb_predictors=1000# N
    for i in range(nb_predictors):
        preds = net.network(x_pred).detach() # T x 1
        samples.append(preds)
     
    samples = torch.stack(samples) #N x T x 1
    means = samples.mean(axis = 0).view(1,-1,1) #1 x T x 1
    aleatoric = torch.exp(net.network.log_noise).detach() #1
    epistemic = samples.std(axis = 0).view(-1,1) #  T x 1
    sigma_noise = aleatoric.view(1,1,1) 
    y_pred=means + epistemic * torch.randn(nb_predictors,len(x_pred),1).to(device) # 1 x T x 1 + (T x 1)*(N x T x 1) = N x T x 1
    return y_pred


def MFVI(dataset,device):
    
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation) 
    
    MFVI=MeanFieldVariationalDistribution(param_count, std_init=1. ,sigma=0.00001, device=device)    

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3*patience, factor=lr_decay, min_lr=min_lr)
    Run=IVI(train_loader, ELBO, optimizer)

    start = timeit.default_timer()
    with trange(2*n_epochs) as tr:
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
    y_pred=model(x_pred,theta)
    y_pred+=sigma_noise*torch.randn_like(y_pred)
    return y_pred

def FuNNeMFVI(dataset,device):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()

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
    
    def projection(theta0,theta1, x_data):
        X=OOD_sampler()
        #compute projection on both paramters with model
        theta0_proj=model(X, theta0).squeeze(2)
        theta1_proj=model(X, theta1).squeeze(2)
        return theta0_proj, theta1_proj

    def kl(x_data, GeN):

        theta=GeN(n_samples_KL) #variationnel
        theta_prior=prior(n_samples_KL) #prior

        theta_proj, theta_prior_proj = projection(theta, theta_prior,x_data)

        K=KL(theta_proj, theta_prior_proj,k=5,device=device)
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3*patience, factor=lr_decay, min_lr=min_lr)

    Run=IVI(train_loader, ELBO, optimizer)
    
    start = timeit.default_timer()
    with trange(2*n_epochs) as tr:
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
    y_pred=model(x_pred,theta)
    y_pred+=sigma_noise*torch.randn_like(y_pred)
    return y_pred




def FuNNeVI(dataset,device):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()

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
    
    
    def projection(theta0,theta1, x_data):
        X=OOD_sampler()
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

    optimizer = torch.optim.Adam(GeN.parameters(), lr=learning_rate)
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
    y_pred=model(x_pred,theta)
    y_pred+=sigma_noise*torch.randn_like(y_pred)
    return y_pred


def GeNNeVI(dataset,device):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    x_train, y_train=setup.train_data()

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
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach().cpu()
    y_pred=model(x_pred,theta)
    y_pred+=sigma_noise*torch.randn_like(y_pred)
    return y_pred

models_HMC = torch.load('Results/HMC_models.pt')


def HMC(dataset,device):
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 
    
    ## predictive model
    input_dim=setup.input_dim
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)

    HMC_=models_HMC[dataset]
    indices = torch.randperm(len(HMC_))[:1000]
    HMC=HMC_[indices].to(device)
    sigma_noise = torch.tensor(setup.sigma_noise)
    y_pred=model(x_pred,HMC)
    y_pred+=sigma_noise*torch.randn_like(y_pred)
    return y_pred


if __name__ == "__main__":
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")
    file_name='Results/NEW/WAVE_OOS'+date_string
    makedirs(file_name)

    x_pred=torch.linspace(-4.,2.,500).unsqueeze(-1).to(device)


    with open(file_name, 'w') as f:
        script=open(__file__)
        f.write(script.read())  

    ## small 
    batch_size=50
    dataset='foong'#['boston','concrete', 'energy', 'wine', 'yacht']
        
    RESULTS=torch.load('Results/NEW/WAVE_OOS2020-10-12-09:49.pt')#torch.load('Results/NEW/WAVE_OOS2020-10-10-20:02.pt')#{}
    
#     method='EnsembleB'
#     y_pred=ensemble_bootstrap(dataset,device)
#     RESULTS.update({method:y_pred})

#     method='McDropOut'
#     y_pred=Mc_dropout(dataset,device)
#     RESULTS.update({method:y_pred})

#     method='FuNNeMFVI'
#     y_pred=FuNNeMFVI(dataset,device)
#     RESULTS.update({method:y_pred})

#     method='MFVI'
#     y_pred=MFVI(dataset,device)
#     RESULTS.update({method:y_pred})

#     method='GeNNeVI'
#     y_pred=GeNNeVI(dataset,device)
#     RESULTS.update({method:y_pred})
   
    method='FuNNeVI'
    y_pred=FuNNeVI(dataset,device)
    RESULTS.update({method:y_pred})

#     method='HMC'
#     y_pred=HMC(dataset,device)
#     RESULTS.update({method:y_pred})
  

    torch.save(RESULTS,file_name+'.pt')
