import torch
from torch import nn
import argparse
import mlflow
import timeit

import numpy as np
import itertools

from Models import BigGenerator

from Experiments import draw_experiment, get_setup, save_model
from Metrics import KL, sw

import tempfile

lat_dim=5

def _KL(s,t,device):
    k=1
    myKL=KL(s,t,k=k, device=device)
    while torch.isinf(myKL):
        k+=1
        myKL=KL(s, t, k=k, device=device)
    return myKL

def run(dataset, method, metrics, device):
    
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 
    G=BigGenerator(lat_dim, setup.param_count, device).to(device)

    metric='KL(-,HMC)'
    print(dataset+': '+metric)
    KLs=[]
    for i,m in models[dataset].items():
        t=models_HMC[dataset].to(device)
        G.load_state_dict(m)
        s=G(t.shape[0]).detach()
        K=_KL(s,t,device)
        print(K.item())
        KLs.append(K.item())
    print(KLs)

    metrics[(method,metric)].update({dataset:(np.mean(KLs).round(decimals=3), np.std(KLs).round(decimals=3))})
    
    metric='KL(HMC,-)'
    print(dataset+': '+metric)

    KLs=[]
    for i,m in models[dataset].items():
        t=models_HMC[dataset].to(device)
        G.load_state_dict(m)
        s=G(t.shape[0]).detach()
        K=_KL(t,s,device)
        print(dataset+': '+str(K.item()))
        KLs.append(K.item())
    print(KLs)
    
    metrics[(method,metric)].update({dataset:(np.mean(KLs).round(decimals=3), np.std(KLs).round(decimals=3))})
    
    metric='SW(-,G)'
    print(dataset+': '+metric)

    KLs=[]
    for i,m in models[dataset].items():
        t=models_HMC[dataset].to(device)
        G.load_state_dict(m)
        s=G(t.shape[0]).detach()
        K=sw(t.cpu() , s.cpu() , 'cpu')
        print(dataset+': '+str(K.item()))
        KLs.append(K.item())
    print(KLs)
    metrics[(method,metric)].update({dataset:(np.mean(KLs).round(decimals=3), np.std(KLs).round(decimals=3))})
    
    models_pairs=list(itertools.combinations(models[dataset].items(),2))
    
    metric='KL(-,-)'
    print(dataset+': '+metric)

    KLs=[]
    for (i,m),(j,n) in models_pairs:
        G.load_state_dict(m)
        s=G(10000).detach()
        G.load_state_dict(n)
        t=G(10000).detach()
        K=_KL(t,s, device)
        K_=_KL(s,t,device)
        print(dataset+': '+str((K.item(), K_.item())))
        KLs.append(K.item())
        KLs.append(K_.item())
    print(KLs)
    metrics[(method,metric)].update({dataset:(np.mean(KLs).round(decimals=3), np.std(KLs).round(decimals=3))})
    
    metric='SW(-,-)'
    print(dataset+': '+metric)

    KLs=[]
    for (i,m),(j,n) in models_pairs:
        G.load_state_dict(m)
        s=G(10000).detach()
        G.load_state_dict(n)
        t=G(10000).detach()
        SW=sw(s.cpu(),t.cpu(),'cpu')
        print(dataset+': '+str(SW))
        KLs.append(SW.item())
    
    metrics[(method,metric)].update({dataset:(np.mean(KLs).round(decimals=3), np.std(KLs).round(decimals=3))})

    return metrics

# compute KL and SW in parameter space 
# on 'models' obtained with 'method' 
# with respect to 'models_HMC' and 
# save a dictionary under 'Results/ParamDivergence'+method+'.pt'

if __name__ == "__main__":
    
    models_HMC = torch.load('Results/HMC_models.pt')
    models = torch.load('mlruns/2/c40e5719924a44a2a88260bb8eb63c6f/artifacts/FuNmodels.pt')
    method = 'FuNNeVI'
    
    
    datasets = [d for d, m in models_HMC.items()]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
    metrics=['KL(-,HMC)', 'KL(HMC,-)', 'SW(-,G)', 'KL(-,-)', 'SW(-,-)']

    metrics = {(method,metric):{} for metric in metrics}

    
    for d in datasets:
        metrics=run(d, method, metrics, device) 
        print(d+': done :-)')

        torch.save(metrics, 'Results/ParamDivergence'+method+'.pt')
