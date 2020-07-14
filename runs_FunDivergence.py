import torch
from torch import nn
import argparse
import mlflow
import timeit

import numpy as np
import itertools

from Models import BigGenerator

from Experiments import draw_experiment, get_setup, save_model
from Metrics import KL, batchKL, FunSW, FunKL, sw

import tempfile

lat_dim=5

def _FunKL(s,t,projection,device):
    k=1
    FKL=FunKL(s,t,projection=projection,device=device,k=k)
    while torch.isinf(FKL):
        k+=1
        FKL=FunKL(s,t,projection=projection,device=device,k=k)
    return FKL

def run(dataset, method, metrics, ratio_ood, device):
    
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 
    G=BigGenerator(lat_dim, setup.param_count, device).to(device)
    def projection(t,s,m):
        return setup.projection(t,s,n_samples=m,ratio_ood=ratio_ood)

    
    metric='KL(-,HMC)'
    print(dataset+': '+metric)
    KLs=[]
    for i,m in models[dataset].items():
        t=models_HMC[dataset].to(device)
        G.load_state_dict(m)
        s=G(t.shape[0]).detach()
        K=_FunKL(s,t,projection,device)
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
        K=_FunKL(t,s,projection,device)
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
        K=FunSW(t,s, projection, device)
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
        K=_FunKL(t,s, projection,device)
        K_=_FunKL(s,t, projection,device)
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
        SW=FunSW(t,s, projection, device)
        print(dataset+': '+str(SW))
        KLs.append(SW.item())
    
    metrics[(method,metric)].update({dataset:(np.mean(KLs).round(decimals=3), np.std(KLs).round(decimals=3))})

    return metrics

parser = argparse.ArgumentParser()
parser.add_argument("--ratio_ood", type=float, default=1.,
                    help="ratio in [0,1] of ood inputs w.r.t data inputs for MC sampling of predictive distance")


# compute KL and SW in function space (with 'ratio_ood' (default=1) and 'projection' defined in each setup) 
# on 'models' obtained with 'method' 
# with respect to 'models_HMC' and 
# save a dictionary under 'Results/Divergence'+method+'.pt'


if __name__ == "__main__":
    
    args = parser.parse_args()
    print(args)

    models_HMC = torch.load('Results/HMC_models.pt')
    models = torch.load('mlruns/2/c40e5719924a44a2a88260bb8eb63c6f/artifacts/FuNmodels.pt')
    method = 'FuNNeVI'
    
    datasets = [d for d, m in models_HMC.items()]

    ratio_ood = args.ratio_ood

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
    metrics=['KL(-,HMC)', 'KL(HMC,-)', 'SW(-,G)', 'KL(-,-)', 'SW(-,-)']

    metrics ={(method,metric):{} for metric in metrics}

    
    for d in datasets:
        metrics=run(d, method, metrics, ratio_ood, device) 
        print(d+': done :-)')

        torch.save(metrics, 'Results/Divergence'+method+'.pt')
