import torch
from torch import nn
import argparse
import mlflow
import timeit

import numpy as np

from Models import BigGenerator

from Experiments import draw_experiment, get_setup, save_model
from Metrics import evaluate_metrics


import tempfile


def run(dataset):
    
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 
    
    model=setup._model
    sigma_noise=setup.sigma_noise
    G=BigGenerator(lat_dim, setup.param_count, device).to(device)

    
    x_test, y_test=setup.test_data()
    
    RMSEs=[]
    LPPs=[]
    PICPs=[]
    MPIWs=[]

    for i,m in models[dataset].items():
        G.load_state_dict(m)
        theta=G(2000).detach()

        std_y_train = torch.tensor(1.)
        if hasattr(setup, '_scaler_y'):
            std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

        metrics=evaluate_metrics(theta, model, x_test, y_test, sigma_noise, std_y_train, device='cpu', std=False)
        
        RMSEs.append(metrics['RMSE'])
        LPPs.append(metrics['LPP'])
        PICPs.append(metrics['PICP'])
        MPIWs.append(metrics['MPIW'])
    
    metrics_dict={(method,dataset):{'RMSE':(np.mean(RMSEs).round(decimals=3),np.std(RMSEs).round(decimals=3)),
                           'LPP': (np.mean(LPPs).round(decimals=3),np.std(LPPs).round(decimals=3)),
                           'PICP':  (np.mean(PICPs).round(decimals=3),np.std(PICPs).round(decimals=3)), 
                           'MPIW':  (np.mean(MPIWs).round(decimals=3),np.std(MPIWs).round(decimals=3))
                           }
                   }
                 
    
    return metrics_dict


if __name__ == "__main__":
    device='cpu'
    log_device = 'cpu'

    models_path=
    results_path=
    models=torch.load(models_path)
    lat_dim=5
    datasets=[d for d,i in models.items()]
    method='HMC'

    results={}


    for d in datasets:
        metrics=run(d) 
        results.update(metrics)
        
    torch.save(results, results_path)
