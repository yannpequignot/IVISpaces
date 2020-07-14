import torch
from torch import nn
import argparse
import mlflow
import timeit

import numpy as np

from tempfile import TemporaryDirectory

from Inference.GeNNeVI import GeNNeVI
from Models import BigGenerator
from Experiments import log_exp_metrics, draw_experiment, get_setup, save_model

import tempfile

lat_dim=5
NNE=1
n_samples_KL=500
n_samples_LL=100
max_iter=30000
learning_rate=0.005
patience=1000
min_lr= 0.001
lr_decay=.7

def learning(loglikelihood, batch, size_data, prior,
                   lat_dim, param_count, 
                   kNNE, n_samples_KL, n_samples_LL,  
                   max_iter, learning_rate, min_lr, patience, lr_decay, 
                   device):

    GeN = BigGenerator(lat_dim, param_count,device).to(device)

    optimizer = GeNNeVI(loglikelihood, batch, size_data, prior,
                          kNNE, n_samples_KL, n_samples_LL, 
                          max_iter, learning_rate, min_lr, patience, lr_decay,
                          device)

    ELBO = optimizer.run(GeN)

    return GeN, optimizer.scores, ELBO.item()




def log_GeNVI_experiment(lat_dim, kNNE, n_samples_KL, n_samples_LL, 
                         max_iter, learning_rate, min_lr, patience, lr_decay,
                         device):
    
    mlflow.set_tag('device', device)
    mlflow.set_tag('NNE', kNNE)
   

    mlflow.log_param('lat_dim', lat_dim)
    

    mlflow.log_param('n_samples_KL', n_samples_KL)
    mlflow.log_param('n_samples_LL', n_samples_LL)
    

    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('patience', patience)
    mlflow.log_param('lr_decay', lr_decay)
    mlflow.log_param('max_iter', max_iter)
    mlflow.log_param('min_lr', min_lr)
    return

    
def log_GeNVI_setup(setup, batch):
    
    mlflow.set_tag('batch_size', batch)

        
    mlflow.set_tag('sigma_noise', setup.sigma_noise)    

    mlflow.set_tag('sigma_prior', setup.sigma_prior)    
    mlflow.set_tag('param_dim', setup.param_count)

    return 
    
    
def log_GeNVI_run(ELBO, scores):    
    for t in range(len(scores['ELBO'])):
        mlflow.log_metric("elbo", float(scores['ELBO'][t]), step=100*t)
        mlflow.log_metric("KL", float(scores['KL'][t]), step=100*t)
        mlflow.log_metric("LL", float(scores['LL'][t]), step=100*t)        
        mlflow.log_metric("learning_rate", float(scores['lr'][t]), step=100*t)
    return
 


def run(setup):
    
    setup_ = get_setup( setup)
    setup=setup_.Setup( device) 
    
    loglikelihood=setup.loglikelihood
    size_sample=setup.n_train_samples
    param_count=setup.param_count
    input_dim=setup.input_dim

    batch=np.min([int(size_sample/6),500])

    rho=batch/size_sample #* input_dim
                  
    def prior(n):
        return setup.sigma_prior*torch.randn(size=(n,param_count), device= device)


    GeN_models_dict=[]
    RMSEs=[]
    LPPs=[]
    PICPs=[]
    MPIWs=[]  
    TIMEs=[]
    for i in range(10):
        seed=42+i
        setup=setup_.Setup(device,seed) 
        with mlflow.start_run(run_name=str(setup.experiment_name)+'-'+str(i) ,nested=True):
            log_GeNVI_setup(setup, batch)

            start = timeit.default_timer()

            GeN, log_scores, ELBO = learning(loglikelihood, batch, setup.n_train_samples, prior, 
                                                  lat_dim, setup.param_count,
                                                  NNE,  n_samples_KL,  n_samples_LL,
                                                  max_iter,  learning_rate,  min_lr,  patience,
                                                  lr_decay,  device)


            stop = timeit.default_timer()
            execution_time = stop - start
        
            log_device = 'cpu'
            theta = GeN(1000).detach().to(log_device)

            LPP_test, RMSE_test, _, PICP_test, MPIW_test = setup.evaluate_metrics(theta,log_device)
        
            RMSEs.append(RMSE_test[0].item())
            LPPs.append(LPP_test[0].item())
            PICPs.append(PICP_test.item())
            MPIWs.append(MPIW_test.item())
            TIMEs.append(execution_time)
            
            log_GeNVI_run(ELBO, log_scores)

            #save_model(GeN)
            GeN_models_dict.append((i,GeN.state_dict().copy()))

            """
            if setup.plot:
                log_device = 'cpu'
                theta = GeN(1000).detach().to(log_device)
                draw_experiment(setup, theta[0:1000], log_device)
            """

    models={str(i): model for i,model in GeN_models_dict} 
    
        
    metrics_dict={('GeNNeVI',dataset):{'RMSE':(np.mean(RMSEs).round(decimals=3),np.std(RMSEs).round(decimals=3)),
                           'LPP': (np.mean(LPPs).round(decimals=3),np.std(LPPs).round(decimals=3)),
                           'PICP':  (np.mean(PICPs).round(decimals=3),np.std(PICPs).round(decimals=3)), 
                           'MPIW':  (np.mean(MPIWs).round(decimals=3),np.std(MPIWs).round(decimals=3))
                           }
                   }
                 
    return models, metrics_dict
    

    




if __name__ == "__main__":
    
    GeNmodels={}
    results={}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    datasets=['powerplant', 'boston', 'yacht', 'concrete', 'energy', 'wine']

    xpname = 'GeNNeVI-splits'
    mlflow.set_experiment(xpname)
    
    with mlflow.start_run():
        log_GeNVI_experiment(lat_dim, NNE, n_samples_KL, n_samples_LL, 
                         max_iter, learning_rate, min_lr, patience, lr_decay,
                         device)
        
        for dataset in datasets:
            print(dataset)
            models, metrics_dict=run(dataset) 
            print(dataset+': done :-)')
            GeNmodels.update({dataset:models})
            results.update(metrics_dict)

            torch.save(GeNmodels, 'Results/SpGeNmodels.pt')
            torch.save(results, 'Results/SpGeNmetrics.pt')

        mlflow.log_artifact('Results/SpGeNmodels.pt')
        mlflow.log_artifact('Results/SpGeNmetrics.pt')

  

    
