import torch
from torch import nn
import argparse
import mlflow
import timeit

import numpy as np


from Inference.FuNNeVI import FuNNeVI
from Models import BigGenerator
from Experiments import log_exp_metrics, draw_experiment, get_setup, save_model

import tempfile

lat_dim=5
nb_models=1#to change
NNE=1
p_norm=2
n_samples_KL=100#500
n_samples_LL=100
max_iter=1#30000
learning_rate=0.005
patience=1000
min_lr= 0.001
lr_decay=.7

def learning(loglikelihood, batch, size_data, prior, projection, p,
                   lat_dim, param_count, 
                   kNNE, n_samples_KL, n_samples_LL,  
                   max_iter, learning_rate, min_lr, patience, lr_decay, 
                   device, rho):

    GeN = BigGenerator(lat_dim, param_count,device).to(device)

    optimizer = FuNNeVI(loglikelihood, batch, size_data, prior, projection, p, kNNE,
                        n_samples_KL, n_samples_LL, max_iter, learning_rate, min_lr, patience, lr_decay, device)

    ELBO = optimizer.run(GeN)

    return GeN, optimizer.scores, ELBO.item()



def log_GeNVI_experiment(sigma_input, ratio_ood, p, lat_dim, kNNE, n_samples_KL, n_samples_LL, 
                         max_iter, learning_rate, min_lr, patience, lr_decay,
                         device):
    
    mlflow.set_tag('device', device)
    mlflow.set_tag('NNE', kNNE)
   

    mlflow.log_param('lat_dim', lat_dim)
    
    mlflow.log_param('L_p norm', p)
    mlflow.log_param('sigma_input', sigma_input)

    mlflow.log_param('ratio_ood', ratio_ood)


    mlflow.log_param('n_samples_KL', n_samples_KL)
    mlflow.log_param('n_samples_LL', n_samples_LL)
    

    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('patience', patience)
    mlflow.log_param('lr_decay', lr_decay)
    mlflow.log_param('max_iter', max_iter)
    mlflow.log_param('min_lr', min_lr)
    return

    
def log_GeNVI_setup(setup,  n_samples_FU, ratio_ood, batch):
    
    mlflow.set_tag('batch_size', batch)

        
    mlflow.set_tag('sigma_noise', setup.sigma_noise)    

    mlflow.set_tag('sigma_prior', setup.sigma_prior)    
    mlflow.set_tag('param_dim', setup.param_count)

    mlflow.log_param('n_samples_FU', n_samples_FU)
#    mlflow.log_param('ratio_ood', ratio_ood)
    return 
    
    
def log_GeNVI_run(ELBO, scores):    
    for t in range(len(scores['ELBO'])):
        mlflow.log_metric("elbo", float(scores['ELBO'][t]), step=100*t)
        mlflow.log_metric("KL", float(scores['KL'][t]), step=100*t)
        mlflow.log_metric("LL", float(scores['LL'][t]), step=100*t)        
        mlflow.log_metric("learning_rate", float(scores['lr'][t]), step=100*t)
    return
        


def run(setup, n_samples_FU, ratio_ood, sigma_input):
    
    setup_ = get_setup( setup)
    setup=setup_.Setup( device) 
    
    loglikelihood=setup.loglikelihood
    
    if ratio_ood is not None:
        projection=lambda x,y: setup.projection(x,y, n_samples_FU, ratio_ood)
    elif sigma_input is not None:
        projection=lambda x,y: setup.projection_normal(x,y, n_samples_FU, sigma_input)
    else:
        raise ValueError('projection not defined, please provide either ratio_ood or sigma_input')
        
#    projection=setup.projection_normal
    size_sample=setup.n_train_samples
    param_count=setup.param_count
    input_dim=setup.input_dim

    batch=np.min([int(size_sample/6),500]) #50 works fine too!

    rho=batch/size_sample #* input_dim
    
    """
    
    if dataset== 'powerplant':
        ratio_ood= 0.1 #0.35 #0.25
    if dataset== 'wine':
        ratio_ood=0.1
    """
    
    """
    if dataset== 'concrete':
        ratio_ood= 0.25 #0.35 #0.25
    
    if dataset == 'boston':
        ratio_ood=0.1 #0.15 #0.2#0.1
    if dataset == 'yacht':
        ratio_ood=0.15#0.1#0.2
    
    if dataset == 'energy':
        ratio_ood=0.1 #0.15#0.1
    
    if dataset== 'wine':
        ratio_ood=0.
    
    """
              
    def prior(n):
        return setup.sigma_prior*torch.randn(size=(n,param_count), device= device)


    GeN_models_dict=[]

    RMSEs=[]
    LPPs=[]
    TIMEs=[]
    PICPs=[]
    MPIWs=[]

    for i in range(10):
        seed=42+i

        setup=setup_.Setup(device,seed) 
        with mlflow.start_run(run_name=str(setup.experiment_name)+'-'+str(i) ,nested=True):
            log_GeNVI_setup(setup,  n_samples_FU, ratio_ood, batch)

            start = timeit.default_timer()

            GeN, log_scores, ELBO = learning(loglikelihood, batch, setup.n_train_samples,
                                                prior, projection, p_norm,
                                                lat_dim, setup.param_count,
                                                NNE,  n_samples_KL,  n_samples_LL,
                                                max_iter,  learning_rate,  min_lr,  patience,
                                                lr_decay,  device, rho=rho)


            stop = timeit.default_timer()
            execution_time = stop - start

            log_GeNVI_run(ELBO, log_scores)


            log_device = 'cpu'
            theta = GeN(1000).detach().to(log_device)

            LPP_test, RMSE_test, _, PICP_test, MPIW_test = setup.evaluate_metrics(theta,log_device)

            RMSEs.append(RMSE_test[0].item())
            LPPs.append(LPP_test[0].item())
            TIMEs.append(execution_time)
            PICPs.append(PICP_test.item())
            MPIWs.append(MPIW_test.item())

            GeN_models_dict.append((i,GeN.state_dict().copy()))


    metrics_dict={('FuNNeVI',dataset):{'RMSE':(np.mean(RMSEs).round(decimals=3),np.std(RMSEs).round(decimals=3)),
                           'LPP': (np.mean(LPPs).round(decimals=3),np.std(LPPs).round(decimals=3)),
                           'PICP':  (np.mean(PICPs).round(decimals=3),np.std(PICPs).round(decimals=3)), 
                           'MPIW':  (np.mean(MPIWs).round(decimals=3),np.std(MPIWs).round(decimals=3))
                           }
                   }
                         
    models={str(i): model for i,model in GeN_models_dict} 
    return models, metrics_dict






parser = argparse.ArgumentParser()
parser.add_argument("--ratio_ood", type=float, default=None,
                    help="sample inputs from uniform and data distribution with ratio_ood")
parser.add_argument("--sigma_input", type=float, default=None,
                    help="sample inputs from Gaussians centered at data input with scale sigma_input")

"""
run the FuNNeVI method on 'datasets'
if ratio_ood is given: 
    it uses 'projection' from each setup 
        ratio_ood: OOD uniform sampling of inputs on hyper-rectangle based train inputs
        1-ratio_ood: training inputs

if sigma_input is given:
    it uses 'projection_normal' from each setup:
        inputs are sampled according to gaussians with scale sigma_input centered at training inputs
        
"""


if __name__ == "__main__":

    args = parser.parse_args()
    print(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    datasets=['boston','concrete', 'energy', 'powerplant',  'wine', 'yacht']

    FuNmodels={}
    results={}
    
    xpname = 'FuNNeVI-splits'
    mlflow.set_experiment(xpname)
    
    with mlflow.start_run():
        log_GeNVI_experiment(args.sigma_input, args.ratio_ood, p_norm, lat_dim, NNE, n_samples_KL, n_samples_LL, 
                         max_iter, learning_rate, min_lr, patience, lr_decay,
                         device)
        
  
        n_samples_FU=500
        for dataset in datasets:
            print(dataset)
            models, metrics_dict=run(dataset, n_samples_FU=n_samples_FU, ratio_ood=args.ratio_ood, sigma_input=args.sigma_input) 
            print(dataset+': done :-)')
            FuNmodels.update({dataset:models})
            results.update(metrics_dict)
            torch.save(FuNmodels, 'Results/SpFuNmodels.pt')
            torch.save(results, 'Results/SpFuNmetrics.pt')

        mlflow.log_artifact('Results/SpFuNmodels.pt')
        mlflow.log_artifact('Results/SpFuNmetrics.pt')

  


