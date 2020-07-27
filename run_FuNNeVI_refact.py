import numpy as np
import math
import torch
from torch import nn

import argparse

import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt

from Models import BigGenerator

from datetime import datetime

from torch.utils.data import Dataset

from Models import get_mlp
from Tools import NormalLogLikelihood
from Metrics import KL, evaluate_metrics

from Experiments import get_setup

from Inference.IVI import IVI

from tqdm import trange

import mlflow
import timeit


datasets=['boston','concrete', 'energy', 'powerplant',  'wine', 'yacht']


#batch_size
#batch_size=50

#predictive model
layerwidth=50
nblayers=1
activation=nn.ReLU()

#generative model
lat_dim=5


#optimizer
learning_rate=0.005

#scheduler
patience=20
lr_decay=.5#.7
min_lr= 0.0001
n_epochs=600



#loss hyperparameters
n_samples_LL=100 #nb of predictor samples for average LogLikelihood

n_samples_KL=500 #nb of predictor samples for KL divergence
kNNE=1 #k-nearest neighbour

n_samples_FU=200 #number of OOD inputs for evaluation of the KL in predictor space



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")


#loss function




def log_experiment(lat_dim, n_samples_KL, n_samples_LL, n_samples_FU,
                   learning_rate, min_lr, patience, lr_decay, n_epochs, device):
    
    mlflow.set_tag('device', device) 

    mlflow.log_param('lat_dim', lat_dim)
    
    mlflow.set_tag('batch_size', 'min(size_data/6,500)')

    mlflow.log_param('n_samples_FU', n_samples_FU)


    mlflow.log_param('n_samples_KL', n_samples_KL)
    mlflow.log_param('n_samples_LL', n_samples_LL)
    

    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('patience', patience)
    mlflow.log_param('lr_decay', lr_decay)
    mlflow.log_param('n_epochs', n_epochs)
    mlflow.log_param('min_lr', min_lr)
    return

def run(dataset):
    #setup
    print(dataset)

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 

    
    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    train_input = torch.utils.data.TensorDataset(x_train)

    
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)

    sigma_noise=setup.sigma_noise

    size_data=len(train_dataset)
    batch_size=int(np.min([size_data/6,500]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    sigma_prior=setup.sigma_prior

    #average log likelihood loss
    def loss(x_data, y_data, GeN):
        r"""

        Parameters:
        x_data (Tensor): tensor of size N X D
        y_data (Tensor): tensor of size N X 1
        GeN: hypernet generating weights for primary network 'model'


        Returns:
        (float):   mean of loglikelihood
        """

        y_pred=model(x_data,GeN(n_samples_LL))
        log_proba=NormalLogLikelihood(y_pred, y_data, sigma_noise)
        return log_proba.mean()
    
    #predictor space KL
    def projection(theta0,theta1):
        #batch sample OOD   
        n_ood=n_samples_FU
        epsilon=0.1
        M = x_train.max(0, keepdim=True)[0]+epsilon
        m = x_train.min(0, keepdim=True)[0]-epsilon
        X_ood = torch.rand(n_ood,input_dim).to(device) * (M-m) + m    

        X=X_ood

        #compute projection on both paramters with model
        theta0_proj=model(X, theta0).squeeze(2)
        theta1_proj=model(X, theta1).squeeze(2)
        return theta0_proj, theta1_proj


    def prior(n):
        return sigma_prior*torch.randn(size=(n,param_count), device=device)

    if method == 'FuNNeVI':
        def kl(GeN):

            theta=GeN(n_samples_KL) #variationnel
            theta_prior=prior(n_samples_KL) #prior

            theta_proj, theta_prior_proj = projection(theta, theta_prior)

            K=KL(theta_proj, theta_prior_proj,k=kNNE,device=device)
            return K
    else:
        def kl(GeN):

            theta=GeN(n_samples_KL) #variationnel
            theta_prior=prior(n_samples_KL) #prior

            K=KL(theta, theta_prior,k=kNNE,device=device)
            return K
        
    #ELBO
    def ELBO(x_data, y_data, GeN):
        Average_LogLikelihood=loss(x_data, y_data, GeN)
        the_KL=kl(GeN)
        the_ELBO= - Average_LogLikelihood+ (len(x_data)/size_data)*the_KL
        return the_ELBO, the_KL, Average_LogLikelihood 

   
    #generative model
    GeN = BigGenerator(lat_dim,param_count,device).to(device)

    #optimizer 
    optimizer = torch.optim.Adam(GeN.parameters(), lr=learning_rate)

    FuN=IVI(train_loader, ELBO, optimizer)

    #scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, verbose=True, min_lr=min_lr)

    KLs=[]
    ELBOs=[]
    LRs=[]
    
    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        for t in tr:
            scores=FuN.run(GeN)
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'])
            ELBOs.append(scores['ELBO'])
            KLs.append(scores['KL'])
            LRs.append(scores['lr'])
            scheduler.step(scores['ELBO'])

    stop = timeit.default_timer()
    execution_time = stop - start
    
    FuNmodels.update({dataset: GeN.state_dict().copy()})
    
    #compute metrics on test
    
    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    log_device='cpu'
    theta=GeN(2000).detach()
    metrics=evaluate_metrics(theta, model, x_test, y_test, sigma_noise, std_y_train, device='cpu')
    results.update({dataset:metrics})
    results[dataset].update({'time':execution_time})
    
    torch.save({'ELBO':ELBOs,'KL':KLs, 'LR':LRs}, 'Results/'+dataset+'_'+date_string+'_training.pt')
    print(dataset)
    for m, r in metrics.items():
        print(m+': '+str(r))
    
parser = argparse.ArgumentParser()
parser.add_argument("--method", type=int, default=0,
                    help="0 for GeNNeVI, 1 for FuNNeVI")    
if __name__ == "__main__":

    FuNmodels={}
    results={}
    args = parser.parse_args()
    
    if args.method == 1:
        method='FuNNeVI'
    else:
        method='GeNNeVI'
    
    xpname = method
    mlflow.set_experiment(xpname)
    
    
    with mlflow.start_run():
        log_experiment(lat_dim, n_samples_KL, n_samples_LL, n_samples_FU,
                       learning_rate, min_lr, patience, lr_decay, n_epochs, device)
        for dataset in datasets:
            run(dataset)
    
        torch.save(FuNmodels, 'Results/FuNrefact_'+date_string+'.pt') 
        torch.save(results, 'Results/FuNrefact_'+date_string+'_results.pt') 
        mlflow.log_artifact('Results/FuNrefact_'+date_string+'.pt')
        mlflow.log_artifact('Results/FuNrefact_'+date_string+'_results.pt') 
  
 
