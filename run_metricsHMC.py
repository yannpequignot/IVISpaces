from Experiments import get_setup
import numpy as np
import torch

from Metrics import evaluate_metrics


path='Results/' 

metrics=['RMSE','LPP','PICP','MPIW']
datasets=['boston','concrete', 'energy', 'powerplant',  'wine', 'yacht']

def run_metrics(models,dataset, method):
    
    log_device = 'cpu'
    setup_ = get_setup(dataset)
    setup=setup_.Setup(log_device) 
    model=setup._model
    x_test, y_test=setup.test_data()
    sigma_noise=setup.sigma_noise
    
    theta=models[dataset]
    print(theta.shape)

    if dataset == 'powerplant': 
        theta=theta[::5]
        print(theta.shape)
    if dataset == 'wine':
        theta=theta[::2]
        print(theta.shape)
#    LPP_test, RMSE_test, _, PICP_test, MPIW_test = setup.evaluate_metrics(theta,log_device)

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_).squeeze().float()

    log_device='cpu'
    metrics=evaluate_metrics(theta, model, x_test, y_test, sigma_noise, std_y_train, device='cpu')
    results.update({dataset:metrics})
                 
    return 

models=torch.load(path+'HMC_models.pt')

results={}

for d in datasets:
    run_metrics(models,d, 'HMC') 
    print(d+': done :-)')
    torch.save(results, 'Results/HMCmetrics.pt')

