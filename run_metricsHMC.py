from Experiments import get_setup
import numpy as np
import torch

path='Results/' 

metrics=['RMSE','LPP','PICP','MPIW']
datasets=['boston','concrete', 'energy', 'powerplant',  'wine', 'yacht']

def run_metrics(models,dataset, method):
    
    log_device = 'cpu'
    setup_ = get_setup(dataset)
    setup=setup_.Setup(log_device) 
    
    RMSEs=[]
    LPPs=[]
    PICPs=[]
    MPIWs=[]
    
    theta=models[dataset]
    print(theta.shape)

    if dataset == 'powerplant': 
        theta=theta[::5]
        print(theta.shape)
    if dataset == 'wine':
        theta=theta[::2]
        print(theta.shape)
    LPP_test, RMSE_test, _, PICP_test, MPIW_test = setup.evaluate_metrics(theta,log_device)

    RMSEs.append(RMSE_test[0].item())
    LPPs.append(LPP_test[0].item())
    PICPs.append(PICP_test.item())
    MPIWs.append(MPIW_test.item())
    
    metrics_dict={(method,dataset):{'RMSE':(np.mean(RMSEs).round(decimals=3),np.std(RMSEs).round(decimals=3)),
                           'LPP': (np.mean(LPPs).round(decimals=3),np.std(LPPs).round(decimals=3)),
                           'PICP':  (np.mean(PICPs).round(decimals=3),np.std(PICPs).round(decimals=3)), 
                           'MPIW':  (np.mean(MPIWs).round(decimals=3),np.std(MPIWs).round(decimals=3))
                           }
                   }
                 
    
    return metrics_dict

models=torch.load(path+'HMC_models.pt')

results={}

for d in datasets:
        metrics=run_metrics(models,d, 'HMC') 
        print(d+': done :-)')
        results.update(metrics)
        torch.save(results, 'Results/HMCmetrics.pt')

#mlflow.log_artifact('Results/HMCmetrics.pt')
