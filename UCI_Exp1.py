import itertools
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn

from Data import get_setup
from Inference import NN_HyVI, FuNN_HyVI
from Metrics import rmse, lpp, batch_entropy_nne, kl_nne, entropy_nne
from Metrics.test_metrics import lpp_gaussian
from Models import get_mlp

def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def run_NN_HyVI(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()
    std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()

    sigma_noise_init = setup.sigma_noise

    gen, model, sigma_noise, time = NN_HyVI(x_train, y_train, batch_size, layerwidth, nblayers, activation,
                                            n_epochs=n_epochs, sigma_noise_init=sigma_noise_init, learn_noise=False)

    x_test, y_test = setup.test_data()
    theta = gen(1000).detach()
    y_pred = model(x_test, theta)
    metrics = get_metrics(y_pred, sigma_noise, y_test, std_y_train, time)
    return metrics, theta


def run_FuNN_HyVI(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()
    std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()

    sigma_noise_init = setup.sigma_noise

    def input_sampler(n_ood=200):
        M = x_train.max(0, keepdim=True)[0]
        m = x_train.min(0, keepdim=True)[0]
        X = torch.rand(n_ood, x_train.shape[1]).to(device) * (M - m) + m
        return X

    gen, model, sigma_noise, time = FuNN_HyVI(x_train, y_train, batch_size, layerwidth, nblayers, activation,
                                              input_sampler, n_epochs=n_epochs, sigma_noise_init=sigma_noise_init,
                                              learn_noise=False)

    x_test, y_test = setup.test_data()
    theta = gen(1000).detach()
    y_pred = model(x_test, theta)
    metrics = get_metrics(y_pred, sigma_noise, y_test, std_y_train, time)
    return metrics, theta


def get_metrics(y_pred, sigma_noise, y_test, std_y_train, time, gaussian_prediction=False):
    metrics = {}
    rmse_test, _ = rmse(y_pred.mean(dim=0).cpu(), y_test.cpu(), std_y_train.cpu())
    metrics.update({'RMSE': rmse_test})

    if gaussian_prediction:
        lpp_test, _ = lpp_gaussian(y_pred.cpu(), y_test.cpu(), sigma_noise.cpu(), std_y_train.cpu())
    else:
        lpp_test, _ = lpp(y_pred.cpu(), y_test.cpu(), sigma_noise.view(1, 1, 1).cpu(), std_y_train.cpu())

    metrics.update({'LPP': lpp_test})
    metrics.update({'time [s]': time})
    metrics.update({'std noise': sigma_noise.item()})
    return metrics


def MeanStd(metric_list, method):
    df = pd.DataFrame(metric_list)
    mean = df.mean().to_dict()
    std = df.std().to_dict()
    metrics = list(mean.keys())
    for j in metrics:
        mean[(method, j)] = mean.pop(j)
        std[(method, j)] = std.pop(j)
    return mean, std


def PredictiveEntropy(theta, dataset):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, _ = setup.train_data()
    x_test, _ = setup.test_data()
    _, model = get_mlp(x_train.shape[1], layerwidth, nblayers, activation)
    X = [x_train[:2000], x_test[:2000], OOD_sampler(x_train, 1000)]
    Y = [model(x, theta) for x in X]
    H = [batch_entropy_nne(y.transpose(0, 1), k=30) for y in Y]
    return H


def OOD_sampler(x_train, n_ood):
    M = x_train.max(0, keepdim=True)[0]
    m = x_train.min(0, keepdim=True)[0]
    X = torch.rand(n_ood, x_train.shape[1]).to(device) * (M - m) + m
    return X


def FunKL(s, t, model, sampler, n=100):
    assert t.shape == s.shape
    KLs = torch.Tensor(n)
    for i in range(n):
        rand_input = sampler()
        t_ = model(rand_input, t).squeeze(2)
        s_ = model(rand_input, s).squeeze(2)
        k = 1
        K = kl_nne(t_, s_, k=k)
        while torch.isinf(K):
            k += 1
            K = kl_nne(t_, s_, k=k)
        KLs[i] = K
    return KLs.mean()


def FunH(s, model, sampler, n=100):
    Hs = torch.Tensor(n)
    for i in range(n):
        rand_input = sampler()
        s_ = model(rand_input, s).squeeze(2)
        k = 1
        H = entropy_nne(s_, k=1, k_MC=200)
        while torch.isinf(H):
            print('infinite entropy!!')
            k += 1
            H = entropy_nne(s_, k=1, k_MC=200)
        Hs[i] = H
    return Hs.mean()

def ComputeEntropy(thetas, dataset, method):
    setup_ = get_setup(dataset)
    device=thetas[0].device
    setup=setup_.Setup(device) 

    entropies={}
    entropies_std={}
    x_train, y_train=setup.train_data()

    sampler= lambda :OOD_sampler(x_train=x_train,n_ood=200)

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
    HMC_=models_HMC[dataset]
    indices = torch.randperm(len(HMC_))[:1000]
    HMC=HMC_[indices].to(device)    
    
    metric=(method,'paramH')
    print(dataset+': '+'paramH')
    Hs=[]
    for theta in thetas:
        H= entropy_nne(theta, k=1, k_MC=1)
        print(H.item())
        Hs.append(H.item())
    
    entropies.update({metric:np.mean(Hs)})
    entropies_std.update({metric:np.std(Hs)})
    
    metric=(method,'funH')
    print(dataset+': '+'funH')
    Hs=[]
    for theta in thetas:
        H= FunH(theta,model,sampler)     
        print(H.item())
        Hs.append(H.item())

    entropies.update({metric:np.mean(Hs)})
    entropies_std.update({metric:np.std(Hs)})    
    return entropies, entropies_std

def paramCompareWithHMC(thetas, dataset, method):
    divergences = {}
    divergences_std = {}
    device = thetas[0].device

    HMC_ = models_HMC[dataset]
    indices = torch.randperm(len(HMC_))[:1000]
    HMC = HMC_[indices].to(device)

    metric = 'KL(-,HMC)'
    print(dataset + ': ' + metric)
    KLs = []
    for theta in thetas:
        K = kl_nne(theta, HMC, k=kNNE)
        print(K.item())
        KLs.append(K.item())

    divergences.update({metric: np.mean(KLs)})
    divergences_std.update({metric: np.std(KLs)})

    metric = 'KL(HMC,-)'
    print(dataset + ': ' + metric)
    KLs = []
    for theta in thetas:
        K = kl_nne(HMC, theta, k=kNNE)
        print(K.item())
        KLs.append(K.item())

    divergences.update({metric: np.mean(KLs)})
    divergences_std.update({metric: np.std(KLs)})

    metric = 'KL(-,-)'
    print(dataset + ': ' + metric)

    models_pairs = list(itertools.combinations(thetas, 2))
    KLs = []
    for theta_0, theta_1 in models_pairs:
        K = kl_nne(theta_0, theta_1, k=kNNE)
        print(K.item())
        KLs.append(K.item())

    divergences.update({metric: np.mean(KLs)})
    divergences_std.update({metric: np.std(KLs)})

    metrics = list(divergences.keys())
    for j in metrics:
        divergences[(method, j)] = divergences.pop(j)
        divergences_std[(method, j)] = divergences_std.pop(j)
    return divergences, divergences_std


def funCompareWithHMC(thetas, dataset, method):
    divergences = {}
    divergences_std = {}
    setup_ = get_setup(dataset)
    setup = setup_.Setup(thetas[0].device)

    x_train, y_train = setup.train_data()
    sampler = lambda: OOD_sampler(x_train=x_train, n_ood=200)
    ## predictive model
    input_dim = x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)

    HMC_ = models_HMC[dataset]
    indices = torch.randperm(len(HMC_))[:1000]
    HMC = HMC_[indices].to(thetas[0].device)

    metric = 'KL(-,HMC)'
    print(dataset + ': ' + metric)
    KLs = []
    for theta in thetas:
        K = FunKL(theta, HMC, model, sampler)
        print(K.item())
        KLs.append(K.item())

    divergences.update({metric: np.mean(KLs)})
    divergences_std.update({metric: np.std(KLs)})

    metric = 'KL(HMC,-)'
    print(dataset + ': ' + metric)
    KLs = []
    for theta in thetas:
        K = FunKL(HMC, theta, model, sampler)
        print(K.item())
        KLs.append(K.item())

    divergences.update({metric: np.mean(KLs)})
    divergences_std.update({metric: np.std(KLs)})

    metric = 'KL(-,-)'
    print(dataset + ': ' + metric)

    models_pairs = list(itertools.combinations(thetas, 2))
    KLs = []
    for theta_0, theta_1 in models_pairs:
        K = FunKL(theta_0, theta_1, model, sampler)
        print(K.item())
        KLs.append(K.item())

    divergences.update({metric: np.mean(KLs)})
    divergences_std.update({metric: np.std(KLs)})

    metrics = list(divergences.keys())
    for j in metrics:
        divergences[(method, j)] = divergences.pop(j)
        divergences_std[(method, j)] = divergences_std.pop(j)
    return divergences, divergences_std


def HMC(dataset,device):
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device) 
    
    x_test, y_test=setup.test_data()

    ## predictive model
    input_dim=x_test.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
    std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()

    HMC_=models_HMC[dataset]
    indices = torch.randperm(len(HMC_))[:1000]
    theta=HMC_[indices].to(device)
    sigma_noise=torch.tensor(setup.sigma_noise)
    y_pred=model(x_test,theta)
    
    metrics=get_metrics(y_pred, sigma_noise, y_test, std_y_train, 'HMC', 0.)
    metrics_keys=list(metrics.keys())
    print(metrics['RMSE'], torch.mean((y_pred.mean(0)-y_test)**2).sqrt()*std_y_train)
    for j in metrics_keys:
        metrics[('HMC',j)] = metrics.pop(j)
    return metrics, theta

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models_HMC = torch.load('Results/HMC_models.pt')
    kNNE = 1

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")
    file_name = 'Results/Exp1/UCI_Exp1_' + date_string
    makedirs(file_name)

    with open(file_name, 'w') as f:
        script = open(__file__)
        f.write(script.read())

    ## small ##

    n_epochs = 2000
    batch_size = 50
    # predictive model architecture
    layerwidth = 50
    nblayers = 1
    activation = nn.ReLU()
    datasets = ['boston', 'concrete', 'energy', 'wine', 'yacht']#
    repeat = range(3)

    RESULTS, STDS = {dataset: {} for dataset in datasets}, {dataset: {} for dataset in datasets}
    PRED_H = {dataset: {} for dataset in datasets}
    DIV, DIV_std = {dataset: {} for dataset in datasets}, {dataset: {} for dataset in datasets}
    pDIV, pDIV_std={dataset:{} for dataset in datasets}, {dataset:{} for dataset in datasets}
    ENT, ENT_std={dataset:{} for dataset in datasets}, {dataset:{} for dataset in datasets}

    for dataset in datasets:
        print(dataset)

        metrics = {}
        stds = {}
        pred_h = {}
        div, div_std = {}, {}
        Pdiv, Pdiv_std = {}, {}
        H, H_std = {}, {}

        method = 'NN-HyVI'
        results = [run_NN_HyVI(dataset, device) for _ in repeat]
        mean, std = MeanStd([m for m, _ in results], method)
        metrics.update(mean), stds.update(std)
        pred_h.update({method: [PredictiveEntropy(theta, dataset) for _, theta in results]})
        Pdivergences, Pdivergences_std = paramCompareWithHMC([theta for _, theta in results], dataset, method)
        Pdiv.update(Pdivergences), Pdiv_std.update(Pdivergences_std)
        divergences, divergences_std = funCompareWithHMC([theta for _, theta in results], dataset, method)
        div.update(divergences), div_std.update(divergences_std)
        entropies, entropies_std = ComputeEntropy([theta for _, theta in results], dataset, method)
        H.update(entropies), H_std.update(entropies_std)

        method = 'FuNN-HyVI'
        results = [run_FuNN_HyVI(dataset, device) for _ in repeat]
        mean, std = MeanStd([m for m, _ in results], method)
        metrics.update(mean), stds.update(std)
        pred_h.update({method: [PredictiveEntropy(theta, dataset) for _, theta in results]})
        Pdivergences, Pdivergences_std = paramCompareWithHMC([theta for _, theta in results], dataset, method)
        Pdiv.update(Pdivergences), Pdiv_std.update(Pdivergences_std)
        divergences, divergences_std = funCompareWithHMC([theta for _, theta in results], dataset, method)
        div.update(divergences), div_std.update(divergences_std)
        entropies, entropies_std = ComputeEntropy([theta for _, theta in results], dataset, method)
        H.update(entropies), H_std.update(entropies_std)

        
        method = 'HMC'
        results = [HMC(dataset,device) for _ in repeat]
        metrics.update(results[0][0])
        pred_h.update({method: [PredictiveEntropy(theta, dataset) for _, theta in results]})
        entropies, entropies_std = ComputeEntropy([theta for _, theta in results], dataset, method)
        H.update(entropies), H_std.update(entropies_std)
        
        RESULTS[dataset].update(metrics)
        STDS[dataset].update(stds)
        PRED_H[dataset].update(pred_h)

        RESULTS[dataset].update(metrics), STDS[dataset].update(stds)
        DIV[dataset].update(div), DIV_std[dataset].update(div_std)
        pDIV[dataset].update(Pdiv), pDIV_std[dataset].update(Pdiv_std)
        ENT[dataset].update(H), ENT_std[dataset].update(H_std)

        torch.save((RESULTS, STDS), file_name + '_metrics.pt')
        torch.save(PRED_H, file_name + '_pred_entropy.pt')
        torch.save([(DIV,DIV_std),(pDIV,pDIV_std)], file_name + '_kldiv.pt')
        torch.save((ENT,ENT_std), file_name + '_post_entropy.pt')
#        [print(key, value) for key, value in RESULTS.items()]