import os
from datetime import datetime

import pandas as pd
import torch
from torch import nn

from Data import get_setup
from Inference import *
from Metrics import rmse, lpp, batch_entropy_nne, kl_nne, lpp_gaussian
import argparse


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def run_ensemble(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()
    std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()

    model_list, time = ensemble(x_train, y_train, batch_size, layerwidth, activation, 
                                num_epochs=num_epochs_ensemble, num_models=5)

    x_test, y_test = setup.test_data()
    x_ood, y_ood = setup.ood_data()
    X = [x_train, x_test, x_ood]
    Y = [torch.cat([ensemble_predict(x_, model_list) for x_ in torch.split(x,10000)], dim=1)  for x in X]
    metrics_test = get_metrics(Y[1], torch.tensor(0.), y_test, std_y_train, time, gaussian_prediction=True)
    metrics_ood = get_metrics(Y[2], torch.tensor(0.), y_ood, std_y_train, time, gaussian_prediction=True)

    Y_target=[y_train,y_test, y_ood]
    Y_uncertain = [y.mean(0) + y.std(0) * torch.randn(nb_predictions, y.shape[1], 1).to(device) for y in Y]
    H = [torch.cat([batch_entropy_nne(y_, k=30) for y_ in torch.split(y.transpose(0,1),1000,dim=0)]) for y in Y_uncertain]
    SE=[(pred.mean(0)-target)**2 for pred,target in zip(Y,Y_target)]
    RSE=[e.sqrt()*std_y_train for e in SE]
    return (metrics_test,metrics_ood), (H,SE)

def run_MCdropout(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()

    batch_size = len(x_train)
    
    std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()
    
    trainer = MC_Dropout(x_train, y_train, batch_size, layerwidth, init_sigma_noise=1., drop_prob=0.05,
                         learn_noise=True,
                         activation=activation)

    weight_decay = 1e-1 / (10 * len(x_train) // 9) ** 0.5
    time = trainer.fit(num_epochs=n_epochs, learn_rate=1e-3, weight_decay=weight_decay)

    x_test, y_test = setup.test_data()
    x_ood, y_ood = setup.ood_data()
    X = [x_train, x_test, x_ood]
    Y = [trainer.predict(x, nb_predictions) for x in X]
    metrics_test = get_metrics(Y[1][0], Y[1][1], y_test, std_y_train, time, gaussian_prediction=True)
    metrics_ood = get_metrics(Y[2][0], Y[2][1], y_ood, std_y_train, time, gaussian_prediction=True)

    Y_target=[y_train,y_test, y_ood]
    Y_uncertain = [y[0].mean(0) + y[0].std(0) * torch.randn(nb_predictions, y[0].shape[1], 1).to(device) for y in Y]
    H = [torch.cat([batch_entropy_nne(y_, k=30) for y_ in torch.split(y.transpose(0,1),1000,dim=0)]) for y in Y_uncertain]
    SE=[(pred[0].mean(0)-target)**2 for pred,target in zip(Y,Y_target)]
    RSE=[e.sqrt()*std_y_train for e in SE]
    return (metrics_test,metrics_ood), (H,SE)

def run_MFVI(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()
    std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()

    MF_dist, model, sigma_noise, time = MFVI(x_train, y_train, batch_size, layerwidth, nblayers, activation,
                                             n_epochs=n_epochs, sigma_noise_init=1.0, learn_noise=True,patience=2*patience)

    theta = MF_dist(nb_predictions).detach()
    x_test, y_test = setup.test_data()
    x_ood, y_ood = setup.ood_data()
    X = [x_train, x_test, x_ood]
    Y = [torch.cat([model(x_, theta) for x_ in torch.split(x,1000)], dim=1)  for x in X]
    metrics_test = get_metrics(Y[1], sigma_noise, y_test, std_y_train, time, gaussian_prediction=True)
    metrics_ood = get_metrics(Y[2], sigma_noise, y_ood, std_y_train, time, gaussian_prediction=True)

    Y_target=[y_train,y_test, y_ood]
    Y_uncertain = [y.mean(0) + y.std(0) * torch.randn(nb_predictions, y.shape[1], 1).to(device) for y in Y]
    H = [torch.cat([batch_entropy_nne(y_, k=30) for y_ in torch.split(y.transpose(0,1),1000,dim=0)]) for y in Y_uncertain]
    SE=[(pred.mean(0)-target)**2 for pred,target in zip(Y,Y_target)]
    RSE=[e.sqrt()*std_y_train for e in SE]
    return (metrics_test,metrics_ood), (H,SE)

def run_FuNN_MFVI(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()
    x_test, y_test = setup.test_data()
    x_ood, y_ood = setup.ood_data()

    std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()

    def input_sampler(n_ood=200):
        x_data=torch.cat([x_train,x_test,x_ood])
        M = x_data.max(0, keepdim=True)[0]
        m = x_data.min(0, keepdim=True)[0]
        X = torch.rand(n_ood, x_train.shape[1]).to(device) * (M - m) + m
        return X

    MF_dist, model, sigma_noise, time = FuNN_MFVI(x_train, y_train, batch_size, layerwidth, nblayers, activation,
                                                  input_sampler, n_epochs=n_epochs, sigma_noise_init=1.0,
                                                  learn_noise=True, patience=patience)

    theta = MF_dist(nb_predictions).detach()
    x_test, y_test = setup.test_data()
    X = [x_train, x_test, x_ood]
    Y = [torch.cat([model(x_, theta) for x_ in torch.split(x,1000)], dim=1)  for x in X]
    metrics_test = get_metrics(Y[1], sigma_noise, y_test, std_y_train, time, gaussian_prediction=True)
    metrics_ood = get_metrics(Y[2], sigma_noise, y_ood, std_y_train, time, gaussian_prediction=True)

    Y_target=[y_train,y_test, y_ood]
    Y_uncertain = [y.mean(0) + y.std(0) * torch.randn(nb_predictions, y.shape[1], 1).to(device) for y in Y]
    H = [torch.cat([batch_entropy_nne(y_, k=30) for y_ in torch.split(y.transpose(0,1),1000,dim=0)]) for y in Y_uncertain]
    SE=[(pred.mean(0)-target)**2 for pred,target in zip(Y,Y_target)]
    RSE=[e.sqrt()*std_y_train for e in SE]
    return (metrics_test,metrics_ood), (H,SE)

def run_NN_HyVI(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()
    std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()

    gen, model, sigma_noise, time = NN_HyVI(x_train, y_train, batch_size, layerwidth, nblayers, activation,
                                            n_epochs=n_epochs, sigma_noise_init=1.0, learn_noise=True, patience=patience)

    theta = gen(nb_predictions).detach()
    x_test, y_test = setup.test_data()
    x_ood, y_ood = setup.ood_data()
    X = [x_train, x_test, x_ood]
    Y = [torch.cat([model(x_, theta) for x_ in torch.split(x,1000)], dim=1)  for x in X]
    metrics_test = get_metrics(Y[1], sigma_noise, y_test, std_y_train, time, gaussian_prediction=True)
    metrics_ood = get_metrics(Y[2], sigma_noise, y_ood, std_y_train, time, gaussian_prediction=True)

    Y_target=[y_train,y_test, y_ood]
    Y_uncertain = [y.mean(0) + y.std(0) * torch.randn(nb_predictions, y.shape[1], 1).to(device) for y in Y]
    H = [torch.cat([batch_entropy_nne(y_, k=30) for y_ in torch.split(y.transpose(0,1),1000,dim=0)]) for y in Y_uncertain]
    SE=[(pred.mean(0)-target)**2 for pred,target in zip(Y,Y_target)]
    RSE=[e.sqrt()*std_y_train for e in SE]
    return (metrics_test,metrics_ood), (H,SE)

def run_FuNN_HyVI(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()
    x_test, y_test = setup.test_data()
    x_ood, y_ood = setup.ood_data()
    std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()

    def input_sampler(n_ood=200):
        x_data=torch.cat([x_train,x_test,x_ood])
        M = x_data.max(0, keepdim=True)[0]
        m = x_data.min(0, keepdim=True)[0]
        X = torch.rand(n_ood, x_train.shape[1]).to(device) * (M - m) + m
        return X

    gen, model, sigma_noise, time = FuNN_HyVI(x_train, y_train, batch_size, layerwidth, nblayers, activation,
                                              input_sampler, n_epochs=n_epochs, sigma_noise_init=1.0,
                                              learn_noise=True, patience=patience)

    theta = gen(nb_predictions).detach()
    x_test, y_test = setup.test_data()
    X = [x_train, x_test, x_ood]
    Y = [torch.cat([model(x_, theta) for x_ in torch.split(x,1000)], dim=1)  for x in X]
    metrics_test = get_metrics(Y[1], sigma_noise, y_test, std_y_train, time, gaussian_prediction=True)
    metrics_ood = get_metrics(Y[2], sigma_noise, y_ood, std_y_train, time, gaussian_prediction=True)

    Y_target=[y_train,y_test, y_ood]
    Y_uncertain = [y.mean(0) + y.std(0) * torch.randn(nb_predictions, y.shape[1], 1).to(device) for y in Y]
    H = [torch.cat([batch_entropy_nne(y_, k=30) for y_ in torch.split(y.transpose(0,1),1000,dim=0)]) for y in Y_uncertain]
    SE=[(pred.mean(0)-target)**2 for pred,target in zip(Y,Y_target)]
    RSE=[e.sqrt()*std_y_train for e in SE]
    return (metrics_test,metrics_ood), (H,SE)

def run_gp_FuNN_HyVI(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()
    x_test, y_test = setup.test_data()
    x_ood, y_ood = setup.ood_data()


    std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()

    def input_sampler(n_ood=200):
        x_data=torch.cat([x_train,x_test,x_ood])
        M = x_data.max(0, keepdim=True)[0]
        m = x_data.min(0, keepdim=True)[0]
        X = torch.rand(n_ood, x_train.shape[1]).to(device) * (M - m) + m
        return X

    gen, model, sigma_noise, time = GP_FuNN_HyVI(x_train, y_train, batch_size, layerwidth, nblayers, activation,
                                                 input_sampler, n_epochs=n_epochs, sigma_noise_init=1.0,
                                                 learn_noise=True, patience=patience)

    theta = gen(nb_predictions).detach()
    x_test, y_test = setup.test_data()
    X = [x_train, x_test, x_ood]
    Y = [torch.cat([model(x_, theta) for x_ in torch.split(x,1000)], dim=1)  for x in X]
    metrics_test = get_metrics(Y[1], sigma_noise, y_test, std_y_train, time, gaussian_prediction=True)
    metrics_ood = get_metrics(Y[2], sigma_noise, y_ood, std_y_train, time, gaussian_prediction=True)

    Y_target=[y_train,y_test, y_ood]
    Y_uncertain = [y.mean(0) + y.std(0) * torch.randn(nb_predictions, y.shape[1], 1).to(device) for y in Y]
    H = [torch.cat([batch_entropy_nne(y_, k=30) for y_ in torch.split(y.transpose(0,1),1000,dim=0)]) for y in Y_uncertain]
    SE=[(pred.mean(0)-target)**2 for pred,target in zip(Y,Y_target)]
    RSE=[e.sqrt()*std_y_train for e in SE]
    return (metrics_test,metrics_ood), (H,SE)

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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str, default="small",
                        help="small or large")
    args = parser.parse_args()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")

    if args.set == "small":
        ## small ##
        file_name = 'Results/2mean/2mean_smallUCI_Exp2_' + date_string
        log_device=device
        n_epochs = 2000
        num_epochs_ensemble = 3000
        batch_size = 50
        patience=30
        # predictive model architecture
        layerwidth = 50
        nblayers = 1
        activation = nn.ReLU()
        datasets = ['boston2', 'concrete2', 'energy2', 'wine2', 'yacht2']
        Repeat = range(3)
        nb_predictions=1000

    
    if args.set == "large": 
        # large ##
        file_name = 'Results/2mean/UCI_large_Exp2_' + date_string
        log_device='cpu'
        n_epochs = 2000
        num_epochs_ensemble = 500
        batch_size = 500
        patience=10
        # predictive model architecture
        layerwidth = 100
        nblayers = 1
        activation = nn.ReLU()
        datasets =['kin8nm2', 'navalC2', 'powerplant2', 'protein2']
        Repeat = range(3)
        nb_predictions=500

 

    makedirs(file_name)
    with open(file_name, 'w') as f:
        script = open(__file__)
        f.write(script.read())

    RESULTS, STDS = {dataset: {} for dataset in datasets}, {dataset: {} for dataset in datasets}
    RESULTS_ood, STDS_ood = {dataset: {} for dataset in datasets}, {dataset: {} for dataset in datasets}
    PRED_H = {dataset: {} for dataset in datasets}

    
#     RESULTS, STDS = torch.load('Results/2mean/2mean_smallUCI_Exp2_2020-11-19-15:56_metrics.pt')
#     RESULTS_ood, STDS_ood = torch.load('Results/2mean/2mean_smallUCI_Exp2_2020-11-19-15:56_metrics_ood.pt')
#     PRED_H = torch.load('Results/2mean/2mean_smallUCI_Exp2_2020-11-19-15:56_entropy.pt')

    for dataset in datasets:
        print(dataset)

        pred_h = {}
        metrics = {}
        stds = {}
        metrics_ood = {}
        stds_ood = {}

        results = [run_MCdropout(dataset, device) for _ in Repeat]
        mean, std = MeanStd([m[0] for m, h in results], 'McDropOut')
        mean_ood, std_ood = MeanStd([m[1] for m, h in results], 'McDropOut')

        pred_h.update({'McDropOut': [h for m, h in results]})
        metrics.update(mean)
        stds.update(std)
        metrics_ood.update(mean_ood)
        stds_ood.update(std_ood)
        
        results = [run_ensemble(dataset, device) for _ in Repeat]
        mean, std = MeanStd([m[0] for m, h in results], 'Ensemble')
        mean_ood, std_ood = MeanStd([m[1] for m, h in results], 'Ensemble')

        pred_h.update({'Ensemble': [h for m, h in results]})
        metrics.update(mean)
        stds.update(std)
        metrics_ood.update(mean_ood)
        stds_ood.update(std_ood)


        results = [run_NN_HyVI(dataset, device) for _ in Repeat]
        mean, std = MeanStd([m[0] for m, h in results], 'NN-HyVI')
        mean_ood, std_ood = MeanStd([m[1] for m, h in results],'NN-HyVI')

        pred_h.update({'NN-HyVI': [h for m, h in results]})
        metrics.update(mean)
        stds.update(std)
        metrics_ood.update(mean_ood)
        stds_ood.update(std_ood)


        results = [run_FuNN_HyVI(dataset, device) for _ in Repeat]
        mean, std = MeanStd([m[0] for m, h in results], 'FuNN-HyVI')
        mean_ood, std_ood = MeanStd([m[1] for m, h in results],'FuNN-HyVI')

        pred_h.update({'FuNN-HyVI': [h for m, h in results]})
        metrics.update(mean)
        stds.update(std)
        metrics_ood.update(mean_ood)
        stds_ood.update(std_ood)

        results = [run_gp_FuNN_HyVI(dataset, device) for _ in Repeat]
        mean, std = MeanStd([m[0] for m, h in results], 'FuNN-HyVI*')
        mean_ood, std_ood = MeanStd([m[1] for m, h in results],'FuNN-HyVI*')

        pred_h.update({'FuNN-HyVI*': [h for m, h in results]})
        metrics.update(mean)
        stds.update(std)
        metrics_ood.update(mean_ood)
        stds_ood.update(std_ood)
        
        results = [run_MFVI(dataset, device) for _ in Repeat]
        mean, std = MeanStd([m[0] for m, h in results], 'MFVI')
        mean_ood, std_ood = MeanStd([m[1] for m, h in results],'MFVI')

        pred_h.update({'MFVI': [h for m, h in results]})
        metrics.update(mean)
        stds.update(std)
        metrics_ood.update(mean_ood)
        stds_ood.update(std_ood)

        results = [run_FuNN_MFVI(dataset, device) for _ in Repeat]
        mean, std = MeanStd([m[0] for m, h in results], 'FuNN-MFVI')
        mean_ood, std_ood = MeanStd([m[1] for m, h in results],'FuNN-MFVI')

        pred_h.update({'FuNN-MFVI': [h for m, h in results]})
        metrics.update(mean)
        stds.update(std)
        metrics_ood.update(mean_ood)
        stds_ood.update(std_ood)

        RESULTS[dataset].update(metrics)
        STDS[dataset].update(stds)
        RESULTS_ood[dataset].update(metrics_ood)
        STDS_ood[dataset].update(stds_ood)
        PRED_H[dataset].update(pred_h)

        torch.save((RESULTS, STDS), file_name + '_metrics.pt')
        torch.save((RESULTS_ood, STDS_ood), file_name + '_metrics_ood.pt')

        torch.save(PRED_H, file_name + '_entropy.pt')
