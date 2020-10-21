import torch
from torch import nn
import pandas as pd
from datetime import datetime
import os

from Metrics.test_metrics import lpp_gaussian
from Metrics import rmse, lpp, batch_entropy_nne
from Experiments import get_setup
from Inference import *  # NN_HyVI, FuNN_HyVI, FuNN_MFVI, MFVI, ensemble, ensemble_predict, MC_Dropout


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def OOD_sampler(x_train, n_ood):
    M = x_train.max(0, keepdim=True)[0]
    m = x_train.min(0, keepdim=True)[0]
    X = torch.rand(n_ood, x_train.shape[1]).to(device) * (M - m) + m
    return X


def run_ensemble(dataset, device, seed):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device, seed=seed)
    x_train, y_train = setup.train_data()
    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train = torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    model_list, time = ensemble(x_train, y_train, batch_size, layerwidth, activation, num_epochs=num_epochs_ensemble,
                                num_models=5)

    x_test, y_test = setup.test_data()
    y_pred = ensemble_predict(x_test, model_list)
    metrics = get_metrics(y_pred, torch.tensor(0.), y_test, std_y_train, time, gaussian_prediction=True)
    X = [x_train, x_test, OOD_sampler(x_train, 1000)]
    _Y = [ensemble_predict(x, model_list) for x in X]
    Y = [y.mean(0) + y.std(0) * torch.randn(1000, y.shape[1], 1).to(device) for y in _Y]
    H = [batch_entropy_nne(y.transpose(0, 1), k=30) for y in Y]
    return metrics, H


def run_MCdropout(dataset, device, seed):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device, seed=seed)
    x_train, y_train = setup.train_data()

    batch_size = len(x_train)

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train = torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    trainer = MC_Dropout(x_train, y_train, batch_size, layerwidth, init_sigma_noise=1., drop_prob=0.05,
                         learn_noise=True,
                         activation=activation)

    weight_decay = 1e-1 / (10 * len(x_train) // 9) ** 0.5
    time = trainer.fit(num_epochs=n_epochs, learn_rate=1e-3, weight_decay=weight_decay)

    x_test, y_test = setup.test_data()
    y_pred, sigma_noise = trainer.predict(x_test, 1000)
    metrics = get_metrics(y_pred, sigma_noise, y_test, std_y_train, time, gaussian_prediction=True)

    X = [x_train, x_test, OOD_sampler(x_train, 1000)]
    _Y = [trainer.predict(x, 1000)[0] for x in X]
    Y = [y.mean(0) + y.std(0) * torch.randn(1000, y.shape[1], 1).to(device) for y in _Y]
    H = [batch_entropy_nne(y.transpose(0, 1), k=30) for y in Y]
    return metrics, H


def run_MFVI(dataset, device, seed):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device, seed=seed)
    x_train, y_train = setup.train_data()
    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train = torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    MF_dist, model, sigma_noise, time = MFVI(x_train, y_train, batch_size, layerwidth, nblayers, activation,
                                             n_epochs=n_epochs, sigma_noise_init=1.0, learn_noise=True)

    x_test, y_test = setup.test_data()
    theta = MF_dist(1000).detach()
    y_pred = model(x_test, theta)
    metrics = get_metrics(y_pred, sigma_noise, y_test, std_y_train, time)
    X = [x_train, x_test, OOD_sampler(x_train, 1000)]
    Y = [model(x, theta) for x in X]
    H = [batch_entropy_nne(y.transpose(0, 1), k=30) for y in Y]
    return metrics, H


def run_FuNN_MFVI(dataset, device, seed):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device, seed=seed)
    x_train, y_train = setup.train_data()
    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train = torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    def input_sampler(n_ood=200):
        M = x_train.max(0, keepdim=True)[0]
        m = x_train.min(0, keepdim=True)[0]
        X = torch.rand(n_ood, x_train.shape[1]).to(device) * (M - m) + m
        return X

    MF_dist, model, sigma_noise, time = FuNN_MFVI(x_train, y_train, batch_size, layerwidth, nblayers, activation,
                                                  input_sampler, n_epochs=n_epochs, sigma_noise_init=1.0,
                                                  learn_noise=True)

    x_test, y_test = setup.test_data()
    theta = MF_dist(1000).detach()
    y_pred = model(x_test, theta)
    metrics = get_metrics(y_pred, sigma_noise, y_test, std_y_train, time)
    X = [x_train, x_test, OOD_sampler(x_train, 1000)]
    Y = [model(x, theta) for x in X]
    H = [batch_entropy_nne(y.transpose(0, 1), k=30) for y in Y]
    return metrics, H


def run_NN_HyVI(dataset, device, seed):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device, seed=seed)
    x_train, y_train = setup.train_data()
    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train = torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    gen, model, sigma_noise, time = NN_HyVI(x_train, y_train, batch_size, layerwidth, nblayers, activation,
                                            n_epochs=n_epochs, sigma_noise_init=1.0, learn_noise=True)

    x_test, y_test = setup.test_data()
    theta = gen(1000).detach()
    y_pred = model(x_test, theta)
    metrics = get_metrics(y_pred, sigma_noise, y_test, std_y_train, time)
    X = [x_train, x_test, OOD_sampler(x_train, 1000)]
    Y = [model(x, theta) for x in X]
    H = [batch_entropy_nne(y.transpose(0, 1), k=30) for y in Y]
    return metrics, H


def run_FuNN_HyVI(dataset, device, seed):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device, seed=seed)
    x_train, y_train = setup.train_data()
    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train = torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    def input_sampler(n_ood=200):
        M = x_train.max(0, keepdim=True)[0]
        m = x_train.min(0, keepdim=True)[0]
        X = torch.rand(n_ood, x_train.shape[1]).to(device) * (M - m) + m
        return X

    gen, model, sigma_noise, time = FuNN_HyVI(x_train, y_train, batch_size, layerwidth, nblayers, activation,
                                              input_sampler, n_epochs=n_epochs, sigma_noise_init=1.0, learn_noise=True)

    x_test, y_test = setup.test_data()
    theta = gen(1000).detach()
    y_pred = model(x_test, theta)
    metrics = get_metrics(y_pred, sigma_noise, y_test, std_y_train, time)
    X = [x_train, x_test, OOD_sampler(x_train, 1000)]
    Y = [model(x, theta) for x in X]
    H = [batch_entropy_nne(y.transpose(0, 1), k=30) for y in Y]
    return metrics, H


def get_metrics(y_pred, sigma_noise, y_test, std_y_train, time, gaussian_prediction=False):
    metrics = {}
    rmse_test, _ = rmse(y_pred.mean(dim=0), y_test, std_y_train)
    metrics.update({'RMSE': rmse_test})

    if gaussian_prediction:
        lpp_test, _ = lpp_gaussian(y_pred, y_test, sigma_noise, std_y_train)
    else:
        lpp_test, _ = lpp(y_pred, y_test, sigma_noise.view(1, 1, 1), std_y_train)

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")
    file_name = 'Results/NEW/UCI_small_Exp2' + date_string
    makedirs(file_name)

    with open(file_name, 'w') as f:
        script = open(__file__)
        f.write(script.read())

    n_epochs = 1  # 2000
    num_epochs_ensemble = 1  # 3000
    ## small ##
    batch_size = 50
    # predictive model architecture
    layerwidth = 50
    nblayers = 1
    activation = nn.ReLU()
    datasets = ['boston', 'concrete']  # ['boston', 'concrete', 'energy', 'wine', 'yacht']

    ## large ##
    # batch_size = 500
    # layerwidth = 100
    # nblayers = 1
    # activation = nn.ReLU()
    # datasets = ['kin8nm', 'powerplant', 'navalC', 'protein']

    RESULTS, STDS = {dataset: {} for dataset in datasets}, {dataset: {} for dataset in datasets}
    PRED_H = {dataset: {} for dataset in datasets}

    SEEDS = [117 + i for i in range(1)]

    for dataset in datasets:
        print(dataset)

        pred_h = {}
        metrics = {}
        stds = {}

        results = [run_ensemble(dataset, device, seed) for seed in SEEDS]
        mean, std = MeanStd([m for m, h in results], 'Ensemble')
        pred_h.update({'Ensemble': [h for m, h in results]})
        metrics.update(mean)
        stds.update(std)

        results = [run_MCdropout(dataset, device, seed) for seed in SEEDS]
        mean, std = MeanStd([m for m, h in results], 'McDropOut')
        pred_h.update({'McDropOut': [h for m, h in results]})
        metrics.update(mean)
        stds.update(std)

        results = [run_NN_HyVI(dataset, device, seed) for seed in SEEDS]
        mean, std = MeanStd([m for m, h in results], 'NN-HyVI')
        pred_h.update({'NN-HyVI': [h for m, h in results]})
        metrics.update(mean)
        stds.update(std)

        results = [run_FuNN_HyVI(dataset, device, seed) for seed in SEEDS]
        mean, std = MeanStd([m for m, h in results], 'FuNN-HyVI')
        pred_h.update({'FuNN-HyVI': [h for m, h in results]})

        metrics.update(mean)
        stds.update(std)

        results = [run_MFVI(dataset, device, seed) for seed in SEEDS]
        mean, std = MeanStd([m for m, h in results], 'MFVI')
        pred_h.update({'MFVI': [h for m, h in results]})
        metrics.update(mean)
        stds.update(std)

        results = [run_FuNN_MFVI(dataset, device, seed) for seed in SEEDS]
        mean, std = MeanStd([m for m, h in results], 'FuNN-MFVI')
        pred_h.update({'FuNN-MFVI': [h for m, h in results]})
        metrics.update(mean)
        stds.update(std)

        RESULTS[dataset].update(metrics)
        STDS[dataset].update(stds)
        PRED_H[dataset].update(pred_h)

        torch.save((RESULTS, STDS), file_name + '_metrics.pt')
        torch.save(PRED_H, file_name + '_entropy.pt')
