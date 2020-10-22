import os
from datetime import datetime

import torch
from torch import nn

from Data import get_setup
from Inference import *


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def run_ensemble(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()

    model_list, time = ensemble(x_train, y_train, batch_size, layerwidth, activation, num_epochs=num_epochs_ensemble,
                                num_models=10)

    y_pred_ = ensemble_predict(x_pred, model_list)
    y_mean = y_pred_.mean(axis=0)
    y_sigma = y_pred_.std(axis=0)
    y_pred = y_mean + y_sigma * torch.randn(1000, len(x_pred), 1).to(device) \
             + setup.sigma_noise * torch.randn(1000, len(x_pred), 1).to(device)
    return y_pred


def run_MCdropout(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()
    batch_size = len(x_train)

    model = MC_Dropout(x_train, y_train, batch_size, layerwidth, init_sigma_noise=setup.sigma_noise, drop_prob=0.05,
                       learn_noise=False,
                       activation=activation)

    weight_decay = 1e-1 / (10 * len(x_train) // 9) ** 0.5
    _ = model.fit(num_epochs=n_epochs, learn_rate=1e-3, weight_decay=weight_decay)

    y_pred_, sigma_noise = model.predict(x_pred, 1000)
    y_mean = y_pred_.mean(axis=0)
    y_sigma = y_pred_.std(axis=0)
    y_pred = y_mean + y_sigma * torch.randn(1000, len(x_pred), 1).to(device) \
             + setup.sigma_noise * torch.randn(1000, len(x_pred), 1).to(device)
    return y_pred


def run_MFVI(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()

    MF_dist, model, sigma_noise, time = MFVI(x_train, y_train, batch_size,
                                             layerwidth, nblayers, activation,
                                             n_epochs=n_epochs, sigma_noise_init=setup.sigma_noise, learn_noise=False)

    theta = MF_dist(1000).detach()
    y_pred_ = model(x_pred, theta)
    y_pred = y_pred_ + setup.sigma_noise * torch.randn_like(y_pred_)
    return y_pred


def run_FuNN_MFVI(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()

    def input_sampler(n_ood=200):
        M = -4.
        m = 2.
        X = torch.rand(n_ood, 1).to(device) * (M - m) + m
        return X

    MF_dist, model, sigma_noise, time = FuNN_MFVI(x_train, y_train, batch_size, layerwidth, nblayers, activation,
                                                  input_sampler, n_epochs=n_epochs,
                                                  sigma_noise_init=setup.sigma_noise, learn_noise=False)

    theta = MF_dist(1000).detach()
    y_pred_ = model(x_pred, theta)
    y_pred = y_pred_ + setup.sigma_noise * torch.randn_like(y_pred_)
    return y_pred


def run_NN_HyVI(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()

    gen, model, sigma_noise, time = NN_HyVI(x_train, y_train, batch_size,
                                            layerwidth, nblayers, activation,
                                            n_epochs=n_epochs, sigma_noise_init=setup.sigma_noise, learn_noise=False)

    theta = gen(1000).detach()
    y_pred_ = model(x_pred, theta)
    y_pred = y_pred_ + setup.sigma_noise * torch.randn_like(y_pred_)
    return y_pred

def run_FuNN_HyVI(dataset, device):
    setup_ = get_setup(dataset)
    setup = setup_.Setup(device)
    x_train, y_train = setup.train_data()

    def input_sampler(n_ood=200):
        M = -4.
        m = 2.
        X = torch.rand(n_ood, 1).to(device) * (M - m) + m
        return X

    gen, model, sigma_noise, time = FuNN_HyVI(x_train, y_train, batch_size,
                                              layerwidth, nblayers, activation,
                                              input_sampler, n_epochs=n_epochs, sigma_noise_init=setup.sigma_noise,
                                              learn_noise=False)

    theta = gen(1000).detach()
    y_pred_ = model(x_pred, theta)
    y_pred = y_pred_ + setup.sigma_noise * torch.randn_like(y_pred_)
    return y_pred


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")

    ## synthetic 1D data ##
    file_name = 'Results/NEW/WaveOOD_' + date_string
    n_epochs = 2000
    num_epochs_ensemble = 3000
    batch_size = 50
    # predictive model architecture
    layerwidth = 50
    nblayers = 1
    activation = nn.Tanh()

    dataset = 'foong'

    makedirs(file_name)
    with open(file_name, 'w') as f:
        script = open(__file__)
        f.write(script.read())

    RESULTS = {}

    x_pred=torch.linspace(-4.,2.,500).unsqueeze(-1).to(device)

    y_pred = run_ensemble(dataset, device)
    RESULTS.update({'Ensemble':y_pred})

    y_pred = run_MCdropout(dataset, device)
    RESULTS.update({'McDropOut':y_pred})

    y_pred = run_NN_HyVI(dataset, device)
    RESULTS.update({'NN-HyVI':y_pred})

    y_pred = run_FuNN_HyVI(dataset, device)
    RESULTS.update({'FuNN-HyVI':y_pred})

    y_pred = run_MFVI(dataset, device)
    RESULTS.update({'MFVI':y_pred})

    y_pred = run_FuNN_MFVI(dataset, device)
    RESULTS.update({'FuNN-MFVI':y_pred})

    torch.save((x_pred,RESULTS), file_name + '.pt')
