import os
from datetime import datetime

import torch
from torch import nn

from Data import get_setup
from Inference_new import *
from Models.VI import *
from Tools import uniform_rect_sampler
import argparse


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


#
# def run_MCdropout(dataset, device, seed):
#     setup_ = get_setup(dataset)
#     setup = setup_.Setup(device, seed=seed)
#     x_train, y_train = setup.train_data()
#
#     batch_size = len(x_train)
#
#     std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()
#
#     trainer = MC_Dropout(x_train, y_train, batch_size, layerwidth, init_sigma_noise=1., drop_prob=0.05,
#                          learn_noise=True,
#                          activation=activation)
#
#     weight_decay = 1e-1 / (10 * len(x_train) // 9) ** 0.5
#     time = trainer.fit(num_epochs=n_epochs, learn_rate=1e-3, weight_decay=weight_decay)
#
#     x_test, y_test = setup.test_data()
#     y_pred, sigma_noise = trainer.predict(x_test, 1000)
#     metrics = get_metrics(y_pred, sigma_noise, y_test, std_y_train, time, gaussian_prediction=True)
#
#     X = [x_train[:nb_samples_H], x_test[:nb_samples_H], OOD_sampler(x_train, nb_samples_H)]
#     _Y = [trainer.predict(x, 1000)[0] for x in X]
#     Y = [y.mean(0) + y.std(0) * torch.randn(1000, y.shape[1], 1).to(device) for y in _Y]
#     H = [batch_entropy_nne(y.transpose(0, 1), k=30) for y in Y]
#     return metrics, H
#
#
# def run_gp_FuNN_HyVI(dataset, device):
#     setup_ = get_setup(dataset)
#     setup = setup_.Setup(device)
#     x_train, y_train = setup.train_data()
#
#
#     std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()
#
#     def input_sampler(n_ood=200):
#         M = x_train.max(0, keepdim=True)[0]
#         m = x_train.min(0, keepdim=True)[0]
#         X = torch.rand(n_ood, x_train.shape[1]).to(device) * (M - m) + m
#         return X
#
#     gen, model, sigma_noise, time = GP_FuNN_HyVI(x_train, y_train, batch_size, layerwidth, nblayers, activation,
#                                                  input_sampler, n_epochs=n_epochs, sigma_noise_init=1.0,
#                                                  learn_noise=True, patience=patience)
#
#     x_test, y_test = setup.test_data()
#     theta = gen(1000).detach()
#     y_pred = model(x_test.cpu(), theta.cpu())
#     metrics = get_metrics(y_pred, sigma_noise, y_test, std_y_train, time)
#     X = [x_train[:nb_samples_H], x_test[:nb_samples_H], OOD_sampler(x_train, nb_samples_H)]
#     Y = [model(x, theta) for x in X]
#     H = [batch_entropy_nne(y.transpose(0, 1), k=30) for y in Y]
#     return metrics, H

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str, default="small",
                        help="large or small")
    args = parser.parse_args()


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")


    if args.set == "small":
        ## small ##
        file_name = 'Results/Exp2/Exp2_small' + date_string
        log_device=device
        nb_input_samples = 200
        n_epochs = 2000
        num_epochs_ensemble = 3000
        batch_size = 50
        patience=30
        # predictive model architecture
        layerwidth = 50
        nblayers = 1
        activation = nn.ReLU()
        datasets = ['boston', 'concrete', 'energy', 'wine', 'yacht']
        SEEDS = [117 + i for i in range(10)]

    if args.set == "large":
        # large ##
        file_name = 'Results/Exp2/Exp2_large' + date_string
        log_device='cpu'
        nb_input_samples = 200
        n_epochs = 2000
        num_epochs_ensemble = 500
        batch_size = 500
        patience=10
        # predictive model architecture
        layerwidth = 100
        nblayers = 1
        activation = nn.ReLU()
        datasets = ['kin8nm', 'navalC', 'powerplant', 'protein']
        SEEDS = [117 + i for i in range(5)]

    makedirs(file_name)
    with open(file_name, 'w') as f:
        script = open(__file__)
        f.write(script.read())

    MODELS = {dataset: [] for dataset in datasets}

    for dataset in datasets:
        for seed in SEEDS:
            setup_ = get_setup(dataset)
            setup = setup_.Setup(device, seed=seed)
            x_train, y_train = setup.train_data()

            input_dim = x_train.shape[1]
            train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

            std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()

            split={}

            model = ensemble(input_dim, layerwidth, activation, num_models=5).to(device)
            logs, time = ensemble_train(model.model_list, train_dataset, batch_size, num_epochs=num_epochs_ensemble)
            split.update({"MFVI": [model.state_dict(), logs, time]})

            model = MFVI(input_dim, layerwidth, nblayers, activation, init_sigma_noise=1., learn_noise=True, std_mu_init=1., sigma_init=0.001).to(device)
            logs, time = BBB_train(model, train_dataset, batch_size,  n_epochs=n_epochs)
            split.update({"MFVI": [model.state_dict(), logs, time]})


            model = HyVI(input_dim, layerwidth, nblayers, activation, init_sigma_noise=1., learn_noise=True, lat_dim=5).to(device)
            logs, time = NN_train(model, train_dataset, batch_size, n_epochs=n_epochs)
            split.update({"NN-HyVI":[model.state_dict(), logs, time]})

            #define input sampler for predictor distance estimation
            input_sampler = uniform_rect_sampler(x_train, n=nb_input_samples)

            model = HyVI(input_dim, layerwidth, nblayers, activation, init_sigma_noise=1., learn_noise=True, lat_dim=5).to(device)
            logs, time = FuNN_train(model, train_dataset, batch_size, input_sampler, n_epochs=n_epochs)
            split.update({"FuNN-HyVI": [model.state_dict(), logs, time]})

            model = MFVI(input_dim, layerwidth, nblayers, activation, init_sigma_noise=1., learn_noise=True, std_mu_init=1., sigma_init=0.001).to(device)
            logs, time = FuNN_train(model, train_dataset, batch_size, input_sampler, n_epochs=n_epochs)
            split.update({"FuNN-MFVI": [model.state_dict(), logs, time]})

            x_test, y_test = setup.test_data()
            split.update({"train":(x_train,y_train),
                          "test":(x_test,y_test)})
            MODELS[dataset].append(split)
            torch.save(MODELS, file_name + '_models.pt')

