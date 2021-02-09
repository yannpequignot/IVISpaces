import os
from datetime import datetime

from Data import get_setup
from Inference import *
from Models.VI import *
from Tools import uniform_rect_sampler
import argparse


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", type=str, default="small",
                        help="large or small")
    parser.add_argument("--device", type=str, default="cpu",
                        help="device")
    args = parser.parse_args()
    print(args)
        
    device = torch.device(args.device)# if torch.cuda.is_available() else 'cpu')

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")

    path="Results / Paper / Exp1 /"

    if args.set == "small":

        ## small ##
        file_name = 'Exp1_small' + date_string
        nb_input_samples = 200
        n_epochs = 2000
        num_epochs_ensemble = 1000
        batch_size = 50
        patience=30
        # predictive model architecture
        layerwidth = 50
        nblayers = 1
        activation = nn.ReLU()
        lat_dim=5

        datasets = ['boston', 'concrete', 'energy', 'wine', 'yacht']

    if args.set == "large":
    # large ##
        file_name = 'Exp1_large' + date_string

        nb_input_samples = 200
        n_epochs = 2000
        num_epochs_ensemble = 500
        batch_size = 500
        patience=10
        # predictive model architecture
        layerwidth = 100
        nblayers = 1
        activation = nn.ReLU()
        lat_dim=5

        datasets = ['navalC', 'powerplant', 'protein']
    
    repeat = range(3)

    MODELS = {dataset: [] for dataset in datasets}

    makedirs(path+file_name)
    with open(path+file_name, 'w') as f:
        script = open(__file__)
        f.write(script.read())

    for dataset in datasets:
        for i in repeat:
            split = {}
            setup_ = get_setup(dataset)
            setup = setup_.Setup(device)
            x_train, y_train = setup.train_data()

            input_dim = x_train.shape[1]
            train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
            sigma_noise_init = setup.sigma_noise


            std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()

            model = ensemble(input_dim, layerwidth, activation, num_models=5).to(device)
            logs, run_time = ensemble_train(model.model_list, train_dataset, batch_size, num_epochs=num_epochs_ensemble)
            split.update({"Ensemble": [model.state_dict(), logs, run_time]})
            MODELS[dataset][i][1].update({"Ensemble": [model.state_dict(), logs, run_time]})

            model = MC_Dropout(input_dim, 1, layerwidth, init_sigma_noise=sigma_noise_init, drop_prob=0.05, learn_noise=False,
                               activation=activation).to(device)
            logs, run_time = MCdo_train(model, train_dataset, batch_size, num_epochs=n_epochs, learn_rate=1e-3,
                                        weight_decay=1e-1 / (10 * len(x_train) // 9) ** 0.5,
                                       desc="MCdo/"+dataset)
            MODELS[dataset][i][1].update({"MC dropout": [model.state_dict(), logs, run_time]})

            model = MFVI(input_dim, layerwidth, nblayers, activation, init_sigma_noise=sigma_noise_init, learn_noise=False,
                         std_mu_init=1., sigma_init=0.001).to(device)
            logs, time = BBB_train(model, train_dataset, batch_size, n_epochs=n_epochs, patience=2 * patience, desc=model.name+"/"+dataset)
            MODELS[dataset][i][1].update({"MFVI": [model.state_dict(), logs, time]})

            model = HyVI(input_dim, layerwidth, nblayers, activation, init_sigma_noise=sigma_noise_init, learn_noise=False,
                         lat_dim=5).to(device)
            logs, time = NN_train(model, train_dataset, batch_size, n_epochs=n_epochs, patience=patience, desc=model.name+"/"+dataset)
            MODELS[dataset][i][1].update({"NN-HyVI": [model.state_dict(), logs, time]})

            # define input sampler for predictor distance estimation
            input_sampler = uniform_rect_sampler(x_train, n=nb_input_samples)

            model = HyVI(input_dim, layerwidth, nblayers, activation, init_sigma_noise=sigma_noise_init, learn_noise=False,
                         lat_dim=5).to(device)
            logs, time = FuNN_train(model, train_dataset, batch_size, input_sampler, n_epochs=n_epochs,
                                    patience=patience, desc=model.name+"/"+dataset)
            MODELS[dataset][i][1].update({"FuNN-HyVI": [model.state_dict(), logs, time]})

            model = MFVI(input_dim, layerwidth, nblayers, activation, init_sigma_noise=sigma_noise_init, learn_noise=False,
                         std_mu_init=1., sigma_init=0.001).to(device)
            logs, time = FuNN_train(model, train_dataset, batch_size, input_sampler, n_epochs=n_epochs,
                                    patience=patience, desc=model.name+"/"+dataset)
            split.update({"FuNN-MFVI": [model.state_dict(), logs, time]})


            x_test, y_test = setup.test_data()
            data = {"train": (x_train, y_train),
                    "test": (x_test, y_test),
                    "scaler_y": torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()}
            MODELS[dataset].append((data, split))

            torch.save(MODELS, pathh+file_name + '_models.pt')
