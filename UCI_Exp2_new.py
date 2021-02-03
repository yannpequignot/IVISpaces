import os
from datetime import datetime

from Data import get_setup
from Inference_new import *
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
    parser.add_argument("--dataset", type=str, default=None,
                        help="dataset")
    args = parser.parse_args()
        
    device = torch.device(args.device)# if torch.cuda.is_available() else 'cpu')

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")


    if args.set == "small":
        ## small ##
        file_name = 'Results/Paper/Exp2/Exp2_small' + date_string
        #MODELS=torch.load("Results/Paper/Exp2/Exp2_small2021-01-28-00:59"+"_models.pt", map_location=device)
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
        lat_dim=5

        #datasets = ['boston', 'concrete', 'energy', 'wine', 'yacht']
        SEEDS = [117 + i for i in range(10)]


    if args.set == "large":
        # large ##
        file_name = 'Results/Paper/Exp2/Exp2_large' + date_string
        #MODELS=torch.load("Results/Paper/Exp2/Exp2_large2021-02-01-16:43"+"_models.pt", map_location=device)

        log_device='cpu'
        nb_input_samples = 200
        n_epochs = 2000
        num_epochs_ensemble = 500
        batch_size = 500
        patience=20
        # predictive model architecture
        layerwidth = 100
        nblayers = 1
        activation = nn.ReLU()
        lat_dim=10
        
        datasets = ['navalC', 'powerplant', 'protein']
        SEEDS = [117 + i for i in range(5)]

    makedirs(file_name)
    with open(file_name, 'w') as f:
        script = open(__file__)
        f.write(script.read())

    MODELS = {dataset: [] for dataset in datasets}
    
    for dataset in datasets:
    #dataset=args.dataset
        for seed in SEEDS:
            setup_ = get_setup(dataset)
            setup = setup_.Setup(device, seed=seed)
            x_train, y_train = setup.train_data()

            input_dim = x_train.shape[1]
            train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

            std_y_train = torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()

          #  split = {}

    #             model = ensemble(input_dim, layerwidth, activation, num_models=5).to(device)
    #             logs, run_time = ensemble_train(model.model_list, train_dataset, batch_size, num_epochs=num_epochs_ensemble)
    #             split.update({"Ensemble": [model.state_dict(), logs, run_time]})

    #             model = MC_Dropout(input_dim, 1, layerwidth, init_sigma_noise=1., drop_prob=0.05, learn_noise=True,
    #                                activation=activation).to(device)
    #             logs, run_time = MCdo_train(model, train_dataset, batch_size, num_epochs=n_epochs, learn_rate=1e-3,
    #                                     weight_decay=1e-1 / (10 * len(x_train) // 9) ** 0.5)
    #             split.update({"MC dropout": [model.state_dict(), logs, run_time]})


            model = MFVI(input_dim, layerwidth, nblayers, activation, init_sigma_noise=1., learn_noise=True,
                         std_mu_init=1., sigma_init=0.001).to(device)
            logs, run_time = BBB_train(model, train_dataset, batch_size, n_epochs=n_epochs, patience=2 * patience)
            MODELS[dataset][seed-117][1].update({"MFVI": [model.state_dict(), logs, run_time]})

            model = HyVI(input_dim, layerwidth, nblayers, activation, init_sigma_noise=1., learn_noise=True,
                         lat_dim=lat_dim).to(device)
            logs, run_time = NN_train(model, train_dataset, batch_size, n_epochs=n_epochs, patience=patience)
            MODELS[dataset][seed-117][1].update({"NN-HyVI": [model.state_dict(), logs, run_time]})

            # define input sampler for predictor distance estimation
            input_sampler = uniform_rect_sampler(x_train, n=nb_input_samples)

            model = HyVI(input_dim, layerwidth, nblayers, activation, init_sigma_noise=1., learn_noise=True,
                         lat_dim=lat_dim).to(device)
            logs, run_time = FuNN_train(model, train_dataset, batch_size, input_sampler, n_epochs=n_epochs,
                                    patience=patience)
            MODELS[dataset][seed-117][1].update({"FuNN-HyVI": [model.state_dict(), logs, run_time]})

            model = MFVI(input_dim, layerwidth, nblayers, activation, init_sigma_noise=1., learn_noise=True,
                         std_mu_init=1., sigma_init=0.001).to(device)
            logs, run_time = FuNN_train(model, train_dataset, batch_size, input_sampler, n_epochs=n_epochs,
                                    patience=patience)
            MODELS[dataset][seed-117][1].update({"FuNN-MFVI": [model.state_dict(), logs, run_time]})

    #             x_test, y_test = setup.test_data()
    #             data = {"train": (x_train, y_train),
    #                     "test": (x_test, y_test),
    #                     "scaler_y": torch.tensor(setup.scaler_y.scale_, device=device).squeeze().float()}
    #            # MODELS[dataset].append((data, split))
            torch.save(MODELS, file_name + '_models.pt')
