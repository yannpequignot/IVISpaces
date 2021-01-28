import timeit

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import trange

from Metrics import kl_nne, entropy_nne
from Models import GaussianProcess
from Tools import average_normal_loglikelihood, log_diagonal_mvn_pdf

# optimizer
learning_rate = 0.005

# scheduler
lr_decay = .7
min_lr = 0.0001

# loss hyperparameters
n_samples_LL = 100  # nb of predictor samples for average LogLikelihood

n_samples_KL = 500  # nb of predictor samples for KL divergence
kNNE = 1  # k-nearest neighbour

sigma_prior = .5  # Default scale for Gaussian prior on weights of predictive network


class VI_trainer():
    def __init__(self, train_loader, ELBO, optimizer):
        self.train_loader = train_loader
        self.ELBO = ELBO
        self.optimizer = optimizer

    def one_epoch(self, model):
        self.scores = {'ELBO': 0.,
                       'KL': 0.,
                       'LL': 0.,
                       'sigma': 0.
                       }
        example_count = 0.

        model.train(True)
        with torch.enable_grad():
            for (x, y) in self.train_loader:
                self.optimizer.zero_grad()

                L, K, LL = self.ELBO(x, y, model)
                L.backward()

                lr = self.optimizer.param_groups[0]['lr']

                self.optimizer.step()

                self.scores['ELBO'] += L.item() * len(x)
                self.scores['KL'] += K.item() * len(x)
                self.scores['LL'] += LL.item() * len(x)
                self.scores['sigma'] += model.sigma_noise.item() * len(x)

                example_count += len(x)

        mean_scores = {'ELBO': self.scores['ELBO'] / example_count,
                       'KL': self.scores['KL'] / example_count,
                       'LL': self.scores['LL'] / example_count,
                       'sigma': self.scores['sigma'] / example_count,
                       'lr': lr
                       }
        return mean_scores


def NN_train(model, train_dataset, batch_size, n_epochs, patience):
    device = next(model.parameters()).device
    size_data = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def prior(n):
        return sigma_prior * torch.randn(size=(n, model.param_count), device=device)

    def kl(model):
        theta = model.gen(n_samples_KL)  # variational
        theta_prior = prior(n_samples_KL)  # prior
        K = kl_nne(theta, theta_prior, k=kNNE)
        return K

    def ELBO(x_data, y_data, model):
        y_pred = model(x_data, n_samples_LL)
        Average_LogLikelihood = average_normal_loglikelihood(y_pred, y_data, model.sigma_noise)
        the_KL = kl(model)
        the_ELBO = - Average_LogLikelihood + (len(x_data) / size_data) * the_KL
        return the_ELBO, the_KL, Average_LogLikelihood

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, min_lr=min_lr)

    Run = VI_trainer(train_loader, ELBO, optimizer)

    logs = {'ELBO': [], 'LL': [], "KL": [], "lr": [], "sigma": []}
    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc='NN-' + model.name, refresh=False)
        for _ in tr:

            scores = Run.one_epoch(model)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'],
                           sigma=scores['sigma'])
            for key, values in logs.items():
                values.append(scores[key])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start

    return logs, time


def FuNN_train(model, train_dataset, batch_size, input_sampler, n_epochs, patience):
    device = next(model.parameters()).device
    size_data = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def prior(n):
        return sigma_prior * torch.randn(size=(n, model.param_count), device=device)

    def kl(model):
        theta = model.gen(n_samples_KL)  # variationnel
        theta_prior = prior(n_samples_KL)  # prior
        X = input_sampler()  # sample OOD inputs
        theta_proj = model.predictor(X, theta).squeeze(2)  # evaluate predictors at OOD inputs
        theta_prior_proj = model.predictor(X, theta_prior).squeeze(2)  # evaluate predictors at OOD inputs
        K = kl_nne(theta_proj, theta_prior_proj, k=kNNE)  # compute NNe of KL on predictor approximations
        return K

    def ELBO(x_data, y_data, model):
        y_pred = model(x_data, n_samples_LL)
        Average_LogLikelihood = average_normal_loglikelihood(y_pred, y_data, model.sigma_noise)
        the_KL = kl(model)
        the_ELBO = - Average_LogLikelihood + (len(x_data) / size_data) * the_KL
        return the_ELBO, the_KL, Average_LogLikelihood

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, min_lr=min_lr)

    Run = VI_trainer(train_loader, ELBO, optimizer)

    logs = {'ELBO': [], 'LL': [], "KL": [], "lr": [], "sigma": []}
    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc='FuNN-' + model.name, refresh=False)
        for _ in tr:

            scores = Run.one_epoch(model)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'],
                           sigma=scores['sigma'])
            for key, values in logs.items():
                values.append(scores[key])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start

    return logs, time


def BBB_train(model, train_dataset, batch_size, n_epochs, patience):
    size_data = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    def ELBO(x_data, y_data, model):
        y_pred = model(x_data, n_samples_LL)
        Average_LogLikelihood = average_normal_loglikelihood(y_pred, y_data, model.sigma_noise)
        theta = model.gen(n_samples_KL)
        the_KL = model.gen.log_prob(theta).mean() - log_diagonal_mvn_pdf(theta, std=sigma_prior).mean()
        the_ELBO = - Average_LogLikelihood + (len(x_data) / size_data) * the_KL
        return the_ELBO, the_KL, Average_LogLikelihood

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, min_lr=min_lr)

    Run = VI_trainer(train_loader, ELBO, optimizer)

    logs = {'ELBO': [], 'LL': [], "KL": [], "lr": [], "sigma": []}
    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc='MFVI', refresh=False)
        for _ in tr:

            scores = Run.one_epoch(model)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'],
                           sigma=scores['sigma'])
            for key, values in logs.items():
                values.append(scores[key])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start

    return logs, time


def FuNN_train(model, train_dataset, batch_size, input_sampler, n_epochs, patience):
    device = next(model.parameters()).device
    size_data = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    prior = GaussianProcess(mean=torch.tensor(0.), lengthscale=1., noise=0.1).to(device)

    def kl(model):
        theta = model.gen(n_samples_KL)  # variational
        X_ood = input_sampler()
        f_theta = model.predictor(X_ood, theta).squeeze(2)
        H = entropy_nne(f_theta, k_MC=X_ood.shape[0])
        logtarget = prior.log_prob(X_ood, f_theta)
        return -H - logtarget.mean()

    def ELBO(x_data, y_data, model):
        y_pred = model(x_data, n_samples_LL)
        Average_LogLikelihood = average_normal_loglikelihood(y_pred, y_data, model.sigma_noise)
        the_KL = kl(model)
        the_ELBO = - Average_LogLikelihood + (len(x_data) / size_data) * the_KL
        return the_ELBO, the_KL, Average_LogLikelihood

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, min_lr=min_lr)

    Run = VI_trainer(train_loader, ELBO, optimizer)

    logs = {'ELBO': [], 'LL': [], "KL": [], "lr": [], "sigma": []}
    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc='FuNN-' + model.name, refresh=False)
        for _ in tr:

            scores = Run.one_epoch(model)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'],
                           sigma=scores['sigma'])
            for key, values in logs.items():
                values.append(scores[key])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start

    return logs, time
