import torch
from tqdm import trange
import timeit
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Models import get_mlp, BigGenerator, MeanFieldVariationalDistribution
from Metrics import kl_nne
from Tools import average_normal_loglikelihood, log_diagonal_mvn_pdf
from Inference.VI_trainer import IVI

# generative model
lat_dim = 5

# optimizer
learning_rate = 0.005

# scheduler
patience = 30
lr_decay = .7
min_lr = 0.0001
n_epochs = 2000

# loss hyperparameters
n_samples_LL = 100  # nb of predictor samples for average LogLikelihood

n_samples_KL = 500  # nb of predictor samples for KL divergence
kNNE = 1  # k-nearest neighbour

sigma_prior = .5  # Default scale for Gaussian prior on weights of predictive network


def NN_HyVI(x_train, y_train, batch_size, layerwidth, nblayers, activation, n_epochs=n_epochs, sigma_noise_init=1.0,
            learn_noise=True):
    device = x_train.device
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # set predictive model
    input_dim = x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)

    # variational distribution: generative model hypernet
    gen = BigGenerator(lat_dim, param_count, device).to(device)

    def prior(n):
        return sigma_prior * torch.randn(size=(n, param_count), device=device)

    def kl(gen):
        theta = gen(n_samples_KL)  # variational
        theta_prior = prior(n_samples_KL)  # prior
        K = kl_nne(theta, theta_prior, k=kNNE)
        return K

    def ELBO(x_data, y_data, gen, _sigma_noise):
        y_pred = model(x_data, gen(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)
        Average_LogLikelihood = average_normal_loglikelihood(y_pred, y_data, sigma_noise)
        the_KL = kl(gen)
        the_ELBO = - Average_LogLikelihood + (len(x_data) / size_data) * the_KL  # (len(x_data)/size_data)*the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise

    ## Parametrize noise for learning aleatoric uncertainty

    if learn_noise:
        _sigma_noise = torch.log(torch.tensor(sigma_noise_init).exp() - 1.).clone().to(device).detach().requires_grad_(
            learn_noise)
        optimizer = torch.optim.Adam(list(gen.parameters()) + [_sigma_noise], lr=learning_rate)
    else:
        _sigma_noise = torch.log(torch.tensor(sigma_noise_init).exp() - 1.)
        optimizer = torch.optim.Adam(gen.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, min_lr=min_lr)

    Run = IVI(train_loader, ELBO, optimizer)

    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc='NN-HyVI', refresh=False)
        for _ in tr:

            scores = Run.one_epoch(gen, _sigma_noise)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'],
                           sigma=scores['sigma'])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach().cpu()

    return gen, model, sigma_noise, time


def FuNN_HyVI(x_train, y_train, batch_size, layerwidth, nblayers, activation, input_sampler, n_epochs=n_epochs,
              sigma_noise_init=1.0,
              learn_noise=True):
    # setup data
    device = x_train.device
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # set predictive model
    input_dim = x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)

    # variational distribution: generative model hypernet
    gen = BigGenerator(lat_dim, param_count, device).to(device)

    def prior(n):
        return sigma_prior * torch.randn(size=(n, param_count), device=device)

    def kl(gen):
        theta = gen(n_samples_KL)  # variationnel
        theta_prior = prior(n_samples_KL)  # prior
        X = input_sampler()  # sample OOD inputs
        theta_proj = model(X, theta).squeeze(2)  # evaluate predictors at OOD inputs
        theta_prior_proj = model(X, theta_prior).squeeze(2)  # evaluate predictors at OOD inputs
        K = kl_nne(theta_proj, theta_prior_proj, k=kNNE)  # compute NNe of KL on predictor approximations
        return K

    def ELBO(x_data, y_data, gen, _sigma_noise):
        y_pred = model(x_data, gen(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood = average_normal_loglikelihood(y_pred, y_data, sigma_noise)
        the_KL = kl(gen)
        the_ELBO = - Average_LogLikelihood + (len(x_data) / size_data) * the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise

    ## Parametrize noise for learning aleatoric uncertainty

    if learn_noise:
        _sigma_noise = torch.log(torch.tensor(sigma_noise_init).exp() - 1.).clone().to(device).detach().requires_grad_(
            learn_noise)
        optimizer = torch.optim.Adam(list(gen.parameters()) + [_sigma_noise], lr=learning_rate)
    else:
        _sigma_noise = torch.log(torch.tensor(sigma_noise_init).exp() - 1.)
        optimizer = torch.optim.Adam(gen.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, min_lr=min_lr)

    Run = IVI(train_loader, ELBO, optimizer)

    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc='FuNN-HyVI', refresh=False)
        for _ in tr:

            scores = Run.one_epoch(gen, _sigma_noise)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'],
                           sigma=scores['sigma'])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach().cpu()

    return gen, model, sigma_noise, time


def MFVI(x_train, y_train, batch_size, layerwidth, nblayers, activation, n_epochs=n_epochs, sigma_noise_init=1.0,
         learn_noise=True):
    # setup data
    device = x_train.device
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # set predictive model
    input_dim = x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)

    # variational distribution
    MFVI = MeanFieldVariationalDistribution(param_count, std_init=0., sigma=0.001, device=device)

    def ELBO(x_data, y_data, MFVI_dist, _sigma_noise):
        y_pred = model(x_data, MFVI_dist(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood = average_normal_loglikelihood(y_pred, y_data, sigma_noise)
        theta = MFVI_dist(n_samples_KL)
        the_KL = MFVI_dist.log_prob(theta).mean() - log_diagonal_mvn_pdf(theta, std=sigma_prior).mean()
        the_ELBO = - Average_LogLikelihood + (len(x_data) / size_data) * the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise

    ## Parametrize noise for learning aleatoric uncertainty

    if learn_noise:
        _sigma_noise = torch.log(torch.tensor(sigma_noise_init).exp() - 1.).clone().to(device).detach().requires_grad_(
            learn_noise)
        optimizer = torch.optim.Adam(list(MFVI.parameters()) + [_sigma_noise], lr=learning_rate)
    else:
        _sigma_noise = torch.log(torch.tensor(sigma_noise_init).exp() - 1.)
        optimizer = torch.optim.Adam(MFVI.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, min_lr=min_lr)

    Run = IVI(train_loader, ELBO, optimizer)

    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc='MFVI', refresh=False)
        for _ in tr:

            scores = Run.one_epoch(MFVI, _sigma_noise)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'],
                           sigma=scores['sigma'])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach().cpu()

    return MFVI, model, sigma_noise, time


def FuNN_MFVI(x_train, y_train, batch_size, layerwidth, nblayers, activation, input_sampler, n_epochs=n_epochs,
              sigma_noise_init=1.0, learn_noise=True):
    # setup data
    device = x_train.device
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # set predictive model
    input_dim = x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)

    # variational distribution
    MFVI = MeanFieldVariationalDistribution(param_count, std_init=0., sigma=0.001, device=device)

    def prior(n):
        return sigma_prior * torch.randn(size=(n, param_count), device=device)

    def kl(var_dist):
        theta = var_dist(n_samples_KL)  # variationnel
        theta_prior = prior(n_samples_KL)  # prior
        X = input_sampler()  # sample OOD inputs
        theta_proj = model(X, theta).squeeze(2)  # evaluate predictors at OOD inputs
        theta_prior_proj = model(X, theta_prior).squeeze(2)  # evaluate predictors at OOD inputs
        K = kl_nne(theta_proj, theta_prior_proj, k=kNNE)  # compute NNe of KL on predictor approximations
        return K

    def ELBO(x_data, y_data, var_dist, _sigma_noise):
        y_pred = model(x_data, var_dist(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood = average_normal_loglikelihood(y_pred, y_data, sigma_noise)
        the_KL = kl(var_dist)
        the_ELBO = - Average_LogLikelihood + (len(x_data) / size_data) * the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise

    ## Parametrize noise for learning aleatoric uncertainty

    if learn_noise:
        _sigma_noise = torch.log(torch.tensor(sigma_noise_init).exp() - 1.).clone().to(device).detach().requires_grad_(
            learn_noise)
        optimizer = torch.optim.Adam(list(MFVI.parameters()) + [_sigma_noise], lr=learning_rate)
    else:
        _sigma_noise = torch.log(torch.tensor(sigma_noise_init).exp() - 1.)
        optimizer = torch.optim.Adam(MFVI.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, min_lr=min_lr)

    Run = IVI(train_loader, ELBO, optimizer)

    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc='FuNN-MFVI', refresh=False)
        for _ in tr:

            scores = Run.one_epoch(MFVI, _sigma_noise)
            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'],
                           sigma=scores['sigma'])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach().cpu()

    return MFVI, model, sigma_noise, time
