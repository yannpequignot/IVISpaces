import numpy as np
import math
import torch
from torch import nn


from datetime import datetime

from torch.utils.data import Dataset

from Models import get_mlp, BigGenerator, MeanFieldVariationalDistribution
from Tools import AverageNormalLogLikelihood, logmvn01pdf
from Metrics import KL, evaluate_metrics, Entropy

from Experiments import get_setup

from Inference.IVI_noise import IVI

from tqdm import trange

import timeit

## Hyperparameters ##

#predictive model
layerwidth=50
nblayers=1
activation=nn.ReLU()

#generative model
lat_dim=5

#optimizer
learning_rate=0.005

#scheduler
patience=40
lr_decay=.5#.7
min_lr= 0.0001
n_epochs=1#5000#2000


#MC_Dropout
drop_prob=0.05
weight_decay= 1e-1/(len(x_train)+len(x_test))**0.5
num_samples=50#nb_predictors
log_every=50

#loss hyperparameters
n_samples_LL=100 #nb of predictor samples for average LogLikelihood

n_samples_KL=500 #nb of predictor samples for KL divergence
kNNE=1 #k-nearest neighbour

n_samples_FU=200 #nb of ood inputs for predictive KL NN estimation


sigma_prior=.5# TO DO check with other experiments setup.sigma_prior    


def Mc_dropout(dataset,device):
        
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device, seed=42) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    batch_size=int(np.min([size_data // 6, 500])) #50 works fine too!
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    #MC_Dropout model
    
    def to_variable(var=(), cuda=True, volatile=False):
        out = []
        for v in var:
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).type(torch.FloatTensor)
                
            if not v.is_cuda and cuda:
                v = v.to(device)
                
            if not isinstance(v, Variable):
                v = Variable(v, volatile=volatile)

            out.append(v)
        return out 
    
    def log_gaussian_loss(output, target, sigma, no_dim):
        exponent = -0.5*(target - output)**2/sigma**2
        log_coeff = -no_dim*torch.log(sigma)
    
        return -(log_coeff + exponent).sum()
    
    class gaussian:
        def __init__(self, mu, sigma):
            self.mu = mu
            self.sigma = sigma
        
        def loglik(self, weights):
            exponent = -0.5*(weights - self.mu)**2/self.sigma**2
            log_coeff = -0.5*(np.log(2*np.pi) + 2*np.log(self.sigma))
        
            return (exponent + log_coeff).sum()
    
    class MC_Dropout_Layer(nn.Module):
        def __init__(self, input_dim, output_dim, dropout_prob):
            super(MC_Dropout_Layer, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.dropout_prob = dropout_prob
        
            self.weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-0.01, 0.01))
            self.biases = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-0.01, 0.01))
        
        def forward(self, x):
        
            dropout_mask = torch.bernoulli((1 - self.dropout_prob)*torch.ones(self.weights.shape)).to(device)
        
            return torch.mm(x, self.weights*dropout_mask) + self.biases
        
        
        
    class MC_Dropout_Model(nn.Module):
        def __init__(self, input_dim, output_dim, no_units, init_log_noise, drop_prob):
            super(MC_Dropout_Model, self).__init__()
        
            self.drop_prob=drop_prob
            self.input_dim = input_dim
            self.output_dim = output_dim
        
            self.layer1 = nn.Linear(input_dim, no_units)
            self.layer2 = nn.Linear(no_units, output_dim)
        
            # activation to be used between hidden layers
            self.activation = nn.ReLU(inplace = True)
            self.log_noise = nn.Parameter(torch.FloatTensor([init_log_noise]).to(device))

    
        def forward(self, x):
        
            x = x.view(-1, self.input_dim)
        
            x = self.layer1(x)
            x = self.activation(x)
        
            x = F.dropout(x, p=drop_prob, training=True)
        
            x = self.layer2(x)
        
            return x
    
    
    class MC_Dropout_Wrapper:
        def __init__(self, input_dim, output_dim, no_units, learn_rate, batch_size, no_batches, weight_decay, init_log_noise):
        
            self.learn_rate = learn_rate
            self.batch_size = batch_size
            self.no_batches = no_batches
        
            self.network = MC_Dropout_Model(input_dim = input_dim, output_dim = output_dim,
                                        no_units = no_units, init_log_noise = init_log_noise, drop_prob = drop_prob)
            self.network.to(device)
        
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learn_rate, weight_decay=weight_decay)
            self.loss_func = log_gaussian_loss
    
        def fit(self, x, y):
            x, y = to_variable(var=(x, y), cuda=True)
        
            # reset gradient and total loss
            self.optimizer.zero_grad()
        
            output = self.network(x)
            loss = self.loss_func(output, y, torch.exp(self.network.log_noise), 1)/len(x)
        
            loss.backward()
            self.optimizer.step()

            return loss
    
        def get_loss_and_rmse(self, x, y, num_samples):
            x, y = to_variable(var=(x, y), cuda=True)
        
            means, stds = [], []
            for i in range(num_samples):
                output = self.network(x)
                means.append(output)
        
            means = torch.cat(means, dim=1)
            mean = means.mean(dim=-1)[:, None]
            std = ((means.var(dim=-1) + torch.exp(self.network.log_noise)**2)**0.5)[:, None]
            loss = self.loss_func(mean, y, std, 1)
        
            rmse = ((mean - y)**2).mean()**0.5

            return loss.detach().cpu(), rmse.detach().cpu()
        
    def train_mc_dropout(x_train,y_train,x_test,y_test, y_stds, drop_prob, num_epochs, num_units, learn_rate, weight_decay, log_every, num_samples):
        in_dim = x_train.shape[1]
        out_dim = y_train.shape[1]
        train_logliks, test_logliks = [], []
        train_rmses, test_rmses = [], []
    
        net = MC_Dropout_Wrapper(input_dim=in_dim, output_dim=out_dim, no_units=num_units,learn_rate=learn_rate, batch_size=batch_size, no_batches=1, init_log_noise=0, weight_decay=weight_decay)


        losses = []
        fit_loss_train = np.zeros(num_epochs)

        for i in range(num_epochs):
            loss = net.fit(x_train, y_train)
            losses.append(loss)
                
            if i % log_every == 0 or i == num_epochs - 1:
                test_loss, rmse = net.get_loss_and_rmse(x_test, y_test, num_samples=num_samples)
                test_loss, rmse = test_loss.cpu().data.numpy(), rmse.cpu().data.numpy()

                print('Epoch: %4d, Train loss: %6.3f Test loss: %6.3f RMSE: %.3f' %
                        (i, loss.cpu().data.numpy()/len(x_train), test_loss/len(x_test), rmse*y_stds[0].cpu().data.numpy()))


        train_loss, train_rmse = net.get_loss_and_rmse(x_train, y_train, num_samples=num_samples)
        test_loss, test_rmse = net.get_loss_and_rmse(x_test, y_test, num_samples=num_samples)
        
        train_logliks.append((train_loss.cpu().data.numpy()/len(x_train) + np.log(y_stds)[0]))
        test_logliks.append((test_loss.cpu().data.numpy()/len(x_test) + np.log(y_stds)[0]))

        train_rmses.append(y_stds[0]*train_rmse.cpu().data.numpy())
        test_rmses.append(y_stds[0]*test_rmse.cpu().data.numpy())
        


        print('Train log. lik. = %6.3f +/- %6.3f' % (-np.array(train_logliks).mean(), np.array(train_logliks).var()**0.5))
        print('Test  log. lik. = %6.3f +/- %6.3f' % (-np.array(test_logliks).mean(), np.array(test_logliks).var()**0.5))
        print('Train RMSE      = %6.3f +/- %6.3f' % (np.array(train_rmses).mean(), np.array(train_rmses).var()**0.5))
        print('Test  RMSE      = %6.3f +/- %6.3f' % (np.array(test_rmses).mean(), np.array(test_rmses).var()**0.5))
    
        return net
    
    
    start = timeit.default_timer()
    net  = train_mc_dropout(x_train=x_train,y_train=y_train, x_test=x_test,y_test=y_test, y_stds=std_y_train,drop_prob=drop_prob, num_epochs=num_epochs,  num_units=layerwidth, learn_rate=learn_rate, weight_decay=weight_decay, num_samples=num_samples, log_every=log_every)
    stop = timeit.default_timer()
    time = stop - start
    
    
    nb_predictors=num_samples
    for i in range(nb_predictors):
        preds = net.network.forward(x_test).cpu().data.numpy()
        samples.append(preds)
        
        
    samples = np.array(samples)
    means = torch.Tensor(samples.mean(axis = 0)).view(1,-1,1)
    aleatoric = torch.exp(net.network.log_noise).detach()
    epistemic = torch.Tensor(samples.var(axis = 0)**0.5).view(-1,1)
    sigma_noise = aleatoric.view(1,-1,1)
    y_pred=means + epistemic * torch.randn(nb_predictors,1,1) 
    metrics=get_metrics(y_pred, sigma_noise, y_test, std_y_train, 'Mc_Drop', time)
    return metrics
    


def MFVI_noise(dataset,device):
    
    setup_ = get_setup(dataset)
    setup=setup_.Setup(device, seed=42) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    batch_size=int(np.min([size_data // 6, 500])) #50 works fine too!
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation) 
    
    MFVI=MeanFieldVariationalDistribution(param_count, std_init=0.,sigma=0.2, device=device)    

    _sigma_noise=torch.log(torch.tensor(1.).exp()-1.).clone().to(device).detach().requires_grad_(True)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    def ELBO(x_data, y_data, MFVI, _sigma_noise):
        alpha=(len(x_data)/size_data) #TODO check with alpah=1.

        y_pred=model(x_data,MFVI(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood=AverageNormalLogLikelihood(y_pred, y_data, sigma_noise)
        theta=MFVI(n_samples_KL)
        the_KL=MFVI.log_prob(theta).mean()-logmvn01pdf(theta,sigma_prior).mean()
        the_ELBO= - Average_LogLikelihood+ alpha* the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise
    
    optimizer = torch.optim.Adam(list(MFVI.parameters())+[_sigma_noise], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, verbose=True, min_lr=min_lr)
    Run=IVI(train_loader, ELBO, optimizer)

    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc=dataset+'/MFVI', refresh=False)
        for t in tr:
            scores=Run.run(MFVI,_sigma_noise)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'], sigma=scores['sigma'])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start

    theta=MFVI(1000).detach()
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach().cpu()
    y_pred=model(x_test,theta)
    metrics=get_metrics(y_pred, sigma_noise, y_test, std_y_train, 'MFVI', time)
    return metrics


def FuNNeVI_noise(dataset,device):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device, seed=42) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    batch_size=int(np.min([size_data // 6, 500])) #50 works fine too!
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
    def prior(n):
        return sigma_prior*torch.randn(size=(n,param_count), device=device)
    
    def projection(theta0,theta1, x_data):
        #batch sample OOD   
        n_ood=n_samples_FU
        epsilon=0.1
        M = x_train.max(0, keepdim=True)[0]+epsilon
        m = x_train.min(0, keepdim=True)[0]-epsilon
        X_ood = torch.rand(n_ood,input_dim).to(device) * (M-m) + m    
        #X_ood = x_data+torch.randn_like(x_data)
        
        #compute projection on both paramters with model
        theta0_proj=model(X_ood, theta0).squeeze(2)
        theta1_proj=model(X_ood, theta1).squeeze(2)
        return theta0_proj, theta1_proj

    def kl(x_data, GeN):

        theta=GeN(n_samples_KL) #variationnel
        theta_prior=prior(n_samples_KL) #prior

        theta_proj, theta_prior_proj = projection(theta, theta_prior,x_data)

        K=KL(theta_proj, theta_prior_proj,k=kNNE,device=device)
        return K
    
    def ELBO(x_data, y_data, GeN, _sigma_noise):
        alpha=(len(x_data)/size_data) #TODO check with alpha=1.
        y_pred=model(x_data,GeN(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood=AverageNormalLogLikelihood(y_pred, y_data, sigma_noise)
        the_KL=kl(x_data, GeN)
        the_ELBO= - Average_LogLikelihood+ alpha* the_KL#(len(x_data)/size_data)*the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise

    #generative model
    GeN = BigGenerator(lat_dim,param_count,device).to(device)

    ## Parametrize noise for learning aleatoric uncertainty
    
    _sigma_noise=torch.log(torch.tensor(1.0).exp()-1.).clone().to(device).detach().requires_grad_(True)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    optimizer = torch.optim.Adam(list(GeN.parameters())+[_sigma_noise], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, verbose=True, min_lr=min_lr)

    Run=IVI(train_loader, ELBO, optimizer)
    
    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc=dataset+'/FuNNeVI', refresh=False)
        for t in tr:
            
            
            scores=Run.run(GeN,_sigma_noise)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'], sigma=scores['sigma'])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start
    
    theta=GeN(1000).detach()
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach().cpu()
    y_pred=model(x_test,theta)
    metrics=get_metrics(y_pred, sigma_noise, y_test, std_y_train, 'FuNNeVI', time)
    return metrics


def GeNNeVI_noise(setup,device):

    setup_ = get_setup(dataset)
    setup=setup_.Setup(device, seed=42) 

    x_train, y_train=setup.train_data()
    x_test, y_test=setup.test_data()

    std_y_train = torch.tensor(1.)
    if hasattr(setup, '_scaler_y'):
        std_y_train=torch.tensor(setup._scaler_y.scale_, device=device).squeeze().float()
    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    size_data=len(train_dataset)
    batch_size=int(np.min([size_data // 6, 500])) #50 works fine too!
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  

    ## predictive model
    input_dim=x_train.shape[1]
    param_count, model = get_mlp(input_dim, layerwidth, nblayers, activation)
    
    def prior(n):
        return sigma_prior*torch.randn(size=(n,param_count), device=device)
    
    def kl(x_data, GeN):

        theta=GeN(n_samples_KL) #variationnel
        theta_prior=prior(n_samples_KL) #prior

        K=KL(theta, theta_prior,k=kNNE,device=device)
        return K
    
    def ELBO(x_data, y_data, GeN, _sigma_noise):
        alpha=(len(x_data)/size_data) #TODO check with alpah=1.
        y_pred=model(x_data,GeN(n_samples_LL))
        sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

        Average_LogLikelihood=AverageNormalLogLikelihood(y_pred, y_data, sigma_noise)
        the_KL=kl(x_data, GeN)
        the_ELBO= - Average_LogLikelihood+ alpha* the_KL#(len(x_data)/size_data)*the_KL
        return the_ELBO, the_KL, Average_LogLikelihood, sigma_noise

    #generative model
    GeN = BigGenerator(lat_dim,param_count,device).to(device)

    ## Parametrize noise for learning aleatoric uncertainty
    
    _sigma_noise=torch.log(torch.tensor(1.0).exp()-1.).clone().to(device).detach().requires_grad_(True)
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.)

    optimizer = torch.optim.Adam(list(GeN.parameters())+[_sigma_noise], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay, verbose=True, min_lr=min_lr)

    Run=IVI(train_loader, ELBO, optimizer)
    
    start = timeit.default_timer()
    with trange(n_epochs) as tr:
        tr.set_description(desc=dataset+'/GeNNeVI', refresh=False)
        for t in tr:

            
            scores=Run.run(GeN,_sigma_noise)

            scheduler.step(scores['ELBO'])
            tr.set_postfix(ELBO=scores['ELBO'], LogLike=scores['LL'], KL=scores['KL'], lr=scores['lr'], sigma=scores['sigma'])

            if scores['lr'] <= 1e-4:
                break
    stop = timeit.default_timer()
    time = stop - start
    
    theta=GeN(1000).detach()
    sigma_noise = torch.log(torch.exp(_sigma_noise) + 1.).detach().cpu()
    y_pred=model(x_test,theta)
    metrics=get_metrics(y_pred, sigma_noise, y_test, std_y_train, 'GeNNeVI', time)
    return metrics

def get_metrics(y_pred, sigma_noise, y_test, std_y_train, method, time):
    metrics=evaluate_metrics(y_pred, sigma_noise.view(1,1,1), y_test,  std_y_train, device='cpu', std=False)
    metrics.update({'time [s]': time})
    metrics.update({'std noise': sigma_noise.item()})
    metrics_list=list(metrics.keys())
    for j in metrics_list:
        metrics[(method,j)] = metrics.pop(j)
    return metrics


if __name__ == "__main__":
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    date_string = datetime.now().strftime("%Y-%m-%d-%H:%M")
    datasets=['boston','concrete', 'energy', 'powerplant',  'wine', 'yacht']
    RESULTS={dataset:{} for dataset in datasets}#
    for dataset in datasets:
        print(dataset)     
 
        metrics={}
        metrics.update(MFVI_noise(dataset,device))
        metrics.update(GeNNeVI_noise(dataset,device))
        metrics.update(FuNNeVI_noise(dataset,device))
            
        RESULTS[dataset].update(metrics)
        torch.save(RESULTS,'Results/NEW/UCI'+date_string+'.pt')
        #RESULTS.append(GeNNeVI_noise(dataset,device))
        #torch.save(RESULTS,'Results/NEW/GeNoise'+date_string+'.pt')