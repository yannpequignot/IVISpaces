from Models import MC_Dropout_Model
import timeit
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import trange


def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma)
    return -(log_coeff + exponent).sum()

class MC_Dropout:
    def __init__(self, x_train, y_train, batch_size, no_units, init_sigma_noise, drop_prob, learn_noise=True,
                 activation=nn.ReLU()):

        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.network = MC_Dropout_Model(input_dim=x_train.shape[1], output_dim=y_train.shape[1], \
                                        no_units=no_units, init_sigma_noise=init_sigma_noise, drop_prob=drop_prob, \
                                        learn_noise=learn_noise, activation=activation)
        self.network.to(x_train.device)
        self.loss_func = log_gaussian_loss




    def fit(self, num_epochs, learn_rate, weight_decay):
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learn_rate, weight_decay=weight_decay)
        logs = {'loss': [], "lr": [], "sigma": []}
        start = timeit.default_timer()
        with trange(num_epochs) as tr:
            tr.set_description(desc='MCdropout', refresh=False)
            for _ in tr:
                epoch_loss = 0.
                train_samples = 0
                for x, y in self.train_loader:
                    self.optimizer.zero_grad()

                    output = self.network(x)
                    loss = self.loss_func(output, y, torch.log(torch.exp(self.network.sigma_noise) + 1.), 1)

                    loss.backward()

                    self.optimizer.step()

                    epoch_loss += loss.item()
                    train_samples += len(x)
                logs['lr'].append(optimizer.param_groups[0]['lr'])
                logs['loss'].append(epoch_loss / train_samples)
                logs['sigma'].append(self.network.sigma_noise.item())
                tr.set_postfix(loss=epoch_loss / train_samples, sigma_noise=self.network.sigma_noise.item())
        stop = timeit.default_timer()
        time = stop - start
        return logs,time

    def predict(self, x, num_samples):
        samples = []
        for i in range(num_samples):
            preds = self.network(x).detach()  # T x 1
            samples.append(preds)
        samples = torch.stack(samples)  # N x T x 1
        return samples, self.network.sigma_noise
