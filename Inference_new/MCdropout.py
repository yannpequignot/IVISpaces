import timeit
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import trange
import torch.nn.functional as F



def log_gaussian_loss(output, target, sigma, no_dim):
    exponent = -0.5*(target - output)**2/sigma**2
    log_coeff = -no_dim*torch.log(sigma)
    return -(log_coeff + exponent).sum()

class MC_Dropout(nn.Module):
    def __init__(self, input_dim, output_dim, no_units, init_sigma_noise, drop_prob, learn_noise, activation):
        super().__init__()
        self.drop_prob=drop_prob
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer1 = nn.Linear(input_dim, no_units)
        self.layer2 = nn.Linear(no_units, output_dim)

        self.activation = activation
        self._sigma_noise = nn.Parameter(torch.log(torch.tensor(init_sigma_noise).exp() - 1.), requires_grad=learn_noise)


    @property
    def sigma_noise(self):
        return torch.log(torch.exp(self._sigma_noise) + 1.)

    def forward(self, x):

        x = x.view(-1, self.input_dim)
        x = self.layer1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.drop_prob, training=True)
        x = self.layer2(x)
        return x

    def predict(self, x, num_samples):
        samples = []
        for i in range(num_samples):
            preds = self.forward(x).detach()  # T x 1
            samples.append(preds)
        samples = torch.stack(samples)  # N x T x 1
        return samples

def MCdo_train(model, train_dataset, num_epochs, learn_rate, weight_decay):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    loss_func = log_gaussian_loss
    logs = {'loss': [], "lr": [], "sigma": []}
    start = timeit.default_timer()
    with trange(num_epochs) as tr:
        tr.set_description(desc='MCdropout', refresh=False)
        for _ in tr:
            epoch_loss = 0.
            train_samples = 0
            for x, y in train_loader:
                optimizer.zero_grad()

                output = model(x)
                loss = loss_func(output, y, torch.log(torch.exp(model.sigma_noise) + 1.), 1)

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()
                train_samples += len(x)
            logs['lr'].append(optimizer.param_groups[0]['lr'])
            logs['loss'].append(epoch_loss / train_samples)
            logs['sigma'].append(model.sigma_noise.item())
            tr.set_postfix(loss=epoch_loss / train_samples, sigma_noise=model.sigma_noise.item())
    stop = timeit.default_timer()
    time = stop - start
    return logs, time
