import timeit

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange


class ensemble(nn.Module):
    def __init__(self, input_dim, layerwidth, activation, num_models):
        super().__init__()
        self.model_list = nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(input_dim, layerwidth),
            activation,
            torch.nn.Linear(layerwidth, 1)) for _ in range(num_models)])

    def forward(self, x):
        predictions = []
        for model in self.model_list:
            predictions.append(model(x).detach())
        return torch.stack(predictions, dim=0)

    @property
    def get_parameters(self):
        thetas = []
        for model in self.model_list:
            thetas.append(torch.cat([t.flatten() for t in model.parameters()]))
        return torch.stack(thetas)


def ensemble_train(model_list, train_dataset, batch_size, num_epochs=3000):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    start = timeit.default_timer()
    i = 0
    logs = {}
    for model in model_list:
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        log = []
        with trange(num_epochs) as tr:
            tr.set_description(desc='EnsembleB-{}'.format(i), refresh=False)
            for t in tr:
                cost = 0.
                count_batch = 0
                for x, y in train_loader:
                    optimizer.zero_grad()
                    fx = model(x)
                    output = loss(fx, y)
                    output.backward()
                    optimizer.step()

                    cost += output.item() * len(x)
                    count_batch += 1
                tr.set_postfix(loss=cost / count_batch)
                log.append(cost / count_batch)
        logs.update({i:log})
        i += 1

    stop = timeit.default_timer()
    time = stop - start
    return logs, time
