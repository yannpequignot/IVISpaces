import torch
import timeit
from tqdm import trange



def ensemble(x_train, y_train, batch_size, layerwidth, activation, num_epochs=3000, num_models=5):
    device=x_train.device
    input_dim = x_train.shape[1]

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    start = timeit.default_timer()
    model_list = []
    for m_i in range(num_models):

        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, layerwidth),
            activation,
            torch.nn.Linear(layerwidth, 1))
        model.to(device)

        loss = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        with trange(num_epochs) as tr:
            tr.set_description(desc='EnsembleB-{}'.format(m_i), refresh=False)
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
        model_list.append(model)

    stop = timeit.default_timer()
    time = stop - start
    return model_list, time

def ensemble_predict(x,models):
    predictions = []
    for m_i in range(len(models)):
        predictions.append(models[m_i](x).detach())
    return torch.stack(predictions, dim=0)