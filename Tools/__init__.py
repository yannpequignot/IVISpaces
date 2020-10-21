import torch
import math
import matplotlib.pyplot as plt


def log_norm(x, mu, std):
    """
    Evaluation of 1D normal distribution on tensors

    Parameters:
        x (Tensor): Data tensor of size S x N x 1 
        mu (Tensor): Mean tensor of size B x S x 1
        std (Float): Tensor of size B x S x 1(standard deviation)

    Returns:
        logproba (Tensor): size B x S x N x 1 with logproba[b,s,n]=[log p(x(s,n)|mu(b,s),std[b])]
    """

    assert x.shape[0] == mu.shape[1]
    assert x.shape[-1] == mu.shape[-1]

    #    assert mu.shape[1] == 1
    B = mu.shape[0]
    S = mu.shape[1]

    var = std.pow(2).unsqueeze(2)  # B x S x 1
    d = (x - mu.unsqueeze(2)) ** 2  # B x S x N x 1
    c = (2 * math.pi * var)  # B x S x 1
    return -0.5 * var.pow(-1) * d - 0.5 * c.log()


def NormalLogLikelihood(y_pred, y_data, sigma_noise):
    """
    Evaluation of a Normal distribution
    
    Parameters:
    y_pred (Tensor): tensor of size M X N X 1
    y_data (Tensor): tensor of size N X 1
    sigma_noise (Scalar): std for point likelihood: p(y_data | y_pred, sigma_noise) Gaussian N(y_pred,sigma_noise)

    Returns:
    logproba (Tensor):  (raw) size M X N , with logproba[m,n]= p(y_data[n] | y_pred[m,n], sigma_noise)                        (non raw) size M , logproba[m]=sum_n logproba[m,n]
    """
    # assert taken care of by log_norm
    #    assert y_pred.shape[1] == y_data.shape[0]
    #    assert y_pred.shape[2] == y_data.shape[1]
    #    assert y_data.shape[1] == 1
    std = sigma_noise * torch.ones_like(y_pred)
    log_proba = log_norm(y_data.unsqueeze(1), y_pred, std)
    return log_proba.view(y_pred.shape[0], y_pred.shape[1])


def average_normal_loglikelihood(y_pred, y_data, sigma_noise):
    """
    Evaluation of a Normal distribution
    
    Parameters:
    y_pred (Tensor): tensor of size M X N X 1
    y_data (Tensor): tensor of size N X 1
    sigma_noise (Scalar): std for point likelihood: p(y_data | y_pred, sigma_noise) Gaussian N(y_pred,sigma_noise)

    Returns:
    logproba (Tensor):  (raw) size M X N , with logproba[m,n]= p(y_data[n] | y_pred[m,n], sigma_noise)                        (non raw) size M , logproba[m]=sum_n logproba[m,n]
    """
    # assert taken care of by log_norm
    #    assert y_pred.shape[1] == y_data.shape[0]
    #    assert y_pred.shape[2] == y_data.shape[1]
    #    assert y_data.shape[1] == 1
    std = sigma_noise * torch.ones_like(y_pred)
    log_proba = log_norm(y_data.unsqueeze(1), y_pred, std).view(y_pred.shape[0], y_pred.shape[1])
    return log_proba.sum(dim=1).mean()


def log_diagonal_mvn_pdf(theta, std=1.):
    """
    Evaluation of log proba with density N(0,v*I_n)

    Parameters:
    x (Tensor): Data tensor of size NxD

    Returns:
    logproba (Tensor): size N, vector of log probabilities
    """
    dim = theta.shape[1]
    S = std * torch.ones(dim).to(theta.device)
    mu = torch.zeros(dim).to(theta.device)
    n_x = theta.shape[0]

    V_inv = S.view(1, 1, dim).pow(-2)
    d = ((theta - mu.view(1, dim)) ** 2).view(n_x, dim)
    const = 0.5 * S.log().sum() + 0.5 * dim * torch.tensor(2 * math.pi).log()
    return -0.5 * (V_inv * d).sum(2).squeeze() - const


def PlotCI(x_pred, y_pred, x, y, device):
    N = y_pred.shape[0] - 1
    print(N)
    m_3 = int(0.001 * N)
    M_3 = N - m_3
    m_2 = int(0.021 * N)
    M_2 = N - m_2
    m_1 = int(0.136 * N)
    M_1 = N - m_1

    x_pred = x_pred.squeeze()

    pred, _ = y_pred.sort(dim=0)
    y_mean = y_pred.mean(dim=0).squeeze().cpu()
    y_3 = pred[m_3, :].squeeze().cpu()
    Y_3 = pred[M_3, :].squeeze().cpu()
    y_2 = pred[m_2, :].squeeze().cpu()
    Y_2 = pred[M_2, :].squeeze().cpu()
    y_1 = pred[m_1, :].squeeze().cpu()
    Y_1 = pred[M_1, :].squeeze().cpu()

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.fill_between(x_pred.cpu(), y_3, Y_3, facecolor='springgreen', alpha=0.1)
    ax.fill_between(x_pred.cpu(), y_2, Y_2, facecolor='springgreen', alpha=0.1)
    ax.fill_between(x_pred.cpu(), y_1, Y_1, facecolor='springgreen', alpha=0.1)
    plt.plot(x_pred.cpu(), y_mean, color='springgreen')

    plt.grid(True, which='major', linewidth=0.5)

    plt.ylim(-4, 4)
    plt.scatter(x.cpu(), y.cpu(), marker='+', color='black', zorder=4)
    return fig
