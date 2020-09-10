import torch
import math


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
    
    var = std.pow(2).unsqueeze(2)
    d = (x-mu.unsqueeze(2))**2 # B x S x N x 1
    c = (2*math.pi*var)
    return -0.5 * var.pow(-1)*d - 0.5 * c.log()


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
    std=sigma_noise*torch.ones_like(y_pred)
    log_proba = log_norm( y_data.unsqueeze(1), y_pred, std)
    return log_proba.view(y_pred.shape[0],y_pred.shape[1])

def logmvn01pdf(theta, device,v=1.):
    """
    Evaluation of log proba with density N(0,v*I_n)

    Parameters:
    x (Tensor): Data tensor of size NxD

    Returns:
    logproba (Tensor): size N, vector of log probabilities
    """
    dim = theta.shape[1]
    S = v*torch.ones(dim).type_as(theta).to(device)
    mu = torch.zeros(dim).type_as(theta).to(device)
    n_x = theta.shape[0]

    H = S.view(dim, 1, 1).inverse().view(1, 1, dim)
    d = ((theta-mu.view(1, dim))**2).view(n_x, dim)
    const = 0.5*S.log().sum()+0.5*dim*torch.tensor(2*math.pi).log()
    return -0.5*(H*d).sum(2).squeeze()-const




