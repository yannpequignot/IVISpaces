import torch
import math

def KL(theta0, theta1, k=1, p=2):
    """
        Parameters:
            theta0 (Tensor): Samples, P X NbDimensions
            theta1 (Tensor): Samples, R X NbDimensions
            k (Int): positive ordinal number

        Returns:
            (Float) k-Nearest Neighbour Estimation of the KL from theta0 to theta1

        Kullback-Leibler Divergence Estimation of Continuous Distributions Fernando Pérez-Cruz
        """

    device = theta0.device
    n0 = theta0.shape[0]
    n1 = theta1.shape[0]
    dim0 = theta0.shape[1]
    dim1 = theta1.shape[1]
    assert dim0 == dim1

    D0 = torch.cdist(theta0, theta0, p=p)
    D1 = torch.cdist(theta0, theta1, p=p)

    a0 = torch.topk(D0, k=k + 1, dim=1, largest=False, sorted=True)[0][:,k].clamp(torch.finfo().eps,float('inf')).to(device)
    a1 = torch.topk(D1, k=k, dim=1, largest=False, sorted=True)[0][:,k - 1].clamp(torch.finfo().eps,float('inf')).to(device)

    assert a0.shape == a1.shape, 'dimension do not match'

    d = torch.as_tensor(float(dim0), device=device)
    N0 = torch.as_tensor(float(n0), device=device)
    N1 = torch.as_tensor(float(n1), device=device)

    Mnn = torch.log(a1).mean() - torch.log(a0).mean()
    return d * Mnn + N1.log() - (N0 - 1).log()


def entropy_nne(theta, k=1, k_MC=1):
    """
    Parameters:
        theta (Tensor): Samples, NbExemples X NbDimensions
        k (Int): ordinal number
        k_MC (Int): for scaling of distances by 1/sqrt(k_MC) in functionnal estimation of entropy based on MC estimation of L_2 distances

    Returns:
        (Float) k-Nearest Neighbour Estimation of the entropy of theta

    """
    device=theta.device
    nb_samples=theta.shape[0]
    dim=theta.shape[1]
    kMC=torch.tensor(float(k_MC))
    D=torch.cdist(theta,theta)
    a = torch.topk(D, k=k+1, dim=0, largest=False, sorted=True)[0][k].clamp(torch.finfo().eps,float('inf')).to(device)
    d=torch.as_tensor(float(dim), device=device)
    K=torch.as_tensor(float(k), device=device)
    N=torch.as_tensor(float(nb_samples), device=device)
    pi=torch.as_tensor(math.pi, device=device)
    lcd = d/2.*pi.log() - torch.lgamma(1. + d/2.0)
    return torch.log(N) - torch.digamma(K) + lcd + d/nb_samples*a.div(torch.sqrt(kMC)).log().sum(-1)

def batch_entropy_nne(theta, k=1, k_MC=1):
    """
    Parameters:
        theta (Tensor): Samples, Batch x NbExemples X NbDimensions
        k (Int): ordinal number

    Returns:
        Tensor: H of size Batch x 1, k-Nearest Neighbour Estimation of the entropy of theta, H[b]=H(theta[b]).

    """
    device = theta.device
    nb_samples = theta.shape[1]
    d = torch.tensor(theta.shape[-1]).float()
    D = torch.cdist(theta,theta)
    a = torch.topk(D, k=k+1, dim=1, largest=False, sorted=True)[0][:,k]
    a= a.clamp(torch.finfo().eps,float('inf')).to(device).log().sum(1)
    K = torch.as_tensor(float(k), device=device)
    K_MC = torch.as_tensor(float(k_MC), device=device)
    N = torch.as_tensor(float(nb_samples), device=device)
    pi = torch.as_tensor(math.pi, device=device)
    lcd = d/2.*pi.log() - torch.lgamma(1. + d/2.0)-d/2*K_MC.log()
    H = torch.log(N) - torch.digamma(K) + lcd + d/nb_samples*a
    return H
