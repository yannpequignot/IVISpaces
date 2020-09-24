import torch
import math

def KL(theta0, theta1, k=1, device='cpu', p=2, beta=1.):
    """
        Parameters:
            theta0 (Tensor): Samples, P X NbDimensions
            theta1 (Tensor): Samples, R X NbDimensions
            k (Int): positive ordinal number

        Returns:
            (Float) k-Nearest Neighbour Estimation of the KL from theta0 to theta1

        Kullback-Leibler Divergence Estimation of Continuous Distributions Fernando Pérez-Cruz
        """

    n0 = theta0.shape[0]
    n1 = theta1.shape[0]
    dim0 = theta0.shape[1]
    dim1 = theta1.shape[1]
    assert dim0 == dim1

    D0 = torch.cdist(theta0, theta0, p=p)
    D1 = torch.cdist(theta0, theta1, p=p)

    a0 = torch.topk(D0, k=k + 1, dim=1, largest=False, sorted=True)[0][:,k]  # .clamp(torch.finfo().eps,float('inf')).to(device)
    a1 = torch.topk(D1, k=k, dim=1, largest=False, sorted=True)[0][:,k - 1]  # .clamp(torch.finfo().eps,float('inf')).to(device)

    assert a0.shape == a1.shape, 'dimension do not match'

    d = torch.as_tensor(float(dim0), device=device)
    N0 = torch.as_tensor(float(n0), device=device)
    N1 = torch.as_tensor(float(n1), device=device)

    Mnn = torch.log(a1).mean() - beta*torch.log(a0).mean()
    return d * Mnn + N1.log() - (N0 - 1).log()



def batchKL(theta0, theta1, k=1, device='cpu', p=2):
    """

    Parameters:
        theta0 (Tensor): Samples, B x P X NbDimensions
        theta1 (Tensor): Samples, B x R X NbDimensions
        k (Int): positive ordinal number

    Returns:
        (Float) k-Nearest Neighbour Estimation of the KL from theta0 to theta1

    Kullback-Leibler Divergence Estimation of Continuous Distributions Fernando Pérez-Cruz

    """

    b0 = theta0.shape[0]
    b1 = theta1.shape[0]
    assert b0 == b1
    n0 = theta0.shape[1]
    n1 = theta1.shape[1]
    dim0 = theta0.shape[2]
    dim1 = theta1.shape[2]
    assert dim0 == dim1

    # TODO check for new batch version of cdist in Pytorch (issue with backward on cuda)

    D0 = torch.stack([torch.cdist(theta0[i], theta0[i], p=p) for i in range(theta0.shape[0])])
    D1 = torch.stack([torch.cdist(theta0[i], theta1[i], p=p) for i in range(theta0.shape[0])])

    # D0=torch.cdist(theta0,theta0, p=p)
    # D1=torch.cdist(theta0,theta1, p=p)

    a0 = torch.topk(D0, k=k + 1, dim=2, largest=False, sorted=True)[0][:, :,
         k]  # .clamp(torch.finfo().eps,float('inf')).to(device)
    a1 = torch.topk(D1, k=k, dim=2, largest=False, sorted=True)[0][:, :,
         k - 1]  # .clamp(torch.finfo().eps,float('inf')).to(device)

    assert a0.shape == a1.shape

    d = torch.as_tensor(float(dim0), device=device)
    N0 = torch.as_tensor(float(n0), device=device)
    N1 = torch.as_tensor(float(n1), device=device)

    Mnn = (torch.log(a1) - torch.log(a0)).mean(dim=1)
    KL = dim0 * Mnn + N1.log() - (N0 - 1).log()
    return KL.mean()




def FunKL(t, s, projection, device, k=1, n=100, m=100):
    assert t.shape == s.shape
    K = torch.Tensor(n)
    for i in range(n):
        t_, s_ = projection(t, s, m)
        K[i] = KL(t_, s_, k=k, device=device)
    return K.mean()  # , K.std()


def SFunKL(t, s, projection, device, k=1, n=100, m=50):
    K = FunKL(t, s, projection, device) + FunKL(s, t, projection, device)
    return K



def Entropy(theta,k=1,k_MC=1,device='cpu'):
    """
    Parameters:
        theta (Tensor): Samples, NbExemples X NbDimensions
        k (Int): ordinal number

    Returns:
        (Float) k-Nearest Neighbour Estimation of the entropy of theta

    """
    nb_samples=theta.shape[0]
    dim=theta.shape[1]
    kMC=torch.tensor(float(k_MC))
    D=torch.cdist(theta,theta)
    a = torch.topk(D, k=k+1, dim=0, largest=False, sorted=True)[0][k].clamp(torch.finfo().eps,float('inf')).to(device)
    d=torch.as_tensor(float(dim), device=device)
    K=torch.as_tensor(float(k), device=device)
    N=torch.as_tensor(float(nb_samples), device=device)
    pi=torch.as_tensor(math.pi, device=device)
    lcd = d/2.*pi.log() - torch.lgamma(1. + d/2.0)#-d/2*K_MC.log()
    return torch.log(N) - torch.digamma(K) + lcd + d/nb_samples*a.div(torch.sqrt(kMC)).log().sum(-1)
