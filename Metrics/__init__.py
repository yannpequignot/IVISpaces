import numpy as np
import torch
from scipy import stats as st
from torch.nn import functional as F

from Tools import NormalLogLikelihood


def RMSE(y, y_pred, std_y_train, device):
    r"""
    Root Mean Squared Error and Root Std Squared Error
    Args:
       
        X: Tensor, N x dim
        y: Tensor, N x 1
        y_pred: Tensor, N x 1

    Returns:
        (Mean.sqrt(), Std.sqrt()) for Mean and Std on data (X,y) of the Squared Error
        of the predictor given y_pred

    """
    SE = (y_pred - y) ** 2
    RMSE = torch.mean(SE).sqrt() * std_y_train
    RStdSE = torch.std(SE).sqrt() * std_y_train
    return (RMSE.item(), RStdSE.item())


def LPP(y_pred_, y_test_, sigma, device):
    r"""
    NLPD from Quinonero-Candela and al.
    NLL or LL from others
    nLPP for negative Log Posterior PREDICTIVE

    Args:
        y_pred: M x N x 1
        y_test: N x 1
        sigma: float
        

    Returns:
        (Mean, Std) for Log Posterior Predictive of ensemble Theta on data (X,y)
    """
    y_pred=y_pred_.double()
    y_test=y_test_.double()
    NLL = NormalLogLikelihood(y_pred, y_test, sigma)
    M = torch.tensor(y_pred.shape[0], device=device).double()
    LPP = NLL.logsumexp(dim=0) - torch.log(M)
    MLPP = torch.mean(LPP).item()
    SLPP = torch.std(LPP).item()
    return (MLPP, SLPP)


def PICP(y_pred, y_test, device):
    r"""

    Args:
        y_pred: Tensor M x N x 1
        y_test: Tensor N x 1
    
    Returns
        Prediction Interval Coverage Probability (PICP)  (Yao,Doshi-Velez, 2018):
        $$
        \frac{1}{N} \sum_{n<N} 1_{y_n \geq \hat{y}^\text{low}_n} 1_{y_n \leq \hat{y}^\text{high}_n}
        $$
        where $\hat{y}^\text{low}_n$ and $\hat{y}^\text{high}_n$ are respectively the $2,5 \%$ and $97,5 \%$ percentiles of the $\hat{y}_n=y_pred[:,n]$.
        &&

    """
    M = y_pred.shape[0]
    M_low = int(0.025 * M)
    M_high = int(0.975 * M)

    y_pred_s, _ = y_pred.sort(dim=0)

    y_low = y_pred_s[M_low, :].squeeze().to(device)
    y_high = y_pred_s[M_high, :].squeeze().to(device)

    inside = (y_test >= y_low).float() * (y_test <= y_high).float()
    return inside.mean().item()


def MPIW(y_pred, device, scale):
    r"""

    Args:
        scale:
        device:
        y_pred: Tensor M x N x 1
    Returns
        float

        Mean Prediction Interval Width  (Yao,Doshi-Velez, 2018):
        $$
        \frac{1}{N} \sum_{n<N}\hat{y}^\text{high}_n - \hat{y}^\text{low}_n} 
        $$
        where $\hat{y}^\text{low}_n$ and $\hat{y}^\text{high}_n$ are respectively the $2,5 \%$ and $97,5 \%$ percentiles of the $\hat{y}_n=y_pred[:,n]$.

    """

    M = y_pred.shape[0]
    M_low = int(0.025 * M)
    M_high = int(0.975 * M)

    y_pred_s, _ = y_pred.sort(dim=0)

    y_low = y_pred_s[M_low, :].squeeze().to(device)
    y_high = y_pred_s[M_high, :].squeeze().to(device)

    width = scale * (y_high - y_low)
    return width.mean().item()

def evaluate_metrics(theta, model, X_test, y_test, sigma_noise, std_y_train, device='cpu', std=True):
    theta = theta.to(device)
    X_test=X_test.to(device)
    y_test=y_test.to(device)
    std_y_train=std_y_train.to(device)
    metrics={}
    
    y_pred=model(X_test, theta).detach()
    LPP_test = LPP(y_pred, y_test, sigma_noise, device)
    
    y_pred_mean = y_pred.mean(dim=0)
    RMSE_test = RMSE(y_pred_mean, y_test, std_y_train, device)
    
    if std:
        metrics.update({'RMSE':RMSE_test})
        metrics.update({'LPP':LPP_test})

    else:
        metrics.update({'RMSE':RMSE_test[0]})
        metrics.update({'LPP':LPP_test[0]})        
            
    PICP_test=PICP(y_pred, y_test, device)
    metrics.update({'PICP':PICP_test})

    MPIW_test= MPIW(y_pred, device, std_y_train)
    metrics.update({'MPIW':MPIW_test})

    return metrics


def KL(theta0, theta1, k=1, device='cpu', p=2):
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

    a0 = torch.topk(D0, k=k + 1, dim=1, largest=False, sorted=True)[0][:,
         k]  # .clamp(torch.finfo().eps,float('inf')).to(device)
    a1 = torch.topk(D1, k=k, dim=1, largest=False, sorted=True)[0][:,
         k - 1]  # .clamp(torch.finfo().eps,float('inf')).to(device)

    assert a0.shape == a1.shape

    d = torch.as_tensor(float(dim0), device=device)
    N0 = torch.as_tensor(float(n0), device=device)
    N1 = torch.as_tensor(float(n1), device=device)

    Mnn = (torch.log(a1) - torch.log(a0)).mean()
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


def FunSW(t, s, projection, device, n=100, n_samples_inputs=100, L=100):
    assert t.shape == s.shape
    W = torch.Tensor(n)
    for i in range(n):
        t_, s_ = projection(t, s, n_samples_inputs)
        # I added 1/sqrt(n_samples_input) the scaling factor we discussed :-)
        W[i] = 1 / torch.tensor(float(n_samples_inputs)).sqrt() * sw(s_.to(device), t_.to(device), device, L=L)
    return W.mean()  # , W.std()


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


def sphere(L, dim):
    """

    Args:
        L:
        dim:

    Returns:

    """
    theta = torch.randn(size=(L, dim))
    directions = F.normalize(theta, p=2, dim=1)
    return directions


def proj1d(S, u):
    """
    inputs:
        S: Tensor M x D
        u: Tensor K x D

    returns:
        dot: Tensor Kx M

    """
    assert S.shape[1] == u.shape[1]
    dim = S.shape[1]
    S_ = S.view(S.shape[0], dim, 1)
    u_ = u.view(u.shape[0], 1, 1, dim)
    dot = torch.matmul(u_, S_).squeeze()
    return dot


def sw(S0, S1, device, L=100):
    assert S0.shape[1] == S1.shape[1]
    dim = S0.shape[1]
    u = sphere(L, dim).to(device)
    S0_1d = proj1d(S0, u)
    S1_1d = proj1d(S1, u)
    W = [st.wasserstein_distance(S0_1d[i, :].cpu(), S1_1d[i, :].cpu()) for i in range(L)]
    return np.mean(W)  # , np.std(W)/L


#pytorch implementation/adaptation of scipy.stats wassertein_distance
def wassertein(u_values, v_values, p=1):
    
    n_u=u_values.shape[1]
    n_v=v_values.shape[1]
    n_all=n_u+n_v

    u_sorted, u_sorter = torch.sort(u_values, dim=1)
    v_sorted, v_sorter = torch.sort(v_values, dim=1)

    all_values = torch.cat([u_values, v_values], dim=1)
    all_values, _ = all_values.sort(dim=1)

    # Compute the differences between pairs of successive values of u and v.
    deltas = all_values[:,1:] - all_values[:,:-1]

    # Get the respective positions of the values of u and v among the values of
    # both distributions.

    u_values_ = u_sorted.unsqueeze(-1)
    all_values_ = all_values[:,:-1].unsqueeze(-2)
    u_cdf_indices = (all_values_ >= u_values_).float().sum(dim=1)

    v_values_ = v_sorted.unsqueeze(-1)
    all_values_ = all_values[:,:-1].unsqueeze(-2)
    v_cdf_indices = (all_values_ >= v_values_).float().sum(dim=1)

    # Calculate the CDFs of u and v
    u_cdf = u_cdf_indices / n_u

    v_cdf = v_cdf_indices / n_v

    # Compute the value of the integral based on the CDFs.
    # If p = 1, we avoid torch.power
    if p == 1:
        return torch.mul(torch.abs(u_cdf - v_cdf), deltas).sum(dim=1)
    else:
        return torch.mul(torch.pow(torch.abs(u_cdf - v_cdf), p),
                                       deltas).sum(dim=1).pow(1/p)
