import torch

from Tools import log_norm

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

def LPP_Gaussian(y_pred_, y_test_, sigma_noise):
    y_sigma=torch.sqrt(y_pred_.var(0)+sigma_noise**2)
    y_mean=y_pred_.mean(0)
    LPP = log_norm( y_test_.unsqueeze(1), y_mean.unsqueeze(0), y_sigma.unsqueeze(0)).mean()
    MLPP = torch.mean(LPP).item()
    SLPP = torch.std(LPP).item()
    return (MLPP, SLPP)

def LPP(y_pred_, y_test_, sigma, device):
    r"""
    NLPD from Quinonero-Candela and al.
    NLL or LL from others
    nLPP for negative Log Posterior PREDICTIVE

    Args:
        y_pred: M x N x 1
        y_test: N x 1
        sigma: M x N x 1
        

    Returns:
        (Mean, Std) for Log Posterior Predictive of ensemble Theta on data (X,y)
    """
    y_pred=y_pred_
    y_test=y_test_
    log_proba = log_norm( y_test.unsqueeze(1), y_pred, sigma).view(y_pred.shape[0],y_pred.shape[1])    
    M = torch.tensor(y_pred.shape[0], device=log_proba.device).float()
    LPP = log_proba.logsumexp(dim=0) - torch.log(M)    
    MLPP = torch.mean(LPP).item()
    SLPP = torch.std(LPP).item()
    return (MLPP, SLPP)

def WAIC(y_pred, sigma_noise, y_test,  device):
    log_proba = log_norm( y_test.unsqueeze(1), y_pred, sigma_noise).view(y_pred.shape[0],y_pred.shape[1])    
    M = torch.tensor(y_pred.shape[0], device=log_proba.device).float()
    LPP = log_proba.logsumexp(dim=0) - torch.log(M)    
    LPP_s = LPP.mean().item()
    pWAIC = log_proba.var(dim=0).mean().item()
    return LPP_s-pWAIC

def PICP(y_pred, sigma_noise, y_test, device):
    r"""

    Args:
        y_pred: Tensor M x N x 1
        sigma_noise: Tensor M x N x 1
        y_test: Tensor N x 1
        
    
    Returns
        Prediction Interval Coverage Probability (PICP)  (Yao,Doshi-Velez, 2018):
        $$
        \frac{1}{N} \sum_{n<N} 1_{y_n \geq \hat{y}^\text{low}_n} 1_{y_n \leq \hat{y}^\text{high}_n}
        $$
        where $\hat{y}^\text{low}_n$ and $\hat{y}^\text{high}_n$ are respectively the $2,5 \%$ and $97,5 \%$ percentiles of the $\hat{y}_n=y_pred[:,n]+ sigma_noise[:,n]*N(0,1)$.
        &&

    """
    #add noise
    M = y_pred.shape[0]
    M_low = int(0.025 * M)
    M_high = int(0.975 * M)

    y_pred_s, _ = y_pred.sort(dim=0)

    y_low = y_pred_s[M_low, :].squeeze().to(device)
    y_high = y_pred_s[M_high, :].squeeze().to(device)

    y_test=y_test.squeeze()
    assert y_test.shape == y_low.shape, 'shape mismatch'
    inside = (y_test.squeeze() >= y_low).float() * (y_test.squeeze() <= y_high).float()
    return inside.mean().item()


def MPIW(y_pred, device, scale):
    r"""

    Args:
        scale: Float 
        device: Pytorch device
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

def evaluate_metrics(y_pred,sigma_noise, y_test, std_y_train, device='cpu', std=True):
    r"""

    Args:
        y_pred : Tensor B x M x 1
        sigma_noise: Tensor B x M x 1
        y_test: Tensor M x 1
        std_y_train: Float Tensor
        device: Pytorch Device
        std: Bool 
    Returns
        dict of metrics (RMSE,LPP,PICP, MPIW) 
        with std over y_test on RMSE and LPP if std is True
        
    """    
    y_pred = y_pred.to(device)
    y_test=y_test.to(device)
    std_y_train=std_y_train.to(device)
    metrics={}
    
    LPP_test = LPP(y_pred, y_test, sigma_noise, device)
    
    gLPP_test=LPP_Gaussian(y_pred, y_test, sigma_noise)
    
    y_pred_mean = y_pred.mean(dim=0)
    RMSE_test = RMSE(y_pred_mean, y_test, std_y_train, device)
    
    if std:
        metrics.update({'RMSE':RMSE_test})
        metrics.update({'LPP':LPP_test})
        metrics.update({'gLPP':gLPP_test})


    else:
        metrics.update({'RMSE':RMSE_test[0]})
        metrics.update({'LPP':LPP_test[0]}) 
        metrics.update({'gLPP':gLPP_test[0]})

    
    WAIC_test=WAIC(y_pred, sigma_noise, y_test,  device)
    metrics.update({'WAIC':WAIC_test})
    
    y_pred=y_pred+(sigma_noise*torch.randn_like(y_pred))
    PICP_test=PICP(y_pred, sigma_noise, y_test, device)
    metrics.update({'PICP':PICP_test})

    MPIW_test= MPIW(y_pred, device, std_y_train)
    metrics.update({'MPIW':MPIW_test})

    return metrics