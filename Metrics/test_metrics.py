import torch

from Tools import log_norm


def rmse(y, y_pred, std_y_train):
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
    return RMSE.item(), RStdSE.item()


def lpp_gaussian(y_pred_, y_test_, sigma_noise, y_scale):
    y_sigma = torch.sqrt(y_pred_.var(0) + sigma_noise ** 2)
    y_mean = y_pred_.mean(0)
    LPP = log_norm(y_test_.unsqueeze(1), y_mean.unsqueeze(0), y_sigma.unsqueeze(0)).mean()
    # account for data scaling
    LPP -= y_scale.log()
    MLPP = torch.mean(LPP).item()
    SLPP = torch.std(LPP).item()
    return MLPP, SLPP


def lpp(y_pred_, y_test_, sigma, y_scale):
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
    y_pred = y_pred_
    y_test = y_test_
    log_proba = log_norm(y_test.unsqueeze(1), y_pred, sigma).view(y_pred.shape[0], y_pred.shape[1])
    # account for data scaling
    log_proba -= y_scale.log()
    M = torch.tensor(y_pred.shape[0], device=log_proba.device).float()
    LPP = log_proba.logsumexp(dim=0) - torch.log(M)
    MLPP = torch.mean(LPP).item()
    SLPP = torch.std(LPP).item()
    return MLPP, SLPP

