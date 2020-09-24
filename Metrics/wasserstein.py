import torch
from scipy import stats as st
import torch.nn.functional as F


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

def FunSW(t, s, projection, device, n=100, n_samples_inputs=100, L=100):
    assert t.shape == s.shape
    W = torch.Tensor(n)
    for i in range(n):
        t_, s_ = projection(t, s, n_samples_inputs)
        # I added 1/sqrt(n_samples_input) the scaling factor we discussed :-)
        W[i] = 1 / torch.tensor(float(n_samples_inputs)).sqrt() * sw(s_.to(device), t_.to(device), device, L=L)
    return W.mean()  # , W.std()



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