import torch
from scipy import stats
import numpy as np


# fully vectorized covariance with pairwise-complete observations like pandas.cov()
def torch_cov_pairwise(X):
    """
    input X: (N, T) or (B, N, T)
    """
    if X.dim() == 2:
        X = X.unsqueeze(0)  # becomes (1, N, T)

    # mask: 1 where valid, 0 where NaN
    mask = ~torch.isnan(X)  # (B, N, T)

    # mean over time dimension (T)
    means = torch.nanmean(X, dim=-1, keepdim=True)  # (B, N, 1)

    # centered data (NaNs propagate)
    Xc = X - means  # (B, N, T)

    # centered data but NaN replaced by 0
    Xc_zero = torch.where(mask, Xc, torch.tensor(0.0, device=X.device, dtype=X.dtype))

    # pairwise valid counts n_ij = sum_t mask[i,t] * mask[j,t]
    valid_counts = mask.float() @ mask.float().transpose(1, 2)  # (B, N, N)

    # numerator: sum of centered products
    numerator = Xc_zero @ Xc_zero.transpose(1, 2)  # (B, N, N)

    # denominator: n_ij - 1
    denom = valid_counts - 1
    denom = torch.where(denom > 0, denom, torch.tensor(float("nan"), device=X.device))

    cov = numerator / denom

    # drop batch if original input had no batch
    if cov.size(0) == 1:
        cov = cov[0]

    return cov


# test of the naiv estimator vectorized :
# we changed to this estimator bc prof B suggested it as its error (which one) is smaller than shaffers
def test_torch_cov_pairwise():
    N, T = 100, 100
    Sigma = stats.invwishart.rvs(df=2 * (N + 1), scale=np.eye(N)) * (
        2 * (N + 1) - N - 1
    )
    R = np.random.multivariate_normal(mean=np.zeros(N), cov=Sigma, size=T).T
    return ((torch_cov_pairwise(torch.tensor(R)).numpy() - np.cov(R)) ** 2).sum()
