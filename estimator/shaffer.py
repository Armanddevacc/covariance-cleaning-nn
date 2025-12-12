import numpy as np
from sklearn.linear_model import LinearRegression
import torch
from scipy import stats


def fit_monotone_regressions(R, t_vec):
    n, T = R.shape
    phi_list = []
    for k in range(n):
        t_k = t_vec[k]
        y = R[k, t_k:]
        if k == 0:
            beta0 = np.nanmean(R[0, :])
            beta = np.array([])
            resid = y - beta0
            sigma2 = float(resid.T @ resid) / len(resid)
            phi_list.append((beta0, beta, sigma2))
        else:
            X = R[:k, t_k:].T
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X, y)

            beta0 = reg.intercept_
            beta = reg.coef_
            resid = y - reg.predict(X)

            sigma2 = (resid.T @ resid) / len(resid)
            phi_list.append((beta0, beta, sigma2))
    return phi_list


def reconstruct_mu_sigma_from_phi(phi_list):

    p = len(phi_list)
    mu = np.zeros(p, dtype=float)
    Sigma = np.zeros((p, p), dtype=float)

    beta0_1, beta_1, sigma2_1 = phi_list[0]
    mu[0] = beta0_1
    Sigma[0, 0] = sigma2_1

    for k in range(1, p):
        beta0_k, beta_k, sigma2_k = phi_list[k]

        mu[k] = beta0_k + beta_k @ mu[:k]

        S11 = Sigma[:k, :k]

        cross = beta_k @ S11
        Sigma[k, :k] = cross
        Sigma[:k, k] = cross.T

        Sigma[k, k] = sigma2_k + cross @ beta_k.T

    return mu, Sigma


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
