import numpy as np
from sklearn.linear_model import LinearRegression


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
