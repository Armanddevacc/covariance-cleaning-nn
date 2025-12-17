import numpy as np
from scipy import stats
from data.generators import make_monotone_pattern
from estimator.shaffer import (
    fit_monotone_regressions,
    reconstruct_mu_sigma_from_phi,
)


def generate_data(T, N):
    Sigma = stats.invwishart.rvs(df=2 * (N + 1), scale=np.eye(N)) * (
        2 * (N + 1) - N - 1
    )
    R = np.random.multivariate_normal(mean=np.zeros(N), cov=Sigma, size=T).T
    return Sigma, R


def generate_financial_covariance(
    T, N, n_factors=5, market_strength=0.7, sector_strength=0.25, noise_strength=0.1
):
    # Market factor (common to all assets) — how each asset react to the market
    market_loadings = np.random.uniform(0.5, 1.0, size=(N, 1))
    Sigma_market = market_strength * (market_loadings @ market_loadings.T)
    # market strength :control how dominant the market effect is

    # sector factors (clustered correlations)
    n_sectors = n_factors - 1  # one market + n_sectors sectors
    sector_size = N // n_sectors
    Sigma_sectors = np.zeros((N, N))
    for s in range(n_sectors):
        idx = slice(s * sector_size, (s + 1) * sector_size)
        A = np.random.randn(
            sector_size, sector_size
        )  # random factor matrix (like for market but localy here)
        Sigma_sectors[idx, idx] = (
            sector_strength * (A @ A.T) / sector_size
        )  # assets in the same sector have additional correlations beyond the market

    # Idiosyncratic noise (diagonal)
    diag_noise = noise_strength * np.random.uniform(0.5, 1.5, N)
    Sigma_diag = np.diag(
        diag_noise
    )  # unique risk per asset, unsure that the matrix is full-rank also

    # Combine all components
    Sigma = Sigma_market + Sigma_sectors + Sigma_diag

    # Normalize trace - > tend to a trace = N, nice for later when we compare cleaned vs noisy vs true cov matrix
    Sigma *= N / np.trace(Sigma)

    R = np.random.multivariate_normal(
        mean=np.zeros(N), cov=Sigma, size=T
    ).T  # log return are normally distributed

    return Sigma, R


def generate_batch(N, T, batch_size, function):
    Sigma_hat_batch = []
    sigma_batch = []
    for _ in range(batch_size):
        Sigma, R = function(T, N)
        R_hat, t_vec, _ = make_monotone_pattern(R)

        _, Sigma_hat = reconstruct_mu_sigma_from_phi(
            fit_monotone_regressions(R_hat, t_vec)
        )
        Sigma_hat_batch.append(Sigma_hat)
        sigma_batch.append(Sigma)

    return np.array(Sigma_hat_batch), np.array(sigma_batch)
