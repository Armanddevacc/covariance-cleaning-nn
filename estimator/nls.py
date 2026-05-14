"""
estimator/nls.py — Analytical Non-Linear Shrinkage (NLS).
Ledoit & Wolf (2020), Annals of Statistics.

Each empirical eigenvalue λ_i is shrunk to:

    δ_i = λ_i / |Φ_n(λ_i + i·h)|²

where Φ_n(z) = 1 - c - c·z·m_n(z), m_n is the Stieltjes transform of the
empirical spectral distribution, and h = N^{-1/3} is a bandwidth parameter.

Also provides Ledoit-Wolf (2004) linear shrinkage via sklearn wrapper.
"""

import numpy as np
from sklearn.covariance import LedoitWolf


# ─── Analytical NLS (LW 2020) ────────────────────────────────────────────────

def _stieltjes_transform(lam, z_real, z_imag):
    """
    m_n(z) = (1/N) Σ_j 1 / (λ_j - z)  at z = z_real + i·z_imag.
    Returns (Re(m_n), Im(m_n)).
    """
    d = lam - z_real                        # (N,)
    denom = d**2 + z_imag**2               # (N,)
    re = np.sum(d / denom) / len(lam)
    im = np.sum(-z_imag / denom) / len(lam)   # Im(1/(d - iz)) = z_imag/(d²+z²)
    return re, im


def nls(R, c=None, bandwidth_factor=1.0):
    """
    Analytical NLS estimator for the covariance matrix.
    Trace-preserving: scaled so Tr(Σ̂_NLS) = Tr(S).

    Parameters
    ----------
    R : (N, T) returns, no NaN
    c : concentration ratio N/T (inferred if None)
    bandwidth_factor : multiplier on the default bandwidth h = N^{-1/3}

    Returns
    -------
    Sigma_nls : (N, N) NLS covariance estimate
    """
    N, T = R.shape
    if c is None:
        c = N / (T - 1)

    Rc = R - R.mean(axis=1, keepdims=True)
    S  = Rc @ Rc.T / (T - 1)
    S  = 0.5 * (S + S.T)

    lam, Q = np.linalg.eigh(S)             # ascending eigenvalues
    lam    = np.maximum(lam, 0.0)

    # Bandwidth h (standard choice: N^{-1/3} × spectral range)
    h = bandwidth_factor * N**(-1/3) * (lam[-1] - lam[0] + 1e-8)

    delta = np.empty(N)
    for i in range(N):
        re_m, im_m = _stieltjes_transform(lam, lam[i], h)
        # Φ_n(λ_i + ih) = 1 - c - c·(λ_i + ih)·m_n
        re_phi = 1 - c - c * (lam[i] * re_m - h * im_m)
        im_phi =          -c * (lam[i] * im_m + h * re_m)
        phi_sq = max(re_phi**2 + im_phi**2, 1e-12)
        delta[i] = lam[i] / phi_sq

    # Trace preservation (ensures Tr(Σ̂_NLS) = Tr(S))
    delta *= lam.sum() / delta.sum()

    # Clip: real data has a large market eigenvalue whose Stieltjes denominator
    # goes near-zero, causing explosive delta. Cap at 10× largest empirical eigenvalue.
    delta = np.clip(delta, 0.0, 10.0 * lam[-1])

    return Q @ np.diag(delta) @ Q.T


def nls_corr(R, c=None, bandwidth_factor=1.0):
    """
    NLS applied to the sample correlation matrix, returning a correlation matrix.
    """
    N, T = R.shape
    Rc   = R - R.mean(axis=1, keepdims=True)
    S    = Rc @ Rc.T / (T - 1)
    std  = np.sqrt(np.maximum(np.diag(S), 1e-12))
    C    = S / np.outer(std, std)
    np.fill_diagonal(C, 1.0)
    C    = 0.5 * (C + C.T)

    if c is None:
        c = N / (T - 1)

    lam, Q = np.linalg.eigh(C)
    lam    = np.maximum(lam, 0.0)
    h      = bandwidth_factor * N**(-1/3) * (lam[-1] - lam[0] + 1e-8)

    delta = np.empty(N)
    for i in range(N):
        re_m, im_m = _stieltjes_transform(lam, lam[i], h)
        re_phi = 1 - c - c * (lam[i] * re_m - h * im_m)
        im_phi =          -c * (lam[i] * im_m + h * re_m)
        phi_sq = max(re_phi**2 + im_phi**2, 1e-12)
        delta[i] = lam[i] / phi_sq

    delta *= lam.sum() / delta.sum()
    delta  = np.clip(delta, 0.0, 10.0 * lam[-1])

    C_nls = Q @ np.diag(delta) @ Q.T
    np.fill_diagonal(C_nls, 1.0)
    return 0.5 * (C_nls + C_nls.T)


# ─── Linear shrinkage wrappers (sklearn) ─────────────────────────────────────

def lw_covariance(R):
    """Ledoit-Wolf linear shrinkage covariance. R: (N, T). Returns (N, N)."""
    est = LedoitWolf().fit(R.T)            # sklearn expects (T, N)
    return est.covariance_


def lw_corr(R):
    """Ledoit-Wolf linear shrinkage → correlation matrix. R: (N, T)."""
    S = lw_covariance(R)
    d = np.sqrt(np.maximum(np.diag(S), 1e-12))
    C = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return 0.5 * (C + C.T)


