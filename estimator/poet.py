"""
estimator/poet.py — POET: Principal Orthogonal complEment Thresholding.
Fan, Liao, Mincheva (2013), JRSS-B.

Estimates Σ = BB' + U where B captures K systematic factors and U is a
thresholded (approximately sparse) idiosyncratic component.

For correlation matrices the reconstruction enforces unit diagonal.
"""

import numpy as np


def _sample_corr(R):
    """Sample correlation matrix from (N, T) returns, no NaN."""
    N, T = R.shape
    Rc = R - R.mean(axis=1, keepdims=True)
    S = Rc @ Rc.T / (T - 1)
    d = np.sqrt(np.maximum(np.diag(S), 1e-12))
    C = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return 0.5 * (C + C.T)


def select_K_bn(C, T, max_K=20):
    """
    Select K via the Bai-Ng (2002) IC_p2 criterion.
    C: (N, N) sample correlation matrix.
    """
    N = C.shape[0]
    lam = np.linalg.eigvalsh(C)[::-1]  # descending
    total_var = lam.sum()
    best_k, best_ic = 1, np.inf
    for k in range(1, min(max_K, N) + 1):
        resid_var = (lam[k:].sum()) / (N * T)
        penalty = k * (N + T) / (N * T) * np.log(N * T / (N + T))
        ic = np.log(max(resid_var, 1e-30)) + penalty
        if ic < best_ic:
            best_ic = ic
            best_k = k
    return best_k


def poet(R, K=None, threshold=None, threshold_method="soft"):
    """
    POET estimator for the correlation matrix.

    Parameters
    ----------
    R : (N, T) returns, no NaN
    K : number of factors (None → Bai-Ng IC_p2 selection)
    threshold : off-diagonal threshold τ (None → universal √(log N / T))
    threshold_method : "hard" | "soft"

    Returns
    -------
    C_poet : (N, N) PSD correlation matrix estimate
    K_used : int, number of factors actually used
    """
    N, T = R.shape
    C = _sample_corr(R)

    lam, Q = np.linalg.eigh(C)
    lam = lam[::-1]      # descending
    Q   = Q[:, ::-1]

    if K is None:
        K = select_K_bn(C, T)
    K = max(1, min(K, N - 1))

    # Systematic component: Σ_{k=1}^K λ_k q_k q_k^T
    systematic = (Q[:, :K] * lam[:K]) @ Q[:, :K].T

    # Idiosyncratic residual
    C_res = C - systematic

    # Universal threshold (Fan et al., eq. 3.2)
    if threshold is None:
        threshold = np.sqrt(2 * np.log(N) / T)

    # Threshold off-diagonal entries of C_res
    diag_res = np.diag(C_res).copy()
    if threshold_method == "hard":
        C_res_t = np.where(np.abs(C_res) >= threshold, C_res, 0.0)
    else:  # soft
        sign = np.sign(C_res)
        C_res_t = sign * np.maximum(np.abs(C_res) - threshold, 0.0)

    # Restore diagonal of idiosyncratic part
    np.fill_diagonal(C_res_t, diag_res)

    # Reconstruct and enforce unit diagonal
    C_poet = systematic + C_res_t
    np.fill_diagonal(C_poet, 1.0)
    C_poet = 0.5 * (C_poet + C_poet.T)

    return C_poet, K


def poet_from_corr(C, T_eff, K=None, threshold=None, threshold_method="soft"):
    """
    Apply POET directly to a pre-computed correlation matrix C.
    Useful when clean returns are unavailable (e.g., after pairwise+PSD repair).

    Parameters
    ----------
    C       : (N, N) PSD correlation matrix
    T_eff   : effective sample size (used for automatic threshold)
    K       : number of factors (None → Bai-Ng)
    """
    N = C.shape[0]
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)

    lam, Q = np.linalg.eigh(C)
    lam = lam[::-1]; Q = Q[:, ::-1]

    if K is None:
        K = select_K_bn(C, T_eff)
    K = max(1, min(K, N - 1))

    systematic = (Q[:, :K] * lam[:K]) @ Q[:, :K].T
    C_res = C - systematic

    if threshold is None:
        threshold = np.sqrt(2 * np.log(N) / T_eff)

    diag_res = np.diag(C_res).copy()
    if threshold_method == "hard":
        C_res_t = np.where(np.abs(C_res) >= threshold, C_res, 0.0)
    else:
        sign = np.sign(C_res)
        C_res_t = sign * np.maximum(np.abs(C_res) - threshold, 0.0)

    np.fill_diagonal(C_res_t, diag_res)

    C_poet = systematic + C_res_t
    np.fill_diagonal(C_poet, 1.0)
    return 0.5 * (C_poet + C_poet.T)


