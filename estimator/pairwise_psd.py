"""
estimator/pairwise_psd.py — Pairwise MLE with PSD repair.

Combines:
  1. pairwise MLE correlation estimate (may be non-PSD)
  2. nearest correlation matrix projection (Higham 2002)
  3. Optional secondary cleaning: QIS or NLS applied to the PSD repair result

Used as a baseline in Step 2 (missingness benchmark).
"""

import numpy as np
from estimator.nearest_correlation.nearest_correlation import nearcorr


# ─── Helper: numpy pairwise covariance (no NaN) ──────────────────────────────

def pairwise_cov_np(R_nan):
    """
    Pairwise covariance from (N, T) array with NaN at missing positions.
    Returns (N, N) symmetric matrix (may be non-PSD).
    """
    N, T = R_nan.shape
    mask = ~np.isnan(R_nan)                         # (N, T) bool

    # Per-stock mean over observed values
    cnt  = mask.sum(axis=1, keepdims=True).clip(min=1)
    mu   = np.where(mask, R_nan, 0.0).sum(axis=1, keepdims=True) / cnt
    Rc   = np.where(mask, R_nan - mu, 0.0)          # centered, 0 at missing

    # Pairwise valid counts
    cnt_ij = (mask.astype(float)) @ (mask.astype(float)).T  # (N, N)
    denom  = np.where(cnt_ij > 1, cnt_ij - 1, np.nan)

    cov = (Rc @ Rc.T) / denom
    return cov


def pairwise_corr_np(R_nan):
    """
    Pairwise correlation from (N, T) array with NaN.  May be non-PSD.
    """
    S   = pairwise_cov_np(R_nan)
    d   = np.sqrt(np.maximum(np.nandiag(S), 1e-12))
    C   = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


def _nandiag(M):
    return np.array([M[i, i] for i in range(M.shape[0])])


# Monkey-patch helper so pairwise_corr_np works even if np.nandiag doesn't exist
np.nandiag = _nandiag


# ─── PSD repair ──────────────────────────────────────────────────────────────

def nearest_psd_corr(C, max_iter=200):
    """
    Project C onto the cone of PSD correlation matrices (unit diagonal).
    Uses Higham (2002) alternating projections.
    """
    try:
        return nearcorr(C, max_iterations=max_iter)
    except Exception:
        # Fallback: eigenvalue clip
        lam, Q = np.linalg.eigh(C)
        lam    = np.maximum(lam, 1e-8)
        C_psd  = Q @ np.diag(lam) @ Q.T
        d      = np.sqrt(np.diag(C_psd))
        C_psd /= np.outer(d, d)
        np.fill_diagonal(C_psd, 1.0)
        return 0.5 * (C_psd + C_psd.T)


# ─── Combined estimators ──────────────────────────────────────────────────────

def pairwise_psd(R_nan, max_iter=200):
    """Pairwise MLE → nearest PSD correlation. R_nan: (N, T) with NaN."""
    C_pair = pairwise_corr_np(R_nan)
    C_pair = np.where(np.isfinite(C_pair), C_pair, 0.0)
    np.fill_diagonal(C_pair, 1.0)
    return nearest_psd_corr(0.5 * (C_pair + C_pair.T), max_iter)


def is_psd(C, tol=0.0):
    """Return True if C is positive semi-definite (all eigenvalues ≥ -tol)."""
    lam = np.linalg.eigvalsh(C)
    return bool(lam.min() >= -tol)
