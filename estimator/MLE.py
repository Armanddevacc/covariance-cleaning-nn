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


import tensorflow as tf


# fully vectorized covariance with pairwise-complete observations like pandas.cov()
def tf_cov_pairwise(X):
    """
    input X: (N, T) or (B, N, T)
    """
    X = tf.convert_to_tensor(X)
    original_rank = X.shape.rank  # can be None in graph, but in eager it's fine

    if original_rank == 2:
        X = tf.expand_dims(X, axis=0)  # becomes (1, N, T)

    X = tf.cast(X, tf.float64)  # match your generator precision

    # mask: 1 where valid, 0 where NaN
    mask = tf.math.is_finite(X)  # (B, N, T)
    mask_f = tf.cast(mask, tf.float64)

    # mean over time dimension (T), ignoring NaNs
    X0 = tf.where(mask, X, tf.zeros_like(X))  # replace NaN by 0 for sums
    cnt = tf.reduce_sum(mask_f, axis=-1, keepdims=True)  # (B, N, 1)
    means = tf.reduce_sum(X0, axis=-1, keepdims=True) / tf.maximum(cnt, 1.0)  # (B,N,1)

    # centered data (NaNs propagate in Xc, but we will zero them out next)
    Xc = X - means  # (B, N, T)

    # centered data but NaN replaced by 0
    Xc_zero = tf.where(mask, Xc, tf.zeros_like(Xc))

    # pairwise valid counts n_ij = sum_t mask[i,t] * mask[j,t]
    valid_counts = tf.matmul(mask_f, mask_f, transpose_b=True)  # (B, N, N)

    # numerator: sum of centered products
    numerator = tf.matmul(Xc_zero, Xc_zero, transpose_b=True)  # (B, N, N)

    # denominator: n_ij - 1
    denom = valid_counts - 1.0
    nan = tf.constant(float("nan"), dtype=tf.float64)
    denom = tf.where(denom > 0.0, denom, nan)

    cov = numerator / denom

    # drop batch if original input had no batch
    if original_rank == 2:
        cov = cov[0]

    return cov
