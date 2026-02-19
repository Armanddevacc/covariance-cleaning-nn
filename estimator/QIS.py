import numpy as np
import torch


def QIS(Sample, ddof=1):
    """
    Compute the Quadratic-Inverse Shrinkage (QIS) estimator for a given sample.
    Parameters:
    Sample (numpy.ndarray): A 2D array where rows represent the stocks and columns represent the time series.
    ddof (int, optional): Degrees of freedom correction for the sample mean. Default is 1: Mean is subtracted.
    Returns:
    numpy.ndarray: The QIS estimator of the covariance matrix.
    Notes:
    - The input sample is expected to be a NumPy array.
    - The function ensures the symmetry of the sample covariance matrix.
    - Eigenvalues are clipped to be non-negative.
    - The function uses a smoothing parameter for the shrinkage estimator.
    - The trace of the covariance matrix is preserved in the final estimator.
    """

    # Y is expected to be a NumPy array
    N, p = Sample.shape  # Get dimensions of Y

    # Default setting: if k is None or NaN, set k = 1 (no de-mean)
    if ddof >= 1:
        Sample = Sample - np.mean(Sample, axis=0)

    # Vars
    n = N - ddof  # Adjust effective sample size
    c = p / n  # Concentration ratio

    # Compute sample covariance matrix
    sample = np.matmul(Sample.T, Sample) / n
    sample = (sample + sample.T) / 2  # Ensure symmetry

    # Eigenvalue decomposition (use eigh for Hermitian matrix)
    lambda1, u = np.linalg.eigh(sample)
    lambda1 = np.clip(
        lambda1.real, a_min=0, a_max=None
    )  # Clip negative eigenvalues to 0

    # Compute Quadratic-Inverse Shrinkage estimator
    h = (min(c**2, 1 / c**2) ** 0.35) / p**0.35  # Smoothing parameter

    # Inverse of (non-null) eigenvalues
    invlambda = 1 / lambda1[max(1, p - n + 1) - 1 : p]

    # Calculate Lj and Lj_i (Differences of inverse eigenvalues)
    Lj = np.repeat(invlambda[:, np.newaxis], min(p, n), axis=1)
    Lj_i = Lj - Lj.T

    # Smoothed Stein shrinker (theta) and its conjugate (Htheta)
    Lj_squared = Lj * Lj
    theta = np.mean(Lj * Lj_i / (Lj_i * Lj_i + Lj_squared * h**2), axis=0)
    Htheta = np.mean(Lj * Lj * h / (Lj_i * Lj_i + Lj_squared * h**2), axis=0)
    Atheta2 = theta**2 + Htheta**2  # Squared amplitude

    # Shrink eigenvalues based on p and n
    if p <= n:
        delta = 1 / (
            (1 - c) ** 2 * invlambda
            + 2 * c * (1 - c) * invlambda * theta
            + c**2 * invlambda * Atheta2
        )
    else:
        delta0 = 1 / ((c - 1) * np.mean(invlambda))  # Shrinkage of null eigenvalues
        delta = np.concatenate((np.full(p - n, delta0), 1 / (invlambda * Atheta2)))

    # Preserve the trace
    deltaQIS = delta * (np.sum(lambda1) / np.sum(delta))

    # Reconstruct covariance matrix
    sigmahat = np.matmul(u, np.matmul(np.diag(deltaQIS), u.T))

    return sigmahat


import numpy as np


def QIS_batched_numpy(Sample, ddof=1):
    """
    Quadratic-Inverse Shrinkage batched version.
    Sample: (B, N, T)
    Returns: (B, N, N)
    """

    B, N, T = Sample.shape

    if ddof >= 1:
        Sample = Sample - Sample.mean(axis=2, keepdims=True)

    n = T - ddof
    p = N
    c = p / n

    # Sample covariance (B,N,N)
    sample = Sample @ Sample.transpose(0, 2, 1) / n
    sample = 0.5 * (sample + sample.transpose(0, 2, 1))

    # Eigen decomposition (batched)
    lambda1, u = np.linalg.eigh(sample)
    lambda1 = np.clip(lambda1.real, 0, None)

    # Smoothing parameter
    h = (min(c**2, 1 / c**2) ** 0.35) / p**0.35

    # Non-null eigenvalues
    start = max(1, p - n + 1) - 1
    invlambda = 1.0 / lambda1[:, start:p]  # (B, m)

    m = invlambda.shape[1]

    # Build Lj and Lj_i batched
    Lj = np.repeat(invlambda[:, :, None], m, axis=2)
    Lj_i = Lj - Lj.transpose(0, 2, 1)

    Lj_squared = Lj * Lj

    theta = np.mean(Lj * Lj_i / (Lj_i**2 + Lj_squared * h**2), axis=1)

    Htheta = np.mean(Lj_squared * h / (Lj_i**2 + Lj_squared * h**2), axis=1)

    Atheta2 = theta**2 + Htheta**2

    if p <= n:
        delta = 1.0 / (
            (1 - c) ** 2 * invlambda
            + 2 * c * (1 - c) * invlambda * theta
            + c**2 * invlambda * Atheta2
        )
    else:
        delta0 = 1.0 / ((c - 1) * np.mean(invlambda, axis=1, keepdims=True))
        delta = np.concatenate(
            (np.repeat(delta0, p - n, axis=1), 1.0 / (invlambda * Atheta2)), axis=1
        )

    # Preserve trace
    deltaQIS = delta * (
        lambda1.sum(axis=1, keepdims=True) / delta.sum(axis=1, keepdims=True)
    )

    # Reconstruct covariance
    sigmahat = u @ (deltaQIS[:, :, None] * u.transpose(0, 2, 1))

    return sigmahat


def QIS_batched(sample, ddof=1):
    """
    Batched QIS estimator.

    sample: (B, N, T)
    returns sigmahat: (B, N, N)
    """

    B, N, T = sample.shape

    # De-mean across time if ddof≥1
    if ddof >= 1:
        sample = sample - sample.mean(dim=2, keepdim=True)

    n = T - ddof
    c = N / n

    # Sample covariance across time → (B, N, N)
    S = sample @ sample.transpose(1, 2) / n
    S = 0.5 * (S + S.transpose(1, 2))

    # Eigendecomposition – batched
    lambda1, U = torch.linalg.eigh(S)
    lambda1 = torch.clamp(lambda1, min=0.0)

    # h smoothing parameter (scalar)
    h = (min(c**2, 1 / c**2) ** 0.35) / (N**0.35)

    # Inverse of non-null eigenvalues
    start = max(1, N - n + 1) - 1
    invlambda = 1.0 / lambda1[:, start:N]  # (B, K), K = min(N,n)

    K = invlambda.shape[1]

    # Lj and Lj_i matrices
    Lj = invlambda.unsqueeze(2).expand(B, K, K)
    Lj_i = Lj - Lj.transpose(1, 2)

    Lj_sq = Lj * Lj
    denom = Lj_i * Lj_i + Lj_sq * (h**2)

    theta = (Lj * Lj_i / denom).mean(dim=1)  # (B, K)
    Htheta = (Lj_sq * h / denom).mean(dim=1)  # (B, K)
    Atheta2 = theta**2 + Htheta**2  # (B, K)

    # Shrink eigenvalues
    if N <= n:
        delta = 1 / (
            (1 - c) ** 2 * invlambda
            + 2 * c * (1 - c) * invlambda * theta
            + c**2 * invlambda * Atheta2
        )
    else:
        delta0 = 1 / ((c - 1) * invlambda.mean(dim=1, keepdim=True))
        delta = torch.cat([delta0.repeat(1, N - n), 1 / (invlambda * Atheta2)], dim=1)

    # Trace preservation
    deltaQIS = delta * (
        lambda1.sum(dim=1, keepdim=True) / delta.sum(dim=1, keepdim=True)
    )

    # Reconstruct covariance (B,N,N)
    D = torch.diag_embed(deltaQIS)
    sigma = U @ D @ U.transpose(1, 2)

    return sigma  # (B, N, N)


import tensorflow as tf


def tf_QIS_batched(sample, ddof=1):
    """
    Batched QIS estimator.

    sample: (B, N, T)
    returns sigmahat: (B, N, N)
    """

    sample = tf.convert_to_tensor(sample)
    dtype = sample.dtype

    B = tf.shape(sample)[0]
    N = tf.shape(sample)[1]
    T = tf.shape(sample)[2]

    # De-mean across time if ddof ≥ 1
    if ddof >= 1:
        mean = tf.reduce_mean(sample, axis=2, keepdims=True)
        sample = sample - mean

    n = tf.cast(T - ddof, dtype)
    c = tf.cast(N, dtype) / n

    # Sample covariance across time → (B, N, N)
    S = tf.matmul(sample, sample, transpose_b=True) / n
    S = 0.5 * (S + tf.transpose(S, perm=[0, 2, 1]))

    # Eigendecomposition – batched
    lambda1, U = tf.linalg.eigh(S)
    lambda1 = tf.maximum(lambda1, tf.cast(0.0, dtype))

    # h smoothing parameter (scalar)
    h = tf.pow(tf.minimum(c**2, 1.0 / (c**2)), tf.cast(0.35, dtype)) / tf.pow(
        tf.cast(N, dtype), tf.cast(0.35, dtype)
    )

    # Inverse of non-null eigenvalues
    start = tf.maximum(1, N - tf.cast(n, tf.int32) + 1) - 1
    invlambda = 1.0 / lambda1[:, start:]  # (B, K)

    K = tf.shape(invlambda)[1]

    # Lj and Lj_i matrices
    Lj = tf.expand_dims(invlambda, 2)
    Lj = tf.tile(Lj, [1, 1, K])

    Lj_i = Lj - tf.transpose(Lj, perm=[0, 2, 1])

    Lj_sq = Lj * Lj
    denom = Lj_i * Lj_i + Lj_sq * (h**2)

    theta = tf.reduce_mean(Lj * Lj_i / denom, axis=1)  # (B, K)
    Htheta = tf.reduce_mean(Lj_sq * h / denom, axis=1)  # (B, K)
    Atheta2 = theta**2 + Htheta**2  # (B, K)

    # Shrink eigenvalues
    if_condition = tf.less_equal(N, tf.cast(n, tf.int32))

    def case_N_le_n():
        return 1.0 / (
            (1 - c) ** 2 * invlambda
            + 2 * c * (1 - c) * invlambda * theta
            + c**2 * invlambda * Atheta2
        )

    def case_N_gt_n():
        delta0 = 1.0 / ((c - 1) * tf.reduce_mean(invlambda, axis=1, keepdims=True))
        delta0 = tf.repeat(delta0, repeats=N - tf.cast(n, tf.int32), axis=1)
        return tf.concat([delta0, 1.0 / (invlambda * Atheta2)], axis=1)

    delta = tf.cond(if_condition, case_N_le_n, case_N_gt_n)

    # Trace preservation
    trace_lambda = tf.reduce_sum(lambda1, axis=1, keepdims=True)
    trace_delta = tf.reduce_sum(delta, axis=1, keepdims=True)

    deltaQIS = delta * (trace_lambda / trace_delta)

    # Reconstruct covariance (B,N,N)
    D = tf.linalg.diag(deltaQIS)
    sigma = tf.matmul(tf.matmul(U, D), U, transpose_b=True)

    return sigma  # (B, N, N)
