import torch
import numpy as np
from utils import reconstruct


# Potters–Bouchaud loss
def loss_function_mat(Mat_oos, Mat_pred, T):
    B, N, _ = Mat_oos.shape

    # Matrix difference
    Delta = Mat_pred - Mat_oos  # (B, N, N)

    ## CB: It seems more efficient to compute the Frobenius norm via squaring element-wise and summing.
    # Square of the matrix (Delta^2 = Delta @ Delta) Symetric matrix so we don't need to transpose !
    Delta2 = torch.transpose(Delta, 1, 2) @ Delta  # (B, N, N)

    # Trace of Delta2 = sum of diagonal
    trace_vals = Delta2.diagonal(dim1=1, dim2=2).sum(dim=1)  # (B,)

    # Normalized Frobenius estimation error (Potters-Bouchaud)
    loss_cov = torch.sqrt(trace_vals) * T / N**2  # (B,)

    return loss_cov.mean()  # scalar


def log_spectrum_mse(Mat_oos, Mat_pred, T):
    """
    lam_pred:   (B, N) predicted eigenvalues (positive)
    lam_target: (B, N) target eigenvalues (positive)
    """
    eps = 1e-6
    lam_pred, _ = torch.linalg.eigh(Mat_oos)
    log_pred = torch.log(lam_pred + eps)

    lam_target, _ = torch.linalg.eigh(Mat_pred)
    log_tgt = torch.log(lam_target + eps)
    return ((log_pred - log_tgt) ** 2).mean()


import tensorflow as tf


# Potters–Bouchaud loss for corr with tensorflow
def loss_function_corr_tensorflow(Corr_oos, Corr_pred, T):
    """
    Corr_true: (B, N, N)
    Corr_pred: (B, N, N)
    T: scalar (int or float)
    """
    B, N, _ = Corr_oos.shape

    Delta = Corr_pred - Corr_oos  # (B, N, N)

    Delta2 = Delta2 = tf.matmul(Delta, Delta, transpose_a=True)

    trace_vals = tf.linalg.trace(Delta2)  # (B,)

    loss_cov = tf.sqrt(trace_vals) * T / N**2  # (B,)

    return tf.reduce_mean(loss_cov)  # (1)


# Potters–Bouchaud loss
def loss_function(lam_pred, Q, Sigma_oos, T):
    B, N = lam_pred.shape

    # build Lambda_pred
    Lambda_pred = torch.diag_embed(lam_pred)  # (B, N, N)

    # Reconstruct covariance(s) for all batch samples
    Sigma_pred = Q @ Lambda_pred @ Q.transpose(1, 2)  # (B, N, N)

    # Matrix difference
    Delta = Sigma_pred - Sigma_oos  # (B, N, N)

    ## CB: It seems more efficient to compute the Frobenius norm via squaring element-wise and summing.
    # Square of the matrix (Delta^2 = Delta @ Delta) Symetric matrix so we don't need to transpose !
    Delta2 = Delta @ Delta  # (B, N, N)

    # Trace of Delta2 = sum of diagonal
    trace_vals = Delta2.diagonal(dim1=1, dim2=2).sum(dim=1)  # (B,)

    # Normalized Frobenius estimation error (Potters-Bouchaud)
    loss_cov = trace_vals * T / N**2  # (B,)

    return loss_cov.mean()  # scalar


# NOT USED For NOW
# loss function which is actually : -profit, beacause we want to maximize them
def loss_function_portfolio(lam_pred, Q, R_oos, T):
    Sigma = reconstruct(Q, lam_pred)
    B, N, _ = Sigma.shape
    ones = torch.ones(B, N, 1, dtype=Sigma.dtype)
    x = torch.linalg.solve(Sigma, ones)
    weights = x / x.sum(dim=1, keepdim=True)
    p_oos = weights.transpose(1, 2) @ R_oos[:, :, T:]
    return -p_oos.mean()
