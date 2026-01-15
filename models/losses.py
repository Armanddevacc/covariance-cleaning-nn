import torch
from utils.utils import reconstruct_cov
import numpy as np


# Potters–Bouchaud loss for corr
def loss_function_corr(Corr_oos, Corr_pred, T):
    B, N, _ = Corr_oos.shape

    # Matrix difference
    Delta = Corr_pred - Corr_oos  # (B, N, N)

    ## CB: It seems more efficient to compute the Frobenius norm via squaring element-wise and summing.
    # Square of the matrix (Delta^2 = Delta @ Delta) Symetric matrix so we don't need to transpose !
    Delta2 = Delta @ Delta  # (B, N, N)

    # Trace of Delta2 = sum of diagonal
    trace_vals = Delta2.diagonal(dim1=1, dim2=2).sum(dim=1)  # (B,)

    # Normalized Frobenius estimation error (Potters-Bouchaud)
    loss_cov = trace_vals * T / N**2  # (B,)

    return loss_cov.mean()  # scalar


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


# loss function which is actually : -profit, beacause we want to maximize them
def loss_function_portfolio(lam_pred, Q, R_oos, T):
    Sigma = reconstruct_cov(Q, lam_pred)
    B, N, _ = Sigma.shape
    ones = torch.ones(B, N, 1, dtype=Sigma.dtype)
    x = torch.linalg.solve(Sigma, ones)
    weights = x / x.sum(dim=1, keepdim=True)
    p_oos = weights.transpose(1, 2) @ R_oos[:, :, T:]
    return -p_oos.mean()
