import torch


# Potters–Bouchaud loss
def loss_function_test(
    lam_pred, Q, Sigma_true, T
):  # size lam_pred (B, N, 1), Q (B, N, N), Sigma_true (B, N, N)
    N = Sigma_true.shape[1]

    Lambda_pred = torch.diag_embed(lam_pred.squeeze(-1))

    Sigma_pred = Q @ Lambda_pred @ Q.transpose(1, 2)

    error = torch.matmul(Sigma_pred - Sigma_true).diagonal(dim1=1, dim2=2).sum(dim=1)

    loss_cov = (1 / N) * error

    return loss_cov


# Potters–Bouchaud loss
def loss_function(lam_pred, Q, Sigma_true, T):
    B, N = lam_pred.shape

    # build Lambda_pred
    Lambda_pred = torch.diag_embed(lam_pred)  # (B, N, N)

    # Reconstruct covariance(s) for all batch samples
    Sigma_pred = Q @ Lambda_pred @ Q.transpose(1, 2)  # (B, N, N)

    # Matrix difference
    Delta = Sigma_pred - Sigma_true  # (B, N, N)

    ## CB: It seems more efficient to compute the Frobenius norm via squaring element-wise and summing.
    # Square of the matrix (Delta^2 = Delta @ Delta)
    Delta2 = Delta @ Delta  # (B, N, N)

    # Trace of Delta2 = sum of diagonal
    trace_vals = Delta2.diagonal(dim1=1, dim2=2).sum(dim=1)  # (B,)

    # Normalized Frobenius estimation error (Potters-Bouchaud)
    loss_cov = trace_vals * T / N**2  # (B,)

    return loss_cov.mean()  # scalar
