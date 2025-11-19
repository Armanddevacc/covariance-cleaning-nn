import torch


def loss_function(lam_pred, Q, Sigma_true, T):
    N = Sigma_true.shape[1]

    Lambda_pred = torch.diag_embed(lam_pred)
    Sigma_pred = Q @ Lambda_pred @ Q.transpose(1, 2)

    frob_error = ((Sigma_pred - Sigma_true) ** 2).sum(dim=(1, 2))
    loss_cov = (T / N**2) * frob_error.mean()

    return loss_cov
