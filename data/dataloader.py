import numpy as np
import torch
from scipy import stats
from data.missing_patterns import make_monotone_pattern
from data.estimators import (
    fit_monotone_regressions,
    reconstruct_mu_sigma_from_phi,
)
from models.utils import eigen_decomp
from data.estimators import torch_cov_pairwise


# generator of the data as suggested by Prof B
def data_generator(batch_size, N_min=20, N_max=300, T_min=50, T_max=300):
    while True:
        lam_emp = []
        Q_emp = []

        N = np.random.randint(N_min, N_max + 1)
        T = np.random.randint(T_min, T_max + 1)

        # sample true covariance
        df = np.random.uniform(2 * (N + 1), 10 * N, size=batch_size)
        # as suggested by prof B so that the population will not be the same and
        # so we don't have the neural network that will learn directly the target without looking at the input.

        Sigma_true_np = np.zeros((batch_size, N, N))
        for i in range(batch_size):
            Sigma_true_np[i] = stats.invwishart.rvs(
                df=df[i], scale=np.eye(N)
            ) * (  # cannot be fully vectorized bc of invwishart
                2 * (N + 1) - N - 1
            )
        Sigma_true = torch.tensor(Sigma_true_np, dtype=torch.float32)

        # Simulate T samples
        Z = torch.randn(batch_size, T, N, dtype=torch.float32)
        L = torch.linalg.cholesky(Sigma_true)
        R_torch = L @ Z.transpose(1, 2)

        R = R_torch.cpu().numpy()

        # Empirical covariance
        R_hat_list = []
        t_vec_list = []
        mask_list = []
        for i in range(batch_size):
            R_hat_i, t_vec_i, mask_i = make_monotone_pattern(R[i])
            R_hat_list.append(R_hat_i)
            t_vec_list.append(t_vec_i)
            mask_list.append(mask_i)

        mask_np = np.stack(mask_list, axis=0)
        mask = torch.tensor(mask_np, dtype=torch.bool)

        R_hat_np = np.stack(R_hat_list, axis=0)
        R_hat = torch.tensor(R_hat_np, dtype=torch.float32)

        Sigma_hat = torch_cov_pairwise(
            R_hat
        )  # takes a tensor (B, N, T) returns (B, N, N)

        eigvals, eigvecs = torch.linalg.eigh(Sigma_hat)

        eigvals_desc = torch.flip(eigvals, dims=[1])  # (B, N)
        eigvecs_desc = torch.flip(eigvecs, dims=[2])  # (B, N, N)

        lam_emp = eigvals_desc.unsqueeze(-1).to(torch.float32)  # (B, N, 1)
        Q_emp = eigvecs_desc.to(torch.float32)  # (B, N, N)

        # m_asset = mask.float().mean(dim=2)  # (B, N)

        # effective coverage par direction
        # v2 = Q_emp**2  # (B, N, N)
        # alpha = (v2 * m_asset.unsqueeze(-1)).sum(dim=1)  # (B, N)

        # last T observations with mask for each asset for each batch
        Tmin = mask.float().argmax(dim=2).unsqueeze(-1)  # (B, N, 1)
        Tmax = mask.flip(dims=[2]).float().argmax(dim=2).unsqueeze(-1)  # (B, N, 1)

        Tmin_mean = Q_emp.transpose(1, 2) @ Tmin.float()
        Tmax_mean = Q_emp.transpose(1, 2) @ Tmax.float()

        yield lam_emp, Q_emp, Sigma_true, T, Tmin_mean, Tmax_mean


def data_generator_2types(batch_size, N_min=20, N_max=300, T_min=50, T_max=300):
    while True:
        lam_emp = []
        Q_emp = []

        N = np.random.randint(N_min, N_max + 1)
        T = np.random.randint(T_min, T_max + 1)

        # sample true covariance
        df = np.random.uniform(2 * (N + 1), 10 * N, size=batch_size)
        # as suggested by prof B so that the population will not be the same and
        # so we don't have the neural network that will learn directly the target without looking at the input.

        Sigma_true_np = np.zeros((batch_size, N, N))
        for i in range(batch_size):
            Sigma_true_np[i] = stats.invwishart.rvs(
                df=df[i], scale=np.eye(N)
            ) * (  # cannot be fully vectorized bc of invwishart
                2 * (N + 1) - N - 1
            )
        Sigma_true = torch.tensor(Sigma_true_np, dtype=torch.float32)

        # Simulate T samples
        Z = torch.randn(batch_size, T, N, dtype=torch.float32)
        L = torch.linalg.cholesky(Sigma_true)
        R_torch = L @ Z.transpose(1, 2)

        R = R_torch.cpu().numpy()

        # Empirical covariance
        R_hat_list = []
        t_vec_list = []
        mask_list = []
        Sigmas_hat_1 = []

        for i in range(batch_size):
            R_hat_i, t_vec_i, mask_i = make_monotone_pattern(R[i])
            R_hat_list.append(R_hat_i)
            t_vec_list.append(t_vec_i)
            mask_list.append(mask_i)

        mask_np = np.stack(mask_list, axis=0)
        mask = torch.tensor(mask_np, dtype=torch.bool)

        R_hat_np = np.stack(R_hat_list, axis=0)
        R_hat = torch.tensor(R_hat_np, dtype=torch.float32)

        Sigma_hat = torch_cov_pairwise(
            R_hat
        )  # takes a tensor (B, N, T) returns (B, N, N)

        eigvals, eigvecs = torch.linalg.eigh(Sigma_hat)

        eigvals_desc = torch.flip(eigvals, dims=[1])  # (B, N)
        eigvecs_desc = torch.flip(eigvecs, dims=[2])  # (B, N, N)

        lam_emp = eigvals_desc.unsqueeze(-1).to(torch.float32)  # (B, N, 1)
        Q_emp = eigvecs_desc.to(torch.float32)  # (B, N, N)

        # m_asset = mask.float().mean(dim=2)  # (B, N)

        # effective coverage par direction
        # v2 = Q_emp**2  # (B, N, N)
        # alpha = (v2 * m_asset.unsqueeze(-1)).sum(dim=1)  # (B, N)

        # last T observations with mask for each asset for each batch
        Tmin = mask.float().argmax(dim=2).unsqueeze(-1)  # (B, N, 1)
        Tmax = mask.flip(dims=[2]).float().argmax(dim=2).unsqueeze(-1)  # (B, N, 1)

        for i in range(batch_size):
            R_hat_i, t_vec_i, _ = make_monotone_pattern(R[i])
            _, Sigma_hat_1 = reconstruct_mu_sigma_from_phi(
                fit_monotone_regressions(R_hat_i, t_vec_i)
            )
            Sigmas_hat_1.append(Sigma_hat_1)
            R_hat_list.append(R_hat_i)
            t_vec_list.append(t_vec_i)

        Sigmas_hat_1 = np.stack(Sigmas_hat_1, axis=0)
        Sigma_hat_1 = torch.tensor(Sigmas_hat_1, dtype=torch.float32)

        eigvals_1, eigvecs_1 = torch.linalg.eigh(Sigma_hat_1)

        eigvals_desc_1 = torch.flip(eigvals_1, dims=[1])  # (B, N)
        eigvecs_desc_1 = torch.flip(eigvecs_1, dims=[2])  # (B, N, N)

        lam_emp_1 = eigvals_desc_1.unsqueeze(-1).to(torch.float32)  # (B, N, 1)
        Q_emp_1 = eigvecs_desc_1.to(torch.float32)  # (B, N, N)

        yield lam_emp, Q_emp, Sigma_true, T, Tmin, Tmax, lam_emp_1, Q_emp_1
