import numpy as np
import torch
from scipy import stats
from data.missing_patterns import make_monotone_pattern, make_random_pattern_vecto
from data.estimators import (
    fit_monotone_regressions,
    reconstruct_mu_sigma_from_phi,
)
from models.utils import eigen_decomp
from data.estimators import torch_cov_pairwise
import scipy.stats as st


# generator of the data as suggested by Prof B
def data_generator(batch_size, N_min=20, N_max=300, T_min=50, T_max=300, df_min_factor= 1, df_max_factor= 10 ):
    while True:
        N = np.random.randint(N_min, N_max + 1)
        T = np.random.randint(T_min, T_max + 1)

        # --------------- use inverse wishart to sample true covariance ----------------

        df = np.random.randint( df_min_factor * (N + 2), df_max_factor * N, size=batch_size)  # degrees of freedom for invwishart
        invwishart_sampler = np.vectorize(lambda x: st.invwishart.rvs(df=x, scale=np.eye(N))*(x-N-1), 
                                       signature='()->(n,n)')
        Sigma_true = invwishart_sampler(df)
        Sigma_true = torch.tensor(Sigma_true, dtype=torch.float32)
        # we don't center nor normalize since i think it is unnecessary here

        # -------------------- Simulate T samples, vectorized ---------------------------
        Z = torch.randn(batch_size, T, N)
        L = torch.linalg.cholesky(Sigma_true)
        R = L @ Z.transpose(1, 2)  # (B, N, T)

        # Empirical covariance
        R_hat, _, mask = make_random_pattern_vecto(R)  # (B, N, T), (B, N), (B, N, T)

        Sigma_hat = torch_cov_pairwise(
            R_hat
        )  # takes a tensor (B, N, T) returns (B, N, N)

        eigvals, eigvecs = torch.linalg.eigh(
            Sigma_hat
        )  # eigh because always symetric by construction

        lam_emp = torch.flip(eigvals, dims=[1]).unsqueeze(-1)  # (B, N)
        Q_emp = torch.flip(eigvecs, dims=[2])  # (B, N, N)

        Tmin = mask.float().argmax(dim=2).unsqueeze(-1)  # (B, N, 1)
        Tmax = mask.flip(dims=[2]).float().argmax(dim=2).unsqueeze(-1)  # (B, N, 1)

        Tmin_mean = Q_emp.transpose(1, 2).pow(2) @ Tmin.float()
        Tmax_mean = Q_emp.transpose(1, 2).pow(2) @ Tmax.float()

        yield lam_emp, Q_emp, Sigma_true, T, Tmin_mean, Tmax_mean


def data_generator_2types(batch_size, N_min=20, N_max=300, T_min=50, T_max=300):
    while True:
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
        Sigmas_shaffer_list = []
        for i in range(batch_size):
            R_hat_i, t_vec_i, mask_i = make_monotone_pattern(R[i])
            _, Sigma_shaffer = reconstruct_mu_sigma_from_phi(
                fit_monotone_regressions(R_hat_i, t_vec_i)
            )
            R_hat_list.append(R_hat_i)
            t_vec_list.append(t_vec_i)
            mask_list.append(mask_i)
            Sigmas_shaffer_list.append(Sigma_shaffer)

        # ------------------------------ additional shaffer estimator -----------------------------
        Sigmas_shaffer_np = np.stack(Sigmas_shaffer_list, axis=0)
        Sigmas_shaffer = torch.tensor(Sigmas_shaffer_np, dtype=torch.float32)

        eigvals_shaffer, eigvecs_shaffer = torch.linalg.eigh(Sigmas_shaffer)

        eigvals_desc_shaffer = torch.flip(eigvals_shaffer, dims=[1])  # (B, N)
        eigvecs_desc_shaffer = torch.flip(eigvecs_shaffer, dims=[2])  # (B, N, N)

        lam_emp_shaffer = eigvals_desc_shaffer.unsqueeze(-1).to(
            torch.float32
        )  # (B, N, 1)
        Q_emp_shaffer = eigvecs_desc_shaffer.to(torch.float32)  # (B, N, N)

        # -----------------------------------------------------------------------------------------
        mask_np = np.stack(mask_list, axis=0)
        mask = torch.tensor(mask_np, dtype=torch.bool)

        R_hat_np = np.stack(R_hat_list, axis=0)
        R_hat = torch.tensor(R_hat_np, dtype=torch.float32)

        # --------- torch covariance with pairwise complete observations --------------------------
        Sigma_hat = torch_cov_pairwise(
            R_hat
        )  # takes a tensor (B, N, T) returns (B, N, N)

        eigvals, eigvecs = torch.linalg.eigh(Sigma_hat)

        eigvals_desc = torch.flip(eigvals, dims=[1])  # (B, N)
        eigvecs_desc = torch.flip(eigvecs, dims=[2])  # (B, N, N)

        lam_emp = eigvals_desc.unsqueeze(-1).to(torch.float32)  # (B, N, 1)
        Q_emp = eigvecs_desc.to(torch.float32)  # (B, N, N)

        # --------- Compute additional features ----------------------------------------------------
        Tmin = mask.float().argmax(dim=2).unsqueeze(-1)  # (B, N, 1)
        Tmax = mask.flip(dims=[2]).float().argmax(dim=2).unsqueeze(-1)  # (B, N, 1)

        Tmin_mean = Q_emp.transpose(1, 2) @ Tmin.float()
        Tmax_mean = Q_emp.transpose(1, 2) @ Tmax.float()

        yield lam_emp, Q_emp, Sigma_true, T, Tmin_mean, Tmax_mean, lam_emp_shaffer, Q_emp_shaffer
