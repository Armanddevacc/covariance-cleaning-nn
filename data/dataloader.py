import numpy as np
import torch
from data.missing_patterns import make_random_pattern_vecto
from estimator.shaffer import (
    fit_monotone_regressions,
    reconstruct_mu_sigma_from_phi,
)
from models.utils import eigen_decomp
from estimator.shaffer import torch_cov_pairwise
import scipy.stats as st
from estimator.QIS import QIS_batched
from data.real_dataloader import real_data_pipeline


# generator of the data as suggested by Prof B
def data_generator(
    batch_size,
    N_min=20,
    N_max=300,
    T_min=50,
    T_max=300,
    df_min_factor=10,
    df_max_factor=100,
):
    while True:
        N = np.random.randint(N_min, N_max + 1)
        T = np.random.randint(T_min, T_max + 1)

        # --------------- use inverse wishart to sample true covariance ----------------

        df = np.random.randint(
            df_min_factor * (N + 2), df_max_factor * N, size=batch_size
        )  # degrees of freedom for invwishart
        invwishart_sampler = np.vectorize(
            lambda x: st.invwishart.rvs(df=x, scale=np.eye(N)) * (x - N - 1),
            signature="()->(n,n)",
        )
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


def data_generator_2types(
    batch_size,
    N_min=20,
    N_max=300,
    T_min=50,
    T_max=300,
    real_data=False,
    dataset=None,
    df_min_factor=1,
    df_max_factor=2,
):
    while True:
        N = np.random.randint(N_min, N_max + 1)
        T = np.random.randint(T_min, T_max + 1)

        (
            Sigma_true,
            lam_emp_cov_no_miss,
            Q_emp_cov_no_miss,
            lam_QIS_no_miss,
            Q_QIS_no_miss,
        ) = (
            None,
            None,
            None,
            None,
            None,
        )

        if not real_data:
            df = np.random.randint(
                df_min_factor * (N + 2), df_max_factor * N, size=batch_size
            )
            invwishart_sampler = np.vectorize(
                lambda x: st.invwishart.rvs(df=x, scale=np.eye(N)) * (x - N - 1),
                signature="()->(n,n)",
            )
            Sigma_true = invwishart_sampler(df)
            Sigma_true = torch.tensor(Sigma_true, dtype=torch.float32)

            Z = torch.randn(batch_size, T, N, dtype=torch.float32)
            L = torch.linalg.cholesky(Sigma_true)
            R = L @ Z.transpose(1, 2)

            R_miss, _, mask = make_random_pattern_vecto(
                R
            )  # (B, N, T), (B, N), (B, N, T)

            Sigma_hat_cov_no_miss = torch_cov_pairwise(R)

            eigvals_cov_no_miss, eigvecs_cov_no_miss = torch.linalg.eigh(
                Sigma_hat_cov_no_miss
            )

            lam_emp_cov_no_miss = torch.flip(eigvals_cov_no_miss, dims=[1]).unsqueeze(
                -1
            )
            Q_emp_cov_no_miss = torch.flip(eigvecs_cov_no_miss, dims=[2])

            # 2.
            Sigma_hat_QIS_no_miss = QIS_batched(R)

            eigvals_QIS_no_miss, eigvecs_QIS_no_miss = torch.linalg.eigh(
                Sigma_hat_QIS_no_miss
            )

            lam_QIS_no_miss = torch.flip(eigvals_QIS_no_miss, dims=[1]).unsqueeze(-1)
            Q_QIS_no_miss = torch.flip(eigvecs_QIS_no_miss, dims=[2])
        else:
            (rin_tf, mask_tf), R_miss_tf = next(iter(dataset))
            # rin = torch.from_numpy(rin_tf.numpy())
            mask = torch.from_numpy(mask_tf.numpy())

            R_miss = torch.from_numpy(R_miss_tf.numpy())

        Sigma_hat_cov_miss = torch_cov_pairwise(R_miss)

        eigvals_cov_miss, eigvecs_cov_miss = torch.linalg.eigh(Sigma_hat_cov_miss)
        lam_emp_cov_miss = torch.flip(eigvals_cov_miss, dims=[1]).unsqueeze(
            -1
        )  # (B, N)
        Q_emp_cov_miss = torch.flip(eigvecs_cov_miss, dims=[2])  # (B, N, N)

        Tmin = mask.float().argmax(dim=2).unsqueeze(-1)
        Tmax = mask.flip(dims=[2]).float().argmax(dim=2).unsqueeze(-1)

        Tmin_mean = Q_emp_cov_miss.transpose(1, 2).pow(2) @ Tmin.float()
        Tmax_mean = Q_emp_cov_miss.transpose(1, 2).pow(2) @ Tmax.float()

        # ------

        # Shaffer ...
        """# Empirical covariance
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
        """
        # -----------------------------------------------------------------------------------------

        # --------- other style --------------------------
        # 1.

        yield lam_emp_cov_miss, Q_emp_cov_miss, Sigma_true, T, Tmin_mean, Tmax_mean, lam_emp_cov_no_miss, Q_emp_cov_no_miss, lam_QIS_no_miss, Q_QIS_no_miss
