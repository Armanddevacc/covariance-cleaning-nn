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
        Sigma_true = np.zeros((batch_size, N, N))

        for i in range(batch_size):
            Sigma_true[i] = stats.invwishart.rvs(
                df=df[i], scale=np.eye(N)
            ) * (  # cannot be fully vectorized bc of invwishart
                2 * (N + 1) - N - 1
            )

        # Simulate T samples
        R = np.zeros((batch_size, N, T))
        for i in range(batch_size):
            R[i] = np.random.multivariate_normal(np.zeros(N), Sigma_true[i], size=T).T
        # Empirical covariance
        R_hat_list = []
        t_vec_list = []
        for i in range(batch_size):
            R_hat_i, t_vec_i, _ = make_monotone_pattern(R[i])
            R_hat_list.append(R_hat_i)
            t_vec_list.append(t_vec_i)

        R_hat = np.stack(R_hat_list, axis=0)
        # _, Sigma_hat = reconstruct_mu_sigma_from_phi(
        #    fit_monotone_regressions(R_hat, t_vec)
        # )
        Sigma_hat = torch_cov_pairwise(
            torch.tensor(R_hat, dtype=torch.float64)
        ).numpy()  # takes a tensor (B, N, T) returns (B, N, N)

        for i in range(batch_size):
            vals, vecs = np.linalg.eigh(Sigma_hat[i])
            idx = vals.argsort()[::-1]
            lam_emp.append(vals[idx])
            Q_emp.append(vecs[:, idx])

        lam_emp = torch.tensor(np.array(lam_emp), dtype=torch.float32).unsqueeze(-1)
        Q_emp = torch.tensor(np.array(Q_emp), dtype=torch.float32)
        Sigma_true = torch.tensor(Sigma_true, dtype=torch.float32)

        yield lam_emp, Q_emp, Sigma_true, T
