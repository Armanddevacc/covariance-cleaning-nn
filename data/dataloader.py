import numpy as np
import torch
from scipy import stats
from data.missing_patterns import make_monotone_pattern
from data.regression_estimators import (
    fit_monotone_regressions,
    reconstruct_mu_sigma_from_phi,
)
from models.utils import eigen_decomp


# generator of the data as suggested by Prof B
def data_generator(batch_size, N_min=20, N_max=300, T_min=50, T_max=300):
    while True:
        N = np.random.randint(N_min, N_max + 1)
        T = np.random.randint(T_min, T_max + 1)

        lam_emp_list = []
        Q_list = []
        Sigma_true_list = []

        for _ in range(batch_size):

            # sample true covariance
            df = np.random.uniform(2 * (N + 1), 10 * N)
            # as suggested by prof B so that the population will not be the same and
            # so we don't have the neural network that will learn directly the target without looking at the input.
            Sigma_true = stats.invwishart.rvs(df=df, scale=np.eye(N)) * (
                2 * (N + 1) - N - 1
            )

            # Simulate T samples
            R = np.random.multivariate_normal(np.zeros(N), Sigma_true, size=T).T

            # Empirical covariance
            R_hat, t_vec, _ = make_monotone_pattern(R)
            _, Sigma_hat = reconstruct_mu_sigma_from_phi(
                fit_monotone_regressions(R_hat, t_vec)
            )

            # sort eigenvalues
            lam_emp, Q_emp = eigen_decomp(Sigma_hat)
            # Store
            lam_emp_list.append(lam_emp.squeeze())
            Q_list.append(Q_emp)
            Sigma_true_list.append(Sigma_true)

        lam_emp = torch.tensor(np.array(lam_emp_list), dtype=torch.float32).unsqueeze(
            -1
        )
        Q_emp = torch.tensor(np.array(Q_list), dtype=torch.float32)
        Sigma_true = torch.tensor(np.array(Sigma_true_list), dtype=torch.float32)

        # Yield batch
        yield lam_emp, Q_emp, Sigma_true, T
