import numpy as np
import torch
from data.missing_patterns import (
    make_random_pattern_vecto,
    tf_make_random_pattern_vecto,
)
from estimator.shaffer import (
    fit_monotone_regressions,
    reconstruct_mu_sigma_from_phi,
)
from estimator.MLE import torch_cov_pairwise, tf_cov_pairwise
import scipy.stats as st
from estimator.QIS import QIS_batched
from data.real_dataloader import real_data_pipeline


# generator of the data as suggested by Prof B
def data_generator(
    batch_size,
    missing_constant,  # >= 1, 1 being no missingness
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
        Sigma_true = torch.tensor(Sigma_true, dtype=torch.float64)
        # We don't center nor normalize since I think it is unnecessary here

        # -------------------- Simulate T samples, vectorized ---------------------------
        Z = torch.randn(batch_size, T, N, dtype=torch.float64)
        L = torch.linalg.cholesky(Sigma_true)
        R = L @ Z.transpose(1, 2)  # (B, N, T)

        # Empirical covariance
        R_hat, _, mask = make_random_pattern_vecto(
            R, missing_constant
        )  # (B, N, T), (B, N), (B, N, T)

        Sigma_hat = torch_cov_pairwise(R_hat).double()  # (B, N, N)
        Sigma_hat_diag = torch.diagonal(Sigma_hat, dim1=1, dim2=2)
        # By definition Sigma_hat is symetric but numericaly it can be asymmetric at ~1e-12 level for instance, the lines prevent it
        # Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.transpose(-1, -2))

        # ----------------------- Compute correlation Matrix ----------------------------
        # eps = 1e-6
        std_pred = torch.sqrt(Sigma_hat_diag)
        corr_hat = Sigma_hat / (std_pred[:, None, :] * std_pred[:, :, None])
        corr_hat = 0.5 * (corr_hat + corr_hat.transpose(-1, -2))

        # corr_hat = 0.5 * (corr_hat + corr_hat.transpose(-1, -2))
        # jitter = 1e-8
        # corr_hat = corr_hat + jitter * torch.eye(N, device=corr_hat.device)

        eigvals, eigvecs = torch.linalg.eigh(
            corr_hat
        )  # eigh because always symetric by construction

        eigvals = eigvals.float()  # float to prepare batch
        eigvecs = eigvecs.float()

        # eps = 1e-6

        # Floor eigenvalues
        # eigvals = torch.clamp(eigvals, min=eps)

        # Enforce trace = N
        # eigvals = eigvals / eigvals.mean(dim=1, keepdim=True)

        lam_emp = eigvals.unsqueeze(-1)
        Q_emp = eigvecs
        # lam_emp = torch.flip(eigvals, dims=[1]).unsqueeze(-1)  # (B, N, 1)
        # Q_emp = torch.flip(eigvecs, dims=[2])  # (B, N, N)

        Tmin = mask.float().argmax(dim=2).unsqueeze(-1)  # (B, N, 1)
        Tmax = mask.flip(dims=[2]).float().argmax(dim=2).unsqueeze(-1)  # (B, N, 1)

        Tmin_mean = Q_emp.transpose(1, 2).pow(2) @ Tmin.float()
        Tmax_mean = Q_emp.transpose(1, 2).pow(2) @ Tmax.float()

        # ----------------------------------- Prepare Batch --------------------------------------

        # Build conditioning scalars
        T_vec = torch.full(
            (lam_emp.shape[0], lam_emp.shape[1], 1), T, dtype=torch.float32
        )
        N_vec = torch.full(
            (lam_emp.shape[0], lam_emp.shape[1], 1),
            lam_emp.shape[1],
            dtype=torch.float32,
        )

        # Build input sequence to the GRU
        input_seq = torch.cat(
            [lam_emp, N_vec, T_vec, N_vec / T_vec, Tmin_mean, Tmax_mean], dim=-1
        )

        yield input_seq, Q_emp, Sigma_true, T, Sigma_hat_diag, R


import tensorflow as tf


# generator of the data as suggested by Prof B but in tensorflow
def tf_data_generator(
    batch_size,
    missing_constant,  # >= 1, 1 being no missingness
    N_min=20,
    N_max=300,
    # T_min=50,
    # T_max=300,
    q_min=0.1,
    q_max=2,
    df_min_factor=1.5,
    df_max_factor=3,
):
    while True:

        N = np.random.randint(N_min, N_max + 1)

        # u = np.random.uniform(0, 1)
        # to draw more big q than small q
        # q = q_min + (q_max - q_min) * u**0.5
        q = np.random.uniform(q_min, q_max + 1)

        T = int(N / q)

        # --------------- use inverse wishart to sample true covariance ----------------

        df = np.random.randint(
            int(df_min_factor * (N + 2)), int(df_max_factor * N), size=batch_size
        )  # degrees of freedom for invwishart

        Sigma_true = np.stack(
            [st.invwishart.rvs(df=d, scale=np.eye(N)) * (d - N - 1) for d in df],
            axis=0,
        )

        Sigma_true = tf.convert_to_tensor(Sigma_true, dtype=tf.float64)  # on average 1

        # -------------------- Simulate T samples, vectorized ---------------------------

        Z = tf.random.normal((batch_size, T, N), dtype=tf.float64)
        L = tf.linalg.cholesky(Sigma_true)  # (B, N, N)
        R = tf.matmul(L, tf.transpose(Z, perm=[0, 2, 1]))  # (B, N, T)

        # Empirical covariance
        R_hat, _, mask = tf_make_random_pattern_vecto(
            R, missing_constant
        )  # (B, N, T), (B, N), (B, N, T)

        Sigma_hat = tf_cov_pairwise(R_hat)  # (B, N, N)

        # -------------------- Compute correlation Matrix ----------------------------

        Sigma_hat_diag = tf.linalg.diag_part(Sigma_hat)
        std_pred = tf.sqrt(Sigma_hat_diag)
        corr_hat = Sigma_hat / (std_pred[:, None, :] * std_pred[:, :, None])

        # We increase symmetry numerically to prevent issues with eigen decomposition, it can be asymmetric at ~1e-12 level for instance
        # We have to do it because sometimes in the trainer the matrix was ill-conditioned, not allowing usage of eigh
        corr_hat = 0.5 * (corr_hat + tf.transpose(corr_hat, perm=[0, 2, 1]))

        # other way to increase symmetry numerically : (not used for now as it was not needed)
        # jitter = 1e-8
        # corr_hat = corr_hat + jitter * tf.eye(
        #    N, batch_shape=[batch_size], dtype=tf.float64
        # )

        # eigh because always symetric by construction
        eigvals, eigvecs = tf.linalg.eigh(corr_hat)

        # everything was in 64 bit before but switch to 32 for now (remaining part of pipeline is in 32)
        eigvals = tf.cast(eigvals, tf.float32)
        eigvecs = tf.cast(eigvecs, tf.float32)

        lam_emp = tf.expand_dims(eigvals, axis=-1)  # (B, N, 1)
        Q_emp = eigvecs  # (B, N, N)

        # ---------------- Feature engineering to try to absorb missingness pattern ---------------
        # positional feature so the NN knows where it is in the spectrum
        pos = tf.linspace(0.0, 1.0, N)  # (N,)
        pos = tf.reshape(pos, (1, N, 1))  # (1,N,1)
        pos = tf.tile(pos, [batch_size, 1, 1])  # (B,N,1)

        Q_sq = tf.square(tf.transpose(Q_emp, perm=[0, 2, 1]))

        Tmin = tf.cast(tf.argmax(mask, axis=2, output_type=tf.int32), tf.float32)
        Tmin = Tmin / tf.cast(T, tf.float32)
        Tmin = tf.expand_dims(Tmin, axis=-1)  # (B, N, 1)
        Tminmean = tf.matmul(Q_sq, Tmin)

        Tmax = tf.cast(
            tf.shape(mask)[2]
            - 1
            - tf.argmax(tf.reverse(mask, axis=[2]), axis=2, output_type=tf.int32),
            tf.float32,
        )
        Tmax = Tmax / tf.cast(T, tf.float32)
        Tmax = tf.expand_dims(Tmax, axis=-1)  # (B, N, 1)
        Tmaxmean = tf.matmul(Q_sq, Tmax)

        # ----------------------------------- Prepare Batch --------------------------------------

        # Build conditioning scalars
        T_vec = tf.fill((batch_size, N, 1), tf.cast(T, tf.float32))
        N_vec = tf.fill((batch_size, N, 1), tf.cast(N, tf.float32))

        # Build input sequence to the GRU
        input_seq = tf.concat(
            [
                lam_emp,
                pos,
                N_vec / T_vec,
                Tminmean,
                Tmaxmean,
            ],  # , N_vec, T_vec, Tmin, Tmax
            axis=-1,
        )

        yield input_seq, Q_emp, Sigma_true, T, Sigma_hat_diag, R_hat
