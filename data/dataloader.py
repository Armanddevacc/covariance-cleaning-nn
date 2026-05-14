import numpy as np
import scipy.stats as st
import tensorflow as tf
from data.missing_patterns import tf_make_random_pattern_vecto
from estimator.MLE import tf_cov_pairwise


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

        q = np.random.uniform(q_min, q_max)

        T = int(N / q)

        # --------------- use inverse wishart to sample true covariance ----------------

        # single df for the whole batch — avoids 50 serial scipy calls
        df_val = np.random.randint(int(df_min_factor * (N + 2)), int(df_max_factor * N))

        # IW(df, I_N): sample Z ~ N(0,1)^{df x N}, then (Z^T Z)^{-1} ~ IW(df, I_N)
        # scale by (df - N - 1) so E[Sigma_true] = I_N
        Z = np.random.randn(batch_size, df_val, N)              # (B, df, N)
        ZtZ = Z.transpose(0, 2, 1) @ Z                          # (B, N, N)  Wishart sample
        I_batch = np.eye(N)[None].repeat(batch_size, axis=0)    # (B, N, N)
        Sigma_true = np.linalg.solve(ZtZ, I_batch) * (df_val - N - 1)  # (B, N, N)

        Sigma_true = tf.convert_to_tensor(Sigma_true, dtype=tf.float64)

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

        # effective concentration ratio per eigenmode:
        # eigenmode j is built from assets whose first observation is at Tminmean[j]*T,
        # so it has only (1 - Tminmean[j]) * T observations on average.
        # q_j^eff = N / ((1 - Tminmean_j) * T) tells the network how noisy eigenmode j is.
        effective_T_frac = tf.maximum(1.0 - Tminmean, 1.0 / tf.cast(T, tf.float32))
        q_eff = (N_vec / T_vec) / effective_T_frac  # (B, N, 1)

        # IPR: N * Σ_i Q_{ij}^4 — eigenmode localization on the asset space
        # Q_sq[b, j, i] = Q_emp[b, i, j]^2  →  Q_sq^2 gives Q_emp^4
        ipr = tf.cast(N, tf.float32) * tf.reduce_sum(tf.square(Q_sq), axis=2)  # (B, N)
        ipr = tf.expand_dims(ipr, axis=-1)  # (B, N, 1)

        # z_MP: Marchenko-Pastur z-score — distance from upper bulk edge in units of √q_eff
        q_eff_safe = tf.maximum(q_eff, 1e-6)
        lam_plus = (1.0 + tf.sqrt(q_eff_safe)) ** 2  # upper MP bulk edge
        z_MP = (lam_emp - lam_plus) / tf.sqrt(q_eff_safe)  # (B, N, 1)

        # Build input sequence to the GRU  — shape (B, N, 5) — S3 feature set
        input_seq = tf.concat(
            [
                lam_emp,  # 0
                pos,      # 1
                q_eff,    # 2
                ipr,      # 3
                z_MP,     # 4
            ],
            axis=-1,
        )

        yield input_seq, Q_emp, Sigma_true, T, Sigma_hat_diag, R_hat


def tf_data_generator_nomiss_oos(
    batch_size,
    N_fixed,
    q_fixed,
    T_oos=50,
    df_min_factor=1.5,
    df_max_factor=3,
):
    """
    Fixed-(N, q) no-missingness generator with separate OOS returns.

    q_fixed determines T = N / q_fixed (in-sample length).
    T_oos OOS returns are drawn from the same Sigma_true for the variance loss.

    Yields:
        input_seq       (B, N, 5)        same 5-feature format as tf_data_generator
        Q_emp           (B, N, N)        empirical eigenvectors (from in-sample corr)
        Sigma_true      (B, N, N)        true covariance (float64)
        T               int              in-sample length
        Sigma_hat_diag  (B, N)           diagonal of empirical covariance
        R_in            (B, N, T)        in-sample returns (float32)
        R_oos           (B, N, T_oos)    OOS returns (float32)
    """
    while True:
        N = N_fixed
        T = int(N / q_fixed)

        df_val = np.random.randint(int(df_min_factor * (N + 2)), int(df_max_factor * N))
        Z_iw = np.random.randn(batch_size, df_val, N)
        ZtZ = Z_iw.transpose(0, 2, 1) @ Z_iw
        I_batch = np.eye(N)[None].repeat(batch_size, axis=0)
        Sigma_true = np.linalg.solve(ZtZ, I_batch) * (df_val - N - 1)
        Sigma_true_tf = tf.convert_to_tensor(Sigma_true, dtype=tf.float64)

        L = tf.linalg.cholesky(Sigma_true_tf)  # (B, N, N)

        # In-sample returns (fully observed)
        Z_in = tf.random.normal((batch_size, T, N), dtype=tf.float64)
        R_in = tf.matmul(L, tf.transpose(Z_in, perm=[0, 2, 1]))  # (B, N, T)

        # OOS returns from same Sigma_true
        Z_oos = tf.random.normal((batch_size, T_oos, N), dtype=tf.float64)
        R_oos = tf.cast(
            tf.matmul(L, tf.transpose(Z_oos, perm=[0, 2, 1])), tf.float32
        )  # (B, N, T_oos)

        # Empirical covariance → correlation
        Sigma_hat = tf_cov_pairwise(R_in)  # (B, N, N)
        Sigma_hat_diag = tf.linalg.diag_part(Sigma_hat)
        std_pred = tf.sqrt(Sigma_hat_diag)
        corr_hat = Sigma_hat / (std_pred[:, None, :] * std_pred[:, :, None])
        corr_hat = 0.5 * (corr_hat + tf.transpose(corr_hat, perm=[0, 2, 1]))

        eigvals, eigvecs = tf.linalg.eigh(corr_hat)
        eigvals = tf.cast(eigvals, tf.float32)
        eigvecs = tf.cast(eigvecs, tf.float32)

        lam_emp = tf.expand_dims(eigvals, axis=-1)  # (B, N, 1)
        Q_emp = eigvecs  # (B, N, N)

        # Features — no missingness → q_eff = q_fixed for all eigenmodes
        pos = tf.tile(
            tf.reshape(tf.linspace(0.0, 1.0, N), (1, N, 1)), [batch_size, 1, 1]
        )  # (B, N, 1)
        Q_sq = tf.square(tf.transpose(Q_emp, perm=[0, 2, 1]))  # (B, N, N)

        q_eff = tf.fill((batch_size, N, 1), tf.cast(q_fixed, tf.float32))  # (B, N, 1)

        ipr = tf.cast(N, tf.float32) * tf.reduce_sum(tf.square(Q_sq), axis=2)
        ipr = tf.expand_dims(ipr, axis=-1)  # (B, N, 1)

        q_eff_safe = tf.maximum(q_eff, 1e-6)
        lam_plus = (1.0 + tf.sqrt(q_eff_safe)) ** 2
        z_MP = (lam_emp - lam_plus) / tf.sqrt(q_eff_safe)  # (B, N, 1)

        input_seq = tf.concat([lam_emp, pos, q_eff, ipr, z_MP], axis=-1)  # (B, N, 5)

        yield input_seq, Q_emp, Sigma_true_tf, T, Sigma_hat_diag, tf.cast(R_in, tf.float32), R_oos
