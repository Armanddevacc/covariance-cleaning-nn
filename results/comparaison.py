import tensorflow as tf
import numpy as np

from estimator.QIS import tf_QIS_batched


def frobenius_mean(A, B):
    A = tf.cast(A, tf.float32)
    B = tf.cast(B, tf.float32)
    diff = A - B
    frob_per_batch = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=[1, 2]))
    return tf.reduce_mean(frob_per_batch)


def accumulate_loss(
    model_generated_data,
    accumulate_step,
    q_min,
    q_max,
    N_min,
    N_max,
    data_gen,
    loss_function,
    batch_size_oos=25,
):
    frob_corr_loss = np.empty(shape=(accumulate_step, 5))
    frob_cov_loss = np.empty(shape=(accumulate_step, 5))

    for step in range(accumulate_step):
        (
            input_seq_cov_miss,
            Q_emp_miss,
            Sigma_true_miss,
            T,
            Sigma_hat_diag_miss,
            R_miss,
        ) = next(
            data_gen(
                batch_size_oos,
                missing_constant=2,
                N_min=N_min,
                N_max=N_max,
                q_min=q_min,
                q_max=q_max,
            )
        )
        (
            input_seq_cov_no_miss,
            Q_emp_no_miss,
            Sigma_true_no_miss,
            T,
            Sigma_hat_diag_no_miss,
            R_no_miss,
        ) = next(
            data_gen(
                batch_size_oos,
                missing_constant=1,
                N_min=N_min,
                N_max=N_max,
                q_min=q_min,
                q_max=q_max,
            )
        )

        R_np = R_miss.numpy()
        born_per_sample = (~np.isnan(R_np)).all(axis=1).sum(axis=-1)

        lam_emp_miss     = input_seq_cov_miss[:, :, 0]
        lam_pred_miss    = model_generated_data(input_seq_cov_miss)
        lam_pred_no_miss = model_generated_data(input_seq_cov_no_miss)

        # --- correlation matrices ---
        def _corr(Q, lam):
            return tf.cast(
                tf.matmul(tf.matmul(Q, tf.linalg.diag(lam)), Q, transpose_b=True),
                tf.float32,
            )

        Corr_pred_miss    = _corr(Q_emp_miss,    lam_pred_miss)
        Corr_pred_no_miss = _corr(Q_emp_no_miss, lam_pred_no_miss)
        Corr_emp_miss     = _corr(Q_emp_miss,    lam_emp_miss)

        def _true_corr(Sigma):
            d = tf.linalg.diag_part(Sigma)
            s = tf.sqrt(tf.maximum(d, 1e-12))
            return tf.cast(Sigma / (s[:, None, :] * s[:, :, None] + 1e-12), tf.float32)

        Corr_true_miss    = _true_corr(Sigma_true_miss)
        Corr_true_no_miss = _true_corr(Sigma_true_no_miss)

        # --- oracle: true eigenvalues in empirical eigenvectors ---
        lam_true_corr, _ = tf.linalg.eigh(Corr_true_miss)
        C_oracle_corr = _corr(Q_emp_miss, lam_true_corr)

        # --- covariance reconstruction ---
        D_miss    = tf.cast(tf.sqrt(tf.linalg.diag(Sigma_hat_diag_miss)),    tf.float32)
        D_no_miss = tf.cast(tf.sqrt(tf.linalg.diag(Sigma_hat_diag_no_miss)), tf.float32)

        Sigma_pred_miss    = tf.matmul(tf.matmul(D_miss,    Corr_pred_miss),    D_miss)
        Sigma_pred_no_miss = tf.matmul(tf.matmul(D_no_miss, Corr_pred_no_miss), D_no_miss)
        Sigma_emp_miss     = tf.matmul(tf.matmul(D_miss,    Corr_emp_miss),     D_miss)
        Sigma_oracle       = tf.matmul(tf.matmul(D_miss,    C_oracle_corr),     D_miss)

        # --- per-sample QIS ---
        Sigma_qis_list = []
        for b in range(R_np.shape[0]):
            born_b = int(born_per_sample[b])
            if born_b >= 2:
                R_b = tf.constant(R_np[b:b+1, :, -born_b:], dtype=tf.float32)
                Sigma_qis_list.append(tf_QIS_batched(R_b)[0])
            else:
                Sigma_qis_list.append(tf.cast(Sigma_emp_miss[b], tf.float32))
        Sigma_QIS = tf.cast(tf.stack(Sigma_qis_list, axis=0), tf.float32)

        # QIS correlation
        d_qis   = tf.linalg.diag_part(Sigma_QIS)
        std_qis = tf.sqrt(tf.maximum(d_qis, 1e-12))
        Corr_QIS = tf.cast(
            Sigma_QIS / (std_qis[:, None, :] * std_qis[:, :, None] + 1e-12), tf.float32
        )

        Sigma_true_miss    = tf.cast(Sigma_true_miss,    tf.float32)
        Sigma_true_no_miss = tf.cast(Sigma_true_no_miss, tf.float32)

        # --- losses  (cols: NN_miss=0  NN_nomiss=1  Emp_miss=2  QIS=3  Oracle=4) ---
        frob_corr_loss[step, :] = [
            loss_function(Corr_pred_miss,    Corr_true_miss,    T),
            loss_function(Corr_pred_no_miss, Corr_true_no_miss, T),
            loss_function(Corr_emp_miss,     Corr_true_miss,    T),
            loss_function(Corr_QIS,          Corr_true_miss,    T),
            loss_function(C_oracle_corr,     Corr_true_miss,    T),
        ]
        frob_cov_loss[step, :] = [
            loss_function(Sigma_pred_miss,    Sigma_true_miss,    T),
            loss_function(Sigma_pred_no_miss, Sigma_true_no_miss, T),
            loss_function(Sigma_emp_miss,     Sigma_true_miss,    T),
            loss_function(Sigma_QIS,          Sigma_true_miss,    T),
            loss_function(Sigma_oracle,       Sigma_true_miss,    T),
        ]

    return frob_corr_loss, frob_cov_loss


def accumulate_eigenvalue_data(model, data_gen, steps, batch_size=20,
                               N_fixed=100, q_min=0.3, q_max=1.5):
    """
    Per-rank eigenvalue MSE, bias, and mean profiles across steps.

    Returns
    -------
    mse_arr  : (steps, 4, 3) — quartile MSE;  estimators: [NN, Emp, QIS]
    bias_arr : (steps, 4, 3) — quartile signed bias
    profiles : dict with 'emp', 'pred', 'true' — mean eigenvalue profile, shape (N_fixed,)
    """
    N = N_fixed
    mse_arr  = np.zeros((steps, 4, 3))
    bias_arr = np.zeros((steps, 4, 3))
    lam_emp_acc  = []
    lam_pred_acc = []
    lam_true_acc = []

    slices = [
        slice(0,          N // 4),
        slice(N // 4,     N // 2),
        slice(N // 2,   3 * N // 4),
        slice(3 * N // 4,        N),
    ]

    for step in range(steps):
        input_seq, _, Sigma_true, _, _, R_miss = next(
            data_gen(batch_size, missing_constant=2,
                     N_min=N_fixed, N_max=N_fixed,
                     q_min=q_min, q_max=q_max)
        )
        B    = int(input_seq.shape[0])
        R_np = R_miss.numpy()

        lam_emp  = input_seq[:, :, 0].numpy()
        lam_pred = np.sort(model(input_seq, training=False).numpy(), axis=1)

        Sigma_np = Sigma_true.numpy()
        lam_true = np.zeros((B, N))
        for b in range(B):
            d = np.diag(Sigma_np[b])
            s = np.sqrt(np.maximum(d, 1e-12))
            C = Sigma_np[b] / np.outer(s, s)
            np.fill_diagonal(C, 1.0)
            lam_true[b] = np.linalg.eigvalsh(C)

        born_per = (~np.isnan(R_np)).all(axis=1).sum(axis=-1)
        lam_qis  = np.zeros((B, N))
        for b in range(B):
            born_b = int(born_per[b])
            R_b = tf.constant(R_np[b:b+1, :, -born_b:], dtype=tf.float32)
            Sigma_b = tf_QIS_batched(R_b).numpy()[0]
            d = np.diag(Sigma_b)
            s = np.sqrt(np.maximum(d, 1e-12))
            C = Sigma_b / np.outer(s, s)
            np.fill_diagonal(C, 1.0)
            C = 0.5 * (C + C.T) + 1e-6 * np.eye(N)
            lam_qis[b] = np.linalg.eigvalsh(C)

        lam_emp_acc.append(lam_emp)
        lam_pred_acc.append(lam_pred)
        lam_true_acc.append(lam_true)

        for k, sl in enumerate(slices):
            for j_est, lam_hat in enumerate([lam_pred, lam_emp, lam_qis]):
                err = lam_hat[:, sl] - lam_true[:, sl]
                mse_arr[step, k, j_est]  = np.mean(err ** 2)
                bias_arr[step, k, j_est] = np.mean(err)

    profiles = {
        'emp':  np.mean(np.concatenate(lam_emp_acc,  axis=0), axis=0),
        'pred': np.mean(np.concatenate(lam_pred_acc, axis=0), axis=0),
        'true': np.mean(np.concatenate(lam_true_acc, axis=0), axis=0),
    }
    return mse_arr, bias_arr, profiles


from estimator.MLE import tf_cov_pairwise
from estimator.MLE import tf_cov_pairwise_mask
from estimator.QIS import tf_QIS_batched


def construct_input_seq(rin, mask):
    """5-feature input matching Trainer_real_data_tf._construct_input_seq.

    rin:  (B, N, T) z-scored in-sample returns, 0 at missing positions
    mask: (B, N, T) True where MISSING
    """
    B, N, T = tf.shape(rin)

    observed = ~mask
    Sigma_hat = tf_cov_pairwise_mask(rin, observed)

    Sigma_hat_diag = tf.linalg.diag_part(Sigma_hat)
    Sigma_hat_diag = tf.maximum(Sigma_hat_diag, 1e-8)
    std_pred = tf.sqrt(Sigma_hat_diag)

    corr_hat = Sigma_hat / (std_pred[:, :, None] * std_pred[:, None, :])
    corr_hat = tf.where(tf.math.is_finite(corr_hat), corr_hat, tf.zeros_like(corr_hat))
    eye = tf.eye(N, batch_shape=[B], dtype=corr_hat.dtype)
    corr_hat = corr_hat - tf.linalg.diag(tf.linalg.diag_part(corr_hat)) + eye
    corr_hat = 0.5 * (corr_hat + tf.transpose(corr_hat, perm=[0, 2, 1]))
    corr_hat = corr_hat + 1e-6 * eye

    eigvals, eigvecs = tf.linalg.eigh(corr_hat)
    eigvals = tf.cast(eigvals, tf.float32)
    eigvecs = tf.cast(eigvecs, tf.float32)

    lam_emp = tf.expand_dims(eigvals, axis=-1)
    Q_emp   = eigvecs

    pos  = tf.tile(tf.reshape(tf.linspace(0.0, 1.0, N), (1, N, 1)), [B, 1, 1])
    Q_sq = tf.square(tf.transpose(Q_emp, perm=[0, 2, 1]))  # (B, N, N)

    observed_f = tf.cast(observed, tf.float32)
    Tmin = tf.cast(tf.argmax(observed_f, axis=2, output_type=tf.int32), tf.float32)
    Tmin = tf.expand_dims(Tmin / tf.cast(T, tf.float32), axis=-1)  # (B, N, 1)
    Tminmean = tf.matmul(Q_sq, Tmin)                                 # (B, N, 1)

    T_vec = tf.fill((B, N, 1), tf.cast(T, tf.float32))
    N_vec = tf.fill((B, N, 1), tf.cast(N, tf.float32))

    effective_T_frac = tf.maximum(1.0 - Tminmean, 1.0 / tf.cast(T, tf.float32))
    q_eff = (N_vec / T_vec) / effective_T_frac                       # (B, N, 1)

    ipr = tf.cast(N, tf.float32) * tf.reduce_sum(tf.square(Q_sq), axis=2, keepdims=True)  # (B, N, 1)

    q_eff_safe = tf.maximum(q_eff, 1e-6)
    z_MP = (lam_emp - tf.square(1.0 + tf.sqrt(q_eff_safe))) / tf.sqrt(q_eff_safe)  # (B, N, 1)

    input_seq = tf.concat([lam_emp, pos, q_eff, ipr, z_MP], axis=-1)  # (B, N, 5)

    return input_seq, Q_emp, Sigma_hat_diag, T


def accumulate_loss_realdata(
    model,
    dataset,
    steps,
    loss_function,
):

    frob_corr_loss = np.zeros((steps, 2))
    frob_cov_loss = np.zeros((steps, 3))

    for step, ((rin, mask), rout) in enumerate(dataset.take(steps)):

        # ===== true covariance from OOS =====
        Sigma_true = tf_cov_pairwise(rout)
        Sigma_true = tf.cast(Sigma_true, tf.float32)

        diag_true = tf.linalg.diag_part(Sigma_true)
        std_true = tf.sqrt(tf.maximum(diag_true, 1e-12))

        Corr_true = Sigma_true / (std_true[:, None, :] * std_true[:, :, None] + 1e-12)
        Corr_true = tf.where(tf.math.is_finite(Corr_true), Corr_true, tf.zeros_like(Corr_true))
        Corr_true = tf.cast(Corr_true, tf.float32)
        # ===== construct same inputs as trainer =====
        input_seq, Q_emp, Sigma_hat_diag, T = construct_input_seq(rin, mask)
        Sigma_hat_diag = tf.cast(Sigma_hat_diag, tf.float32)
        Q_emp = tf.cast(Q_emp, tf.float32)

        # empirical correlation
        lam_emp = input_seq[:, :, 0]

        Corr_emp = tf.matmul(
            tf.matmul(Q_emp, tf.linalg.diag(lam_emp)),
            Q_emp,
            transpose_b=True,
        )

        # ===== model prediction =====
        lam_pred = model(input_seq, training=False)

        Corr_pred = tf.matmul(
            tf.matmul(Q_emp, tf.linalg.diag(lam_pred)),
            Q_emp,
            transpose_b=True,
        )

        # ===== correlation losses =====
        fro_corr_pred = loss_function(Corr_pred, Corr_true, T)
        fro_corr_emp = loss_function(Corr_emp, Corr_true, T)

        frob_corr_loss[step] = [fro_corr_pred, fro_corr_emp]

        # ===== covariance reconstruction =====
        D = tf.sqrt(tf.linalg.diag(Sigma_hat_diag))

        Sigma_pred = tf.matmul(tf.matmul(D, Corr_pred), D)
        Sigma_emp = tf.matmul(tf.matmul(D, Corr_emp), D)

        mask_np = mask.numpy()
        born = int(np.min((~mask_np).all(axis=1).sum(axis=-1)))
        N_val = int(rin.shape[1])

        try:
            if born >= N_val + 2:
                Sigma_QIS = tf.cast(tf_QIS_batched(rin[:, :, -born:]), tf.float32)
            else:
                raise ValueError("born too small")
        except Exception:
            Sigma_QIS = tf.cast(Sigma_emp, tf.float32)  # fallback to empirical

        fro_cov_pred = loss_function(Sigma_pred, Sigma_true, T)
        fro_cov_emp = loss_function(Sigma_emp, Sigma_true, T)
        fro_cov_qis = loss_function(Sigma_QIS, Sigma_true, T)

        frob_cov_loss[step] = [
            fro_cov_pred,
            fro_cov_emp,
            fro_cov_qis,
        ]

    return frob_corr_loss, frob_cov_loss


def accumulate_portfolio_variance_realdata(model, dataset, steps):
    """
    Evaluate realized min-variance portfolio variance on short OOS returns.

    Returns (steps, 3) array — columns: [NN, Emp_pairwise, QIS].
    QIS falls back to empirical when the synchronous window is too short (born < N+2).
    """
    from models.losses import tf_variance_loss

    port_var = np.zeros((steps, 3))

    for step, ((rin, mask), rout_nan) in enumerate(dataset.take(steps)):
        rout = tf.where(tf.math.is_nan(rout_nan), tf.zeros_like(rout_nan), rout_nan)
        rout = tf.cast(rout, tf.float32)

        input_seq, Q_emp, Sigma_hat_diag, T = construct_input_seq(rin, mask)
        Q_emp   = tf.cast(Q_emp, tf.float32)
        lam_emp = input_seq[:, :, 0]

        lam_pred = model(input_seq, training=False)
        nn_var   = float(tf_variance_loss(lam_pred, Q_emp, rout))
        emp_var  = float(tf_variance_loss(lam_emp,  Q_emp, rout))

        # Per-sample QIS: each sample uses its own synchronous window
        mask_np = mask.numpy()
        rin_np  = rin.numpy() if hasattr(rin, 'numpy') else np.array(rin)
        rout_np = rout.numpy() if hasattr(rout, 'numpy') else np.array(rout)
        B_val   = mask_np.shape[0]
        N_val   = mask_np.shape[1]

        qis_vars = []
        for b in range(B_val):
            born_b = int((~mask_np[b]).all(axis=0).sum())
            r_b    = tf.constant(rout_np[b:b+1], dtype=tf.float32)  # (1, N, T_oos)
            try:
                if born_b >= N_val + 2:
                    rin_b     = tf.constant(rin_np[b:b+1, :, -born_b:], dtype=tf.float32)
                    Sigma_b   = tf.cast(tf_QIS_batched(rin_b), tf.float32)[0]  # (N, N)
                    diag_b    = tf.linalg.diag_part(Sigma_b)
                    std_b     = tf.sqrt(tf.maximum(diag_b, 1e-8))
                    Corr_b    = Sigma_b / (std_b[:, None] * std_b[None, :])
                    Corr_b    = tf.where(tf.math.is_finite(Corr_b), Corr_b, tf.zeros_like(Corr_b))
                    eye_b     = tf.eye(N_val)
                    Corr_b    = Corr_b - tf.linalg.diag(tf.linalg.diag_part(Corr_b)) + eye_b
                    lam_b, Q_b = tf.linalg.eigh(Corr_b + 1e-6 * eye_b)
                    lam_b     = tf.maximum(tf.cast(lam_b, tf.float32), 1e-4)[None, :]
                    Q_b       = tf.cast(Q_b, tf.float32)[None, :, :]
                    qis_vars.append(float(tf_variance_loss(lam_b, Q_b, r_b)))
                else:
                    qis_vars.append(float(tf_variance_loss(
                        lam_emp[b:b+1], Q_emp[b:b+1], r_b
                    )))
            except Exception:
                qis_vars.append(float(tf_variance_loss(
                    lam_emp[b:b+1], Q_emp[b:b+1], r_b
                )))

        port_var[step] = [nn_var, emp_var, float(np.mean(qis_vars))]

    return port_var
