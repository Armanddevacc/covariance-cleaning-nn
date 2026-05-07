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
    frob_corr_loss = np.empty(shape=(accumulate_step, 4))
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

        # synchronous window: time steps where ALL assets are jointly observed
        # take the minimum across the batch so the slice is NaN-free for every sample
        born_per_sample = (~np.isnan(R_miss.numpy())).all(axis=1).sum(axis=-1)  # (B,)
        born = int(born_per_sample.min())
        lam_emp_miss = input_seq_cov_miss[:, :, 0]
        lam_pred_miss = model_generated_data(input_seq_cov_miss)  # ,lam_emp_miss
        lam_emp_no_miss = input_seq_cov_no_miss[:, :, 0]
        lam_pred_no_miss = model_generated_data(
            input_seq_cov_no_miss
        )  # ,lam_emp_no_miss

        # Corr from predicted eigenvalues
        Corr_pred_miss = tf.matmul(
            tf.matmul(Q_emp_miss, tf.linalg.diag(lam_pred_miss)),
            Q_emp_miss,
            transpose_b=True,
        )
        Corr_pred_no_miss = tf.matmul(
            tf.matmul(Q_emp_no_miss, tf.linalg.diag(lam_pred_no_miss)),
            Q_emp_no_miss,
            transpose_b=True,
        )

        # Corr from empirical eigenvalues
        Corr_emp_miss = tf.matmul(
            tf.matmul(Q_emp_miss, tf.linalg.diag(lam_emp_miss)),
            Q_emp_miss,
            transpose_b=True,
        )
        Corr_emp_no_miss = tf.matmul(
            tf.matmul(Q_emp_no_miss, tf.linalg.diag(lam_emp_no_miss)),
            Q_emp_no_miss,
            transpose_b=True,
        )

        # MISS
        diag_oos_miss = tf.linalg.diag_part(Sigma_true_miss)
        std_oos_miss = tf.sqrt(tf.maximum(diag_oos_miss, 1e-12))
        Corr_true_miss = Sigma_true_miss / (
            std_oos_miss[:, None, :] * std_oos_miss[:, :, None] + 1e-12
        )

        # NO MISS
        diag_oos_no_miss = tf.linalg.diag_part(Sigma_true_no_miss)
        std_oos_no_miss = tf.sqrt(tf.maximum(diag_oos_no_miss, 1e-12))
        Corr_true_no_miss = Sigma_true_no_miss / (
            std_oos_no_miss[:, None, :] * std_oos_no_miss[:, :, None] + 1e-12
        )

        Corr_pred_miss = tf.cast(Corr_pred_miss, tf.float32)
        Corr_pred_no_miss = tf.cast(Corr_pred_no_miss, tf.float32)
        Corr_emp_miss = tf.cast(Corr_emp_miss, tf.float32)
        Corr_emp_no_miss = tf.cast(Corr_emp_no_miss, tf.float32)

        Corr_true_miss = tf.cast(Corr_true_miss, tf.float32)
        Corr_true_no_miss = tf.cast(Corr_true_no_miss, tf.float32)

        # compute the frob loss
        fro_Corr_pred_miss = loss_function(Corr_pred_miss, Corr_true_miss, T)
        fro_Corr_pred_no_miss = loss_function(Corr_pred_no_miss, Corr_true_no_miss, T)
        fro_Corr_emp_miss = loss_function(Corr_emp_miss, Corr_true_miss, T)
        fro_Corr_emp_no_miss = loss_function(Corr_emp_no_miss, Corr_true_no_miss, T)

        frob_corr_loss[step, :] = [
            fro_Corr_pred_miss,
            fro_Corr_pred_no_miss,
            fro_Corr_emp_miss,
            fro_Corr_emp_no_miss,
        ]

        D_miss = tf.cast(tf.sqrt(tf.linalg.diag(Sigma_hat_diag_miss)), tf.float32)
        D_no_miss = tf.cast(tf.sqrt(tf.linalg.diag(Sigma_hat_diag_no_miss)), tf.float32)

        Sigma_pred_miss = tf.matmul(tf.matmul(D_miss, Corr_pred_miss), D_miss)
        Sigma_pred_no_miss = tf.matmul(
            tf.matmul(D_no_miss, Corr_pred_no_miss), D_no_miss
        )
        Sigma_emp_miss = tf.matmul(tf.matmul(D_miss, Corr_emp_miss), D_miss)
        Sigma_emp_no_miss = tf.matmul(tf.matmul(D_no_miss, Corr_emp_no_miss), D_no_miss)
        if born < 2:
            Sigma_QIS = tf.cast(Sigma_emp_miss, tf.float32)  # fallback: no sync window
        else:
            Sigma_QIS = tf_QIS_batched(R_miss[:, :, -born:])

        Sigma_true_miss = tf.cast(Sigma_true_miss, tf.float32)
        Sigma_true_no_miss = tf.cast(Sigma_true_no_miss, tf.float32)
        Sigma_QIS = tf.cast(Sigma_QIS, tf.float32)

        fro_Sigma_pred_miss = loss_function(Sigma_pred_miss, Sigma_true_miss, T)
        fro_Sigma_pred_no_miss = loss_function(
            Sigma_pred_no_miss, Sigma_true_no_miss, T
        )
        fro_Sigma_emp_miss = loss_function(Sigma_emp_miss, Sigma_true_miss, T)
        fro_Sigma_emp_no_miss = loss_function(Sigma_emp_no_miss, Sigma_true_no_miss, T)
        fro_Sigma_QIS = loss_function(Sigma_QIS, Sigma_true_miss, T)

        frob_cov_loss[step, :] = [
            fro_Sigma_pred_miss,
            fro_Sigma_pred_no_miss,
            fro_Sigma_emp_miss,
            fro_Sigma_emp_no_miss,
            fro_Sigma_QIS,
        ]

    return frob_corr_loss, frob_cov_loss


from estimator.MLE import tf_cov_pairwise
from estimator.MLE import tf_cov_pairwise_mask
from estimator.QIS import tf_QIS_batched


def construct_input_seq(rin, mask):
    B, N, T = tf.shape(rin)

    Sigma_hat = tf_cov_pairwise_mask(rin, ~mask)

    Sigma_hat_diag = tf.linalg.diag_part(Sigma_hat)
    Sigma_hat_diag = tf.maximum(Sigma_hat_diag, 1e-12)
    # we clamp bc some lines are all zeros but we can't just remove that line + it is rare so we prefer adding this
    std_pred = tf.sqrt(Sigma_hat_diag)

    corr_hat = Sigma_hat / (std_pred[:, None, :] * std_pred[:, :, None])
    corr_hat = 0.5 * (corr_hat + tf.transpose(corr_hat, perm=[0, 2, 1]))

    eigvals, eigvecs = tf.linalg.eigh(corr_hat)

    eigvals = tf.cast(eigvals, tf.float32)
    eigvecs = tf.cast(eigvecs, tf.float32)

    lam_emp = tf.expand_dims(eigvals, axis=-1)
    Q_emp = eigvecs

    pos = tf.linspace(0.0, 1.0, N)
    pos = tf.reshape(pos, (1, N, 1))
    pos = tf.tile(pos, [B, 1, 1])

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

    T_vec = tf.fill((B, N, 1), tf.cast(T, tf.float32))
    N_vec = tf.fill((B, N, 1), tf.cast(N, tf.float32))

    effective_T_frac = tf.maximum(1.0 - Tminmean, 1.0 / tf.cast(T, tf.float32))
    q_eff = (N_vec / T_vec) / effective_T_frac  # (B, N, 1)

    input_seq = tf.concat(
        [
            lam_emp,
            pos,
            N_vec / T_vec,
            Tminmean,
            Tmaxmean,
            q_eff,
        ],
        axis=-1,
    )

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
        born = np.min((~mask_np).all(axis=1).sum(axis=-1))

        Sigma_QIS = tf_QIS_batched(rin[:, :, -born:])
        Sigma_QIS = tf.cast(Sigma_QIS, tf.float32)

        fro_cov_pred = loss_function(Sigma_pred, Sigma_true, T)
        fro_cov_emp = loss_function(Sigma_emp, Sigma_true, T)
        fro_cov_qis = loss_function(Sigma_QIS, Sigma_true, T)

        frob_cov_loss[step] = [
            fro_cov_pred,
            fro_cov_emp,
            fro_cov_qis,
        ]

    return frob_corr_loss, frob_cov_loss
