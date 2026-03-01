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
    T_min,
    T_max,
    N_min,
    N_max,
    data_gen,
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
                T_min=T_min,
                T_max=T_max,
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
                T_min=T_min,
                T_max=T_max,
            )
        )

        born = np.min((~np.isnan(R_miss.numpy())).all(axis=1).sum(axis=-1))
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

        # compute the frob loss
        fro_Corr_pred_miss = frobenius_mean(Corr_pred_miss, Corr_true_miss)
        fro_Corr_pred_no_miss = frobenius_mean(Corr_pred_no_miss, Corr_true_no_miss)
        fro_Corr_emp_miss = frobenius_mean(Corr_emp_miss, Corr_true_miss)
        fro_Corr_emp_no_miss = frobenius_mean(Corr_emp_no_miss, Corr_true_no_miss)

        frob_corr_loss[step, :] = [
            fro_Corr_pred_miss,
            fro_Corr_pred_no_miss,
            fro_Corr_emp_miss,
            fro_Corr_emp_no_miss,
        ]

        D_miss = tf.cast(tf.sqrt(tf.linalg.diag(Sigma_hat_diag_miss)), tf.float32)
        D_no_miss = tf.cast(tf.sqrt(tf.linalg.diag(Sigma_hat_diag_no_miss)), tf.float32)

        Corr_pred_miss = tf.cast(Corr_pred_miss, tf.float32)
        Corr_pred_no_miss = tf.cast(Corr_pred_no_miss, tf.float32)
        Corr_emp_miss = tf.cast(Corr_emp_miss, tf.float32)
        Corr_emp_no_miss = tf.cast(Corr_emp_no_miss, tf.float32)

        Sigma_pred_miss = tf.matmul(tf.matmul(D_miss, Corr_pred_miss), D_miss)
        Sigma_pred_no_miss = tf.matmul(
            tf.matmul(D_no_miss, Corr_pred_no_miss), D_no_miss
        )
        Sigma_emp_miss = tf.matmul(tf.matmul(D_miss, Corr_emp_miss), D_miss)
        Sigma_emp_no_miss = tf.matmul(tf.matmul(D_no_miss, Corr_emp_no_miss), D_no_miss)
        Sigma_QIS = tf_QIS_batched(R_miss[:, :, -born:])

        fro_Sigma_pred_miss = frobenius_mean(Sigma_pred_miss, Sigma_true_miss)
        fro_Sigma_pred_no_miss = frobenius_mean(Sigma_pred_no_miss, Sigma_true_no_miss)
        fro_Sigma_emp_miss = frobenius_mean(Sigma_emp_miss, Sigma_true_miss)
        fro_Sigma_emp_no_miss = frobenius_mean(Sigma_emp_no_miss, Sigma_true_no_miss)
        fro_Sigma_QIS = frobenius_mean(Sigma_QIS, Sigma_true_miss)

        frob_cov_loss[step, :] = [
            fro_Sigma_pred_miss,
            fro_Sigma_pred_no_miss,
            fro_Sigma_emp_miss,
            fro_Sigma_emp_no_miss,
            fro_Sigma_QIS,
        ]

    return frob_corr_loss, frob_cov_loss
