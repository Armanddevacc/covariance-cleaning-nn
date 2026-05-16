import tensorflow as tf


def tf_variance_loss(lam_pred, Q_emp, R_oos):
    """
    Minimize realized portfolio variance directly on raw OOS returns.
    Works for any T_oos / N ratio — no covariance matrix estimation needed.

    lam_pred: (B, N)    predicted eigenvalues of the correlation matrix
    Q_emp:    (B, N, N) empirical eigenvectors
    R_oos:    (B, N, T_oos) raw OOS returns (zeros at missing positions)
    """
    # Clamp eigenvalues to ensure PSD before solving for portfolio weights
    lam_safe = tf.maximum(lam_pred, 1e-4)
    Corr_pred = tf.matmul(
        tf.matmul(Q_emp, tf.linalg.diag(lam_safe)), Q_emp, transpose_b=True
    )  # (B, N, N)

    B = tf.shape(lam_pred)[0]
    N = tf.shape(lam_pred)[1]
    ones = tf.ones((B, N, 1), dtype=tf.float32)

    # MVP weights from the predicted correlation matrix
    x = tf.linalg.solve(Corr_pred, ones)             # (B, N, 1)
    w = x / tf.reduce_sum(x, axis=1, keepdims=True)  # (B, N, 1), sums to 1

    # Realized second moment: (1/T_oos) * sum_t (w^T r_t)^2
    # Using E[r^2] rather than Var(r) = E[r^2] - E[r]^2 for stability at small T_oos
    R_oos = tf.cast(R_oos, tf.float32)               # (B, N, T_oos)
    port_ret = tf.matmul(tf.transpose(R_oos, [0, 2, 1]), w)  # (B, T_oos, 1)
    variance = tf.reduce_mean(tf.square(port_ret), axis=1)    # (B, 1)

    return tf.cast(N, tf.float32) * tf.reduce_mean(variance)


# Potters–Bouchaud loss
def tf_loss_function_mat(Mat_ref, Mat_pred, T):
    B = tf.shape(Mat_ref)[0]
    N = tf.shape(Mat_ref)[1]

    # Matrix difference
    Delta = Mat_pred - Mat_ref  # (B, N, N)

    ## CB: It seems more efficient to compute the Frobenius norm via squaring element-wise and summing.
    # Square of the matrix (Delta^2 = Delta @ Delta) Symetric matrix so we don't need to transpose !
    Delta2 = tf.matmul(tf.transpose(Delta, perm=[0, 2, 1]), Delta)  # (B, N, N)

    # Trace of Delta2 = sum of diagonal
    trace_vals = tf.reduce_sum(tf.linalg.diag_part(Delta2), axis=1)  # (B,)

    # Make sure types match
    T = tf.cast(T, tf.float32)
    N = tf.cast(N, tf.float32)

    # Normalized Frobenius estimation error (Potters-Bouchaud)
    # loss_cov = tf.sqrt(trace_vals) * np.sqrt(T) / (N)  # (B,)

    # scaled frob -> used for QIS in LW 2015 and 2020
    loss_cov = tf.sqrt(trace_vals) / (N)  # (B,)

    return tf.reduce_mean(loss_cov)  # scalar
