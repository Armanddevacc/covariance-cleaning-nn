import tensorflow as tf


# fully vectorized covariance with pairwise-complete observations like pandas.cov()
def tf_cov_pairwise(X):
    """
    input X: (N, T) or (B, N, T) X has NaNs
    """
    X = tf.convert_to_tensor(X)
    original_rank = X.shape.rank  # can be None in graph, but in eager it's fine

    if original_rank == 2:
        X = tf.expand_dims(X, axis=0)  # becomes (1, N, T)

    X = tf.cast(X, tf.float64)  # match your generator precision

    # mask: 1 where valid, 0 where NaN
    mask = tf.math.is_finite(X)  # (B, N, T)
    mask_f = tf.cast(mask, tf.float64)

    # mean over time dimension (T), ignoring NaNs
    X0 = tf.where(mask, X, tf.zeros_like(X))  # replace NaN by 0 for sums
    cnt = tf.reduce_sum(mask_f, axis=-1, keepdims=True)  # (B, N, 1)
    means = tf.reduce_sum(X0, axis=-1, keepdims=True) / tf.maximum(cnt, 1.0)  # (B,N,1)

    # centered data (NaNs propagate in Xc, but we will zero them out next)
    Xc = X - means  # (B, N, T)

    # centered data but NaN replaced by 0
    Xc_zero = tf.where(mask, Xc, tf.zeros_like(Xc))

    # pairwise valid counts n_ij = sum_t mask[i,t] * mask[j,t]
    valid_counts = tf.matmul(mask_f, mask_f, transpose_b=True)  # (B, N, N)

    # numerator: sum of centered products
    numerator = tf.matmul(Xc_zero, Xc_zero, transpose_b=True)  # (B, N, N)

    valid = valid_counts > 1
    denom = tf.where(valid, valid_counts - 1, tf.ones_like(valid_counts))
    cov = tf.where(valid, numerator / denom, tf.zeros_like(numerator))

    # drop batch if original input had no batch
    if original_rank == 2:
        cov = cov[0]

    return cov


def tf_cov_pairwise_mask(X, mask):
    """
    input X : (B, N, T) X doesn't contain any NaNs
    mask : (B, N, T) mask is True when observed, False when missing
    """
    original_rank = X.shape.rank

    if original_rank == 2:
        X = tf.expand_dims(X, axis=0)
        mask = tf.expand_dims(mask, axis=0)

    X = tf.cast(X, tf.float64)
    mask_f = tf.cast(mask, tf.float64)

    # mean over time dimension (T), using input mask
    X0 = tf.where(mask, X, tf.zeros_like(X))
    cnt = tf.reduce_sum(mask_f, axis=-1, keepdims=True)         # (B, N, 1)
    means = tf.reduce_sum(X0, axis=-1, keepdims=True) / tf.maximum(cnt, 1.0)

    # centered data, masked positions zeroed
    Xc = X - means
    Xc_zero = tf.where(mask, Xc, tf.zeros_like(Xc))

    # pairwise valid counts n_ij = sum_t mask[i,t] * mask[j,t]
    valid_counts = tf.matmul(mask_f, mask_f, transpose_b=True)  # (B, N, N)

    # numerator: sum of centered products
    numerator = tf.matmul(Xc_zero, Xc_zero, transpose_b=True)  # (B, N, N)

    # denominator: n_ij - 1, zero out pairs with fewer than 2 joint observations
    valid = valid_counts > 1
    denom = tf.where(valid, valid_counts - 1, tf.ones_like(valid_counts))
    cov = tf.where(valid, numerator / denom, tf.zeros_like(numerator))

    if original_rank == 2:
        cov = cov[0]

    return cov
