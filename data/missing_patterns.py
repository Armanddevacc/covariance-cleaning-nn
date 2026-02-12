import numpy as np
import torch


def make_monotone_pattern(R):
    n, T = R.shape

    t = np.concatenate(
        [np.random.default_rng().integers(low=T // 2.5, high=T + 1, size=n - 1), [T]]
    )
    t_vec = T - np.sort(t)[::-1]

    cols = np.arange(T)[None, :]

    mask = cols >= t_vec[:, None]

    R_mono = R.copy().astype(float)
    R_mono[~mask] = float("nan")

    return R_mono, t_vec, mask


def make_random_pattern_vecto(R, missing_constant):  # size (B, N, T)
    # vectorized version
    B, N, T = R.shape

    T_rand = torch.randint(
        low=T // missing_constant, high=T + 1, size=(B, N - 1)
    )  # (B, N-1)
    T_last = torch.full(
        (B, 1), T
    )  # (B, 1) garantee to have at least one full observation
    T_full = torch.cat([T_rand, T_last], dim=1)  # (B, N)

    # convert to t_vec
    t_sorted, _ = torch.sort(T_full, descending=True, dim=1)  # (B,N)
    t_vec = T - t_sorted  # (B,N)

    # 3. build mask
    cols = torch.arange(T).view(1, 1, T)  # (1,1,T)
    mask = cols >= t_vec.unsqueeze(-1)  # (B,N,T)

    # 4. apply mask
    R_mono = R.clone()
    R_mono[~mask] = float("nan")

    return R_mono, t_vec, mask


import tensorflow as tf


def tf_make_random_pattern_vecto(R, missing_constant):  # size (B, N, T)
    # vectorized version

    R = tf.convert_to_tensor(R)
    B = tf.shape(R)[0]
    N = tf.shape(R)[1]
    T = tf.shape(R)[2]

    # 1. draw random effective lengths
    T_rand = tf.random.uniform(
        shape=(B, N - 1),
        minval=T // missing_constant,
        maxval=T + 1,
        dtype=tf.int32,
    )  # (B, N-1)

    T_last = tf.fill((B, 1), T)  # (B, 1) garantee to have at least one full observation

    T_full = tf.concat([T_rand, T_last], axis=1)  # (B, N)

    # 2. convert to t_vec
    t_sorted = tf.sort(T_full, direction="DESCENDING", axis=1)  # (B, N)
    t_vec = T - t_sorted  # (B, N)

    # 3. build mask
    cols = tf.reshape(tf.range(T), (1, 1, -1))  # (1,1,T)
    mask = cols >= tf.expand_dims(t_vec, axis=-1)  # (B,N,T)

    # 4. apply mask
    R_mono = tf.identity(R)
    nan_tensor = tf.cast(float("nan"), R.dtype)

    R_mono = tf.where(mask, R_mono, nan_tensor)

    return R_mono, t_vec, mask
