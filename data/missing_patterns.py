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

    # 0. draw n biased toward large values
    u = tf.random.uniform((B,), 0.0, 1.0)
    n = tf.cast(tf.floor((u**3) * tf.cast(N, tf.float32)), tf.int32)
    # u^3 biases toward larger n

    # 1. draw random effective lengths
    T_rand = tf.random.uniform(
        shape=(B, N),
        minval=T // missing_constant,
        maxval=T + 1,
        dtype=tf.int32,
    )  # (B, N)

    # assets before n are fully observed
    asset_idx = tf.range(N)[None, :]

    # first asset always fully observed
    first_asset = asset_idx == 0

    # assets before n fully observed
    full_obs = tf.logical_or(first_asset, asset_idx < n[:, None])

    T_full = tf.where(full_obs, T, T_rand)

    # 2. convert to t_vec
    t_sorted = tf.sort(T_full, direction="DESCENDING", axis=1)  # (B, N)
    t_vec = T - t_sorted  # (B, N)

    # 3. build mask
    cols = tf.reshape(tf.range(T), (1, 1, T))
    mask = cols >= t_vec[..., None]

    # 4. apply mask
    nan_tensor = tf.cast(float("nan"), R.dtype)
    R_mono = tf.where(mask, R, nan_tensor)

    return R_mono, t_vec, mask
