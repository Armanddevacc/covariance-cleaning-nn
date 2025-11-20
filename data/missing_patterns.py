import numpy as np


def make_monotone_pattern(R):
    n, T = R.shape

    t = np.concatenate(
        [np.random.default_rng().integers(low=T // 2.5, high=T + 1, size=n - 1), [T]]
    )
    t_vec = T - np.sort(t)[::-1]

    cols = np.arange(T)[None, :]

    mask = cols >= t_vec[:, None]

    R_mono = R.copy().astype(float)
    R_mono[~mask] = -5.0

    return R_mono, t_vec, mask
