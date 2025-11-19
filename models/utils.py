import numpy as np


def eigen_decomp(Sigma):
    vals, vecs = np.linalg.eig(Sigma)
    # sort desending
    # for Sigma in Sigmas
    idx = np.argsort(vals)[::-1]
    return vals[idx], vecs[:, idx]
