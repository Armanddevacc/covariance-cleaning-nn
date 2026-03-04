import torch
import tensorflow as tf


def reconstruct(Q, lam):
    Lambda = torch.diag_embed(lam.squeeze(-1))
    Sigma = Q @ Lambda @ Q.transpose(1, 2)
    return Sigma


def frobenius_mean(A, B, T=None):
    A = tf.cast(A, tf.float32)
    B = tf.cast(B, tf.float32)
    diff = A - B
    frob_per_batch = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=[1, 2]))
    return tf.reduce_mean(frob_per_batch)


def mse(A, B):
    A = tf.cast(A, tf.float32)
    B = tf.cast(B, tf.float32)
    return tf.reduce_mean(tf.square(A - B))
