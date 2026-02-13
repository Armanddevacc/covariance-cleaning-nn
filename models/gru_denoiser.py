import torch.nn as nn
import torch


class BiGRUSpectralDenoiser(nn.Module):
    def __init__(self, hidden_size):
        # N : the number of eigenvalues per sample;
        # hidden_size : the dimension of the GRU’s hidden state
        super().__init__()

        self.gru = nn.GRU(
            input_size=6,  # The number of input features per sequence data, 1 here
            hidden_size=hidden_size,  # Number of features in the hidden state, one still one eigenvalue
            num_layers=1,  # amount of GRU we put one after-another
            bias=True,  # We always want bias
            batch_first=True,  # input and output tensors are provided as (batch, seq, feature)
            bidirectional=True,  #
        )
        self.fc = nn.Linear(
            2 * hidden_size, 1
        )  # we apply a Dense(2H to 1) at each time step
        self.activation = nn.Softplus()  # keeps positive eigenvalues

    def forward(self, x):
        h, _ = self.gru(x)  # (B, N, 2H)
        out = self.fc(h)
        out = out.squeeze(-1)
        out = self.activation(out)
        out = out / out.mean(dim=1, keepdim=True)
        return out  # if you reorder at the end of the NN we need to reorder the eigenvector too otherwise it makes no sense


"""
    def forward(self, input_seq, eps=1e-8):

        B, N, _ = input_seq.shape
        # lam_emp = input_seq[:, :, 0].detach()

        h, _ = self.gru(input_seq)  # (B, N, 2H)
        logg = self.fc(h).squeeze(-1)  # (B, N)

        # positive gaps
        g = self.activation(logg) + eps  # (B, N)

        # decreasing spectrum via reversed cumsum:
        # lam_i = sum_{k=i..N} g_k
        lam = torch.flip(
            torch.cumsum(torch.flip(g, dims=[1]), dim=1), dims=[1]
        )  # (B, N)

        # trace constraint
        lam = lam * (N / (lam.sum(dim=1, keepdim=True) + eps))

        return lam
"""

import tensorflow as tf


class BiGRUSpectralDenoiserTensorFlow(tf.keras.Model):
    def __init__(self, hidden_size):
        super().__init__()
        self.bigru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=hidden_size, return_sequences=True)
        )

        self.fc = tf.keras.layers.Dense(1)
        self.activation = tf.keras.layers.Activation("softplus")

    def call(self, x):
        # x shape: (B, N, 6)
        h = self.bigru(x)  # (B, N, 2H)
        out = self.fc(h)  # (B, N, 1)
        out = tf.squeeze(out, axis=-1)  # (B, N)
        out = self.activation(out)
        return out
