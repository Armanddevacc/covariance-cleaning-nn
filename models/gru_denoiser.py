import torch
import torch.nn as nn


class BiGRUSpectralDenoiser(nn.Module):
    def __init__(self, hidden_size):
        # N : the number of eigenvalues per sample;
        # hidden_size : the dimension of the GRU’s hidden state
        super().__init__()

        self.gru = nn.GRU(
            input_size=5,  # The number of input features per sequence data, 1 here
            hidden_size=hidden_size,  # Tumber of features in the hidden state, one still one eigenvalue
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
        return out
