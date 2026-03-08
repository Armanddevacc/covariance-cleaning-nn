import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.dataloader import tf_data_generator
from models.gru_denoiser import BiGRUSpectralDenoiserTensorFlow
from training.trainer import Trainer_tf


from data.real_dataloader import real_data_pipeline

T = 600
N = 50
batch_size = 32
dataset = real_data_pipeline(
    batch_size=32,
    date_bounds=("1995-01-01", "2015-01-01"),
    n_days_out=10,
    n_days_in=T,
    n_stocks=N,
    market_cap_range=(2500, 3000),
    shift=0,
)
(rin, mask), rout = next(iter(dataset))


mse = nn.MSELoss()
N_min = 70
N_max = 250
T_min = 30
T_max = 70
batch_size = 100
### Training loop — Generated Data
model_generated_data = BiGRUSpectralDenoiserTensorFlow(
    hidden_size=96
)  # hidden_size to be tunned
# lr, weight_decay, batch_size, epochs, hidden_size to be tunned
trainer = Trainer_tf(
    model_generated_data,
    tf_data_generator,
    batch_size,
    epochs=50,
    missing_constant=2,
    N_min=N_min,
    N_max=N_max,
    T_min=T_min,
    T_max=T_max,
)
losses = trainer.train(50)
