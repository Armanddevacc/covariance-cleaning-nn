import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.dataloader import data_generator
from models.losses import loss_function_mat
from models.gru_denoiser import BiGRUSpectralDenoiser
from training.trainer import Trainer


mse = nn.MSELoss()
N_min = 70
N_max = 250
T_min = 30
T_max = 70
batch_size = 100
### Training loop — Generated Data
model_generated_data = BiGRUSpectralDenoiser(hidden_size=96)  # hidden_size to be tunned
# lr, weight_decay, batch_size, epochs, hidden_size to be tunned
trainer = Trainer(
    model=model_generated_data,
    is_train_on_real_data=False,
    loss_function=loss_function_mat,
    data_generator=data_generator,
    lr=1e-4,
    weight_decay=1e-5,
    batch_size=batch_size,
    epochs=50,
    N_min=N_min,
    N_max=N_max,
    T_min=T_min,
    T_max=T_max,
    log_interval=10,
    accumulate_steps=2,
    dataset=None,
    missing_constant=2,
)
# need to epochs to be big when there is lot of choise for N and T
# need for big batch_size when df is big
losses = trainer.train()
