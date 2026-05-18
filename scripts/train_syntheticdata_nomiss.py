"""
Train BiGRU spectral denoiser on synthetic data with NO missingness.
q sampled uniformly from [0.2, 3.0] — matches the real-data fine-tuning ceiling.
Saves weights to models/bigru_weights_nomiss.weights.h5.
"""

import sys

sys.path.insert(0, ".")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from data.dataloader import tf_data_generator
from models.gru_denoiser import BiGRUSpectralDenoiserTensorFlow
from training.trainer import Trainer_tf

# ── config ─────────────────────────────────────────────────────────────────
HIDDEN = 64
BATCH = 20
EPOCHS = 400
STEPS = 5
LR = 1e-4
PATIENCE = 25
N_MIN, N_MAX = 70, 150
Q_MIN, Q_MAX = 0.2, 3.0
MISSING = 1  # no missingness
SAVE_PATH = "models/bigru_weights_nomiss.weights.h5"
# ───────────────────────────────────────────────────────────────────────────

model = BiGRUSpectralDenoiserTensorFlow(hidden_size=HIDDEN)
_ = model(tf.random.normal((1, 100, 5)))  # build weights
print(f"Parameters: {model.count_params():,}")

trainer = Trainer_tf(
    model,
    tf_data_generator,
    batch_size=BATCH,
    epochs=EPOCHS,
    missing_constant=MISSING,
    N_min=N_MIN,
    N_max=N_MAX,
    q_min=Q_MIN,
    q_max=Q_MAX,
    lr=LR,
    patience=PATIENCE,
    val_steps=30,
)

losses = trainer.train(steps_per_epoch=STEPS)

model.save_weights(SAVE_PATH)
print(f"\nSaved → {SAVE_PATH}")

# training curve
window = 10
rolling = np.convolve(losses, np.ones(window) / window, mode="valid")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(losses, color="lightgrey", lw=0.8, label="raw")
ax.plot(
    np.arange(window - 1, len(losses)),
    rolling,
    color="steelblue",
    lw=2,
    label=f"{window}-epoch rolling mean",
)
ax.set_title("Training loss — no missingness, q ∈ [0.2, 3.0]")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("images/training_loss_nomiss.png", dpi=150)
print("Training curve saved → images/training_loss_nomiss.png")
