"""
Train BiGRU spectral denoiser on synthetic data with monotone missingness.
missing_constant=2: each asset observed for at least T/2 timesteps.
q sampled uniformly from [0.3, 3.0].
Warm-starts from the no-missingness model, then fine-tunes with lower lr.
Loss: correlation Frobenius (equivalent to direct MSE on optimal shrinkage d*).
Saves weights to models/bigru_weights_syntheticdata.weights.h5.
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
EPOCHS = 300
STEPS = 5
LR = 3e-5  # fine-tuning lr (lower than scratch 1e-4 — warm-start from nomiss)
PATIENCE = 15
N_MIN, N_MAX = 70, 150
Q_MIN, Q_MAX = 0.3, 3.0
MISSING = 2  # monotone missingness: each asset observed ≥ T/2 steps
WARMSTART = "models/bigru_weights_nomiss.weights.h5"
SAVE_PATH = "models/bigru_weights_syntheticdata.weights.h5"
# ───────────────────────────────────────────────────────────────────────────

model = BiGRUSpectralDenoiserTensorFlow(hidden_size=HIDDEN)
_ = model(tf.random.normal((1, 100, 5)))  # build weights
model.load_weights(WARMSTART)
print(f"Parameters: {model.count_params():,}  — warm-started from {WARMSTART}")

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
    val_steps=20,
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
ax.set_title("Training loss — synthetic data with missingness, q ∈ [0.3, 3.0]")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("images/training_loss_syntheticdata.png", dpi=150)
print("Training curve saved → images/training_loss_syntheticdata.png")
