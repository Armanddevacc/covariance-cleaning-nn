"""
Train BiGRU on real data, synchronous setting (no missingness).

Initialization (Kim/Tae/Lee 2025 spirit): warm-start from bigru_weights_nomiss.weights.h5,
a model pre-trained on synthetic data with Frobenius loss against the true covariance.
This is the no-miss analogue of Kim/Tae/Lee's initialization to the sample covariance:
the model already produces sensible eigenvalue estimates before the decision-focused
fine-tuning begins.

Loss: decision-focused regret loss (Kim, Tae, Lee 2025):
    L = N * (V_hat - V_opt)
with a 100-day OOS window (q_oos ∈ [0.47, 1.0] for N ∈ [70, 100]) — reliable oracle.

N varies per batch (70–150), q drawn uniformly from (0.05, 3.0),
T_in = clip(N/q, 25, 400).  Covers the full benchmark range q ∈ [0.3, 3.0]
with a flat q distribution matching the miss training setup.

no_miss=True: mask is zeroed — all stocks treated as fully observed.
Saves: models/bigru_weights_realdata_nomiss.weights.h5
"""

import sys, os, warnings

sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

import tensorflow as tf

from data.real_dataloader import real_data_pipeline
from models.gru_denoiser import BiGRUSpectralDenoiserTensorFlow
from training.trainer import Trainer_real_data_tf

HIDDEN         = 64
BATCH          = 16
N_STOCKS_RANGE = (70, 150)   # matches synthetic no-miss training range
Q_RANGE        = (0.05, 3.0) # q drawn uniformly; same range as miss
N_DAYS_IN_RANGE = (25, 400)  # lookback buffer: T = clip(N/q, 25, 400)
N_DAYS_OUT     = 5
SHIFT          = 1
EPOCHS         = 300
STEPS          = 20
LR             = 3e-5
PATIENCE       = None
MARKET_RANGE   = (0, 500)    # large-cap pool
WARMSTART      = "models/bigru_weights_nomiss.weights.h5"
SAVE_PATH      = "models/bigru_weights_realdata_nomiss.weights.h5"

pipeline_kwargs = dict(
    n_days_out=N_DAYS_OUT,
    n_days_in_range=N_DAYS_IN_RANGE,
    q_range=Q_RANGE,
    shift=SHIFT,
    n_stocks_range=N_STOCKS_RANGE,
    market_cap_range=MARKET_RANGE,
    sequential=False,
    return_generator=False,
    no_miss=True,
)

print(f"No-miss real: hidden={HIDDEN}, N∈{N_STOCKS_RANGE}, q~U{Q_RANGE}, "
      f"T_in=clip(N/q,{N_DAYS_IN_RANGE[0]},{N_DAYS_IN_RANGE[1]}), "
      f"T_OOS={N_DAYS_OUT}d, warmstart={WARMSTART}")

if os.path.exists(SAVE_PATH):
    print(f"Skipping training — {SAVE_PATH} already exists. Delete it to retrain.")
    raise SystemExit(0)

dataset = real_data_pipeline(
    BATCH, date_bounds=("2005-01-01", "2020-01-01"), **pipeline_kwargs
)

model = BiGRUSpectralDenoiserTensorFlow(hidden_size=HIDDEN)
model(tf.random.normal((1, 100, 5)))  # build weights
model.load_weights(WARMSTART)
print(f"Parameters: {model.count_params():,}  — warm-started from {WARMSTART}")

trainer = Trainer_real_data_tf(
    model,
    dataset,
    BATCH,
    epochs=EPOCHS,
    lr=LR,
    patience=PATIENCE,
    no_miss=True,
)
trainer.train(STEPS)

model.save_weights(SAVE_PATH)
print(f"Saved → {SAVE_PATH}")
