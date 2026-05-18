"""
Train BiGRU on real asynchronous data (with missingness).

Loss: realized portfolio variance (Bongiorno et al.):
    L = N * w^T Sigma_out w
where w are the predicted MVP weights and Sigma_out is the realized OOS
second-moment matrix over T_OOS=5 days.  No oracle term — Sigma_out is
only used as a quadratic form (never inverted), so rank-deficiency is irrelevant.

Warm-starts from models/bigru_weights_syntheticdata.weights.h5 so the model retains
its synthetic prior (eigenvalue shrinkage toward the MP bulk) and only adapts to
the real-data distribution.

N varies per batch (40–120), q drawn uniformly from (0.05, 3.0),
T_in = clip(N/q, 25, 800).  Covers the full benchmark range q ∈ [0.05, 3.0]
with a flat q distribution matching the no-miss training setup.
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

HIDDEN = 64  # matches bigru_weights_syntheticdata (warm-start)
BATCH = 64
N_STOCKS_RANGE = (40, 120)   # covers both benchmark sizes (N=50 and N=80)
Q_RANGE        = (0.05, 3.0) # q drawn uniformly; same range as no-miss
N_DAYS_IN_RANGE = (25, 800)  # lookback buffer: T = clip(N/q, 25, 800)
N_DAYS_OUT = 5
SHIFT = 1
EPOCHS = 300
STEPS = 20
LR = 3e-5
PATIENCE = None
MARKET_RANGE = (2500, 3000)  # micro-cap: richest staircase NaN patterns
WARMSTART = "models/bigru_weights_syntheticdata.weights.h5"
SAVE_PATH = "models/bigru_weights_realdata.weights.h5"

pipeline_kwargs = dict(
    n_days_out=N_DAYS_OUT,
    n_days_in_range=N_DAYS_IN_RANGE,
    q_range=Q_RANGE,
    shift=SHIFT,
    n_stocks_range=N_STOCKS_RANGE,
    market_cap_range=MARKET_RANGE,
    sequential=False,
    return_generator=False,
)

print(
    f"Miss real: hidden={HIDDEN}, N∈{N_STOCKS_RANGE}, q~U{Q_RANGE}, "
    f"T_in=clip(N/q,{N_DAYS_IN_RANGE[0]},{N_DAYS_IN_RANGE[1]}), "
    f"T_OOS={N_DAYS_OUT}d, warmstart={WARMSTART}"
)

if os.path.exists(SAVE_PATH):
    print(f"Skipping training — {SAVE_PATH} already exists. Delete it to retrain.")
    raise SystemExit(0)

dataset = real_data_pipeline(
    BATCH, date_bounds=("2005-01-01", "2020-01-01"), **pipeline_kwargs
)

model = BiGRUSpectralDenoiserTensorFlow(hidden_size=HIDDEN)
model(tf.random.normal((1, 80, 5)))  # build weights with representative N
model.load_weights(WARMSTART)
print(f"Parameters: {model.count_params():,}  — warm-started from {WARMSTART}")

trainer = Trainer_real_data_tf(
    model,
    dataset,
    BATCH,
    epochs=EPOCHS,
    lr=LR,
    patience=PATIENCE,
    no_miss=False,
)
trainer.train(STEPS)

model.save_weights(SAVE_PATH)
print(f"Saved → {SAVE_PATH}")
