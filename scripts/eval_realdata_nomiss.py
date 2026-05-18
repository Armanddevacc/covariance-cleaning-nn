"""
Evaluate NN vs QIS vs Sample on real data, no-missingness setting.

Metric: realised portfolio variance on T_oos=5 OOS days.
Model:  models/bigru_weights_realdata_nomiss.weights.h5

Run:
    .venv/bin/python scripts/eval_realdata_nomiss.py
"""

import sys, os
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from scipy import stats

from data.real_dataloader import real_data_pipeline
from models.gru_denoiser import BiGRUSpectralDenoiserTensorFlow
from results.comparaison import construct_input_seq
from estimator.QIS import tf_QIS_batched
from models.losses import tf_variance_loss

# ─── Config ───────────────────────────────────────────────────────────────────
NN_WEIGHTS   = "models/bigru_weights_realdata_nomiss.weights.h5"
HIDDEN       = 64
BATCH        = 16
N_STOCKS     = 100
Q_VALUES     = [0.3, 0.5, 0.7, 0.95, 1.2, 1.5]
N_DAYS_OUT   = 5
SHIFT        = 1
STEPS        = 100   # per q value
MARKET_RANGE = (0, 500)
DATE_BOUNDS  = ("2005-01-01", "2020-01-01")

print(f"Sweep: N={N_STOCKS}, q∈{Q_VALUES}, T_oos={N_DAYS_OUT}, "
      f"{STEPS} steps × {BATCH} batch = {STEPS*BATCH:,} samples per q")

# ─── Load model ───────────────────────────────────────────────────────────────
model = BiGRUSpectralDenoiserTensorFlow(hidden_size=HIDDEN)
model(tf.zeros((1, N_STOCKS, 5)))
model.load_weights(NN_WEIGHTS)
print(f"Loaded {NN_WEIGHTS}")

# ─── Sweep over q values ──────────────────────────────────────────────────────
def run_q(q):
    N_DAYS_IN = int(N_STOCKS / q)
    dataset = real_data_pipeline(
        BATCH,
        date_bounds=DATE_BOUNDS,
        n_days_out=N_DAYS_OUT,
        n_days_in=N_DAYS_IN,
        n_stocks=N_STOCKS,
        shift=SHIFT,
        market_cap_range=MARKET_RANGE,
        sequential=False,
        return_generator=False,
        no_miss=True,
    )

    port_vars = np.zeros((STEPS, 3))  # [NN, QIS, Sample]

    for step, ((rin, mask), rout_nan) in enumerate(dataset.take(STEPS)):
        rout = tf.where(tf.math.is_nan(rout_nan), tf.zeros_like(rout_nan), rout_nan)
        rout = tf.cast(rout, tf.float32)

        input_seq, Q_emp, _, _ = construct_input_seq(rin, mask)
        input_seq = tf.cast(input_seq, tf.float32)
        Q_emp     = tf.cast(Q_emp,     tf.float32)
        lam_emp   = input_seq[:, :, 0]
        N         = Q_emp.shape[1]
        eye       = tf.eye(N, batch_shape=[BATCH], dtype=tf.float32)

        lam_pred = model(input_seq, training=False)
        port_vars[step, 0] = float(tf_variance_loss(lam_pred, Q_emp, rout))

        rin_f32   = tf.cast(rin, tf.float32)
        Sigma_qis = tf_QIS_batched(rin_f32)
        d = tf.linalg.diag_part(Sigma_qis)
        s = tf.sqrt(tf.maximum(d, 1e-8))
        C_qis = Sigma_qis / (s[:, :, None] * s[:, None, :])
        C_qis = 0.5 * (C_qis + tf.transpose(C_qis, [0, 2, 1])) + 1e-6 * eye
        lq, Qq = tf.linalg.eigh(C_qis)
        lq = tf.maximum(tf.cast(lq, tf.float32), 1e-4)
        port_vars[step, 1] = float(tf_variance_loss(lq, tf.cast(Qq, tf.float32), rout))

        port_vars[step, 2] = float(tf_variance_loss(lam_emp, Q_emp, rout))

    return port_vars

# ─── Results ──────────────────────────────────────────────────────────────────
all_results = {}
for q in Q_VALUES:
    T_in = int(N_STOCKS / q)
    print(f"\nq={q:.2f}  T_in={T_in}", flush=True)
    all_results[q] = run_q(q)
    m = all_results[q].mean(axis=0)
    gain = (m[1] - m[0]) / m[1] * 100
    t, p = stats.ttest_1samp(all_results[q][:, 0] - all_results[q][:, 1], 0)
    print(f"  NN={m[0]:.5f}  QIS={m[1]:.5f}  gain={gain:+.2f}%  p={p:.4f}")

print(f"\n{'='*65}")
print(f"{'q':>5}  {'T_in':>5}  {'NN':>10}  {'QIS':>10}  {'Sample':>10}  {'NN vs QIS':>10}")
print(f"{'-'*65}")
for q in Q_VALUES:
    T_in = int(N_STOCKS / q)
    m    = all_results[q].mean(axis=0)
    gain = (m[1] - m[0]) / m[1] * 100
    sig  = ""
    t, p = stats.ttest_1samp(all_results[q][:, 0] - all_results[q][:, 1], 0)
    if p < 0.01:  sig = "**"
    elif p < 0.05: sig = "*"
    print(f"{q:>5.2f}  {T_in:>5}  {m[0]:>10.5f}  {m[1]:>10.5f}  {m[2]:>10.5f}  {gain:>+8.2f}%{sig}")
print(f"{'='*65}")
print("** p<0.01  * p<0.05")
