"""
scripts/benchmark_nomiss.py — Step 1: No-missingness benchmark on synchronous real data.

Estimators: Sample | LW | QIS | POET | NN
Metric: L_var (realized min-variance portfolio variance on 20-day OOS window)
Truth proxy: sample covariance on 1000-day long window (for Frobenius context only)

Run:  .venv/bin/python scripts/benchmark_nomiss.py
"""

import sys, os
sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf

from estimator.QIS  import QIS_batched_numpy
from estimator.nls  import lw_corr
from estimator.poet import poet
from models.gru_denoiser import BiGRUSpectralDenoiserTensorFlow

os.makedirs("images", exist_ok=True)
np.random.seed(42)
tf.random.set_seed(42)

# ─── Config ──────────────────────────────────────────────────────────────────
DATA            = "data/vanilla_returns_top_3000_with_NaN_dtin_max_1200.joblib"
N               = 100
Q_VALUES        = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]
N_STEPS         = 30
T_LONG          = 1000   # long-horizon proxy
T_OOS           = 20
LARGE_CAP       = slice(0, 300)
NN_WEIGHTS      = "models/bigru_weights_realdata.weights.h5"

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data…")
bundle   = joblib.load(DATA, mmap_mode="r")
ret_df   = bundle.returns.copy()
avail_df = bundle.available_stocks.copy()
ret_df.index   = pd.to_datetime(ret_df.index)
avail_df.index = pd.to_datetime(avail_df.index)

is_num = pd.api.types.is_numeric_dtype(ret_df.columns)
codes, unique_ids = pd.factorize(avail_df.to_numpy().ravel())
if is_num:
    unique_ids = pd.to_numeric(unique_ids)
else:
    ret_df.columns = ret_df.columns.astype(str)
    unique_ids = unique_ids.astype(str)

avail_codes  = codes.reshape(avail_df.shape)
ret_mat      = ret_df.reindex(columns=unique_ids).values
avail_to_ret = ret_df.index.get_indexer(avail_df.index)
assert (avail_to_ret >= 0).all()
n_avail = avail_codes.shape[0]
print(f"  {ret_mat.shape}  avail {avail_codes.shape}")

# ─── Load NN ─────────────────────────────────────────────────────────────────
print("Loading NN…")
nn_model = None
if os.path.exists(NN_WEIGHTS):
    nn_model = BiGRUSpectralDenoiserTensorFlow(hidden_size=256)
    nn_model(tf.zeros((1, N, 6)))
    nn_model.load_weights(NN_WEIGHTS)
    print("  loaded (hidden=256, 6-feat, L_var-trained on real async data)")

# ─── Helpers ─────────────────────────────────────────────────────────────────

def sample_corr(R):
    Rc = R - R.mean(axis=1, keepdims=True)
    S  = Rc @ Rc.T / (R.shape[1] - 1)
    d  = np.sqrt(np.maximum(np.diag(S), 1e-12))
    C  = S / np.outer(d, d); np.fill_diagonal(C, 1.0)
    return 0.5 * (C + C.T)


def min_var_lvar(C, R_oos):
    N_ = C.shape[0]
    ones = np.ones((N_, 1))
    try:
        x = np.linalg.solve(C + 1e-6 * np.eye(N_), ones)
        w = x / x.sum()
    except np.linalg.LinAlgError:
        w = ones / N_
    return float(np.mean((R_oos.T @ w) ** 2))


def make_input_6feat(eigvals, eigvecs, q_global, T_len):
    """6-feature input for NN on synchronous data (no missingness → T̃_min=0, q_eff=q_global)."""
    N_  = eigvals.shape[0]
    lam = eigvals.astype(np.float32)
    pos = np.linspace(0, 1, N_, dtype=np.float32)
    q   = np.full(N_, float(q_global), dtype=np.float32)
    t_min = np.zeros(N_, dtype=np.float32)
    t_max = np.full(N_, (T_len - 1) / T_len, dtype=np.float32)
    inp = np.stack([lam, pos, q, t_min, t_max, q], axis=1)   # (N, 6)
    return tf.constant(inp[None], dtype=tf.float32)


# ─── Estimator names ─────────────────────────────────────────────────────────

NAMES = ["Sample", "LW", "QIS", "POET", "NN"]
n_est = len(NAMES)

lvar_results = np.full((len(Q_VALUES), N_STEPS, n_est), np.nan)

rng = np.random.default_rng(0)

for qi, q in enumerate(Q_VALUES):
    T_est = int(N / q)
    T_need = T_LONG + T_OOS
    print(f"\nq={q:.1f}  T_est={T_est}")

    step = 0; attempts = 0
    while step < N_STEPS and attempts < 2000:
        attempts += 1
        t0 = rng.integers(0, n_avail - T_need)

        codes_t0 = avail_codes[t0, LARGE_CAP]
        ret_rows = avail_to_ret[t0: t0 + T_need]
        window   = ret_mat[np.ix_(ret_rows, codes_t0)]
        valid    = ~np.isnan(window).any(axis=0)
        if valid.sum() < N:
            continue

        chosen = rng.choice(np.where(valid)[0], N, replace=False)
        R_full = window[:, chosen].T                         # (N, T_need)
        R_est  = R_full[:, T_LONG - T_est: T_LONG]
        R_oos  = R_full[:, T_LONG: T_LONG + T_OOS]

        # ── Sample ──
        C_sample = sample_corr(R_est)
        lvar_results[qi, step, 0] = min_var_lvar(C_sample, R_oos)

        # ── LW ──
        try:
            lvar_results[qi, step, 1] = min_var_lvar(lw_corr(R_est), R_oos)
        except Exception: pass

        # ── QIS ──
        try:
            Sv = QIS_batched_numpy(R_est[None])[0]
            d  = np.sqrt(np.maximum(np.diag(Sv), 1e-12))
            C_qis = Sv / np.outer(d, d); np.fill_diagonal(C_qis, 1.0)
            lvar_results[qi, step, 2] = min_var_lvar(C_qis, R_oos)
        except Exception: pass

        # ── POET ──
        try:
            C_poet, _ = poet(R_est)
            lvar_results[qi, step, 3] = min_var_lvar(C_poet, R_oos)
        except Exception: pass

        # ── NN ──
        if nn_model is not None:
            try:
                lam_e, Q_e = np.linalg.eigh(C_sample)
                inp      = make_input_6feat(lam_e, Q_e, N / (T_est - 1), T_est)
                lam_pred = nn_model(inp, training=False)[0].numpy()
                lam_pred = np.maximum(lam_pred, 1e-6)
                C_nn     = Q_e @ np.diag(lam_pred) @ Q_e.T
                np.fill_diagonal(C_nn, 1.0)
                lvar_results[qi, step, 4] = min_var_lvar(0.5*(C_nn+C_nn.T), R_oos)
            except Exception: pass

        step += 1
        if step % 10 == 0:
            print(f"  {step}/{N_STEPS}", flush=True)

np.save("results/ablation_features/nomiss_lvar.npy", lvar_results)

# ─── Plot ─────────────────────────────────────────────────────────────────────
qs = np.array(Q_VALUES)
COLORS = {"Sample":("dimgrey","-"), "LW":("#4e79a7","--"),
          "QIS":("darkorange","-"), "POET":("#e15759","-"), "NN":("steelblue","-")}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for i, name in enumerate(NAMES):
    color, ls = COLORS[name]
    y = np.nanmean(lvar_results[:, :, i], axis=1)
    s = np.nanstd(lvar_results[:, :, i], axis=1)
    valid = np.isfinite(y)
    ax1.plot(qs[valid], y[valid], color=color, linestyle=ls,
             linewidth=2, marker="o", markersize=4, label=name)
    ax1.fill_between(qs[valid], (y-s)[valid], (y+s)[valid], alpha=0.08, color=color)

ax1.axvline(1, color="grey", linewidth=1, linestyle=":", alpha=0.6)
ax1.set_xlabel("q = N / T", fontsize=12)
ax1.set_ylabel("Realized portfolio variance", fontsize=12)
ax1.set_title("Min-variance portfolio variance — synchronous large-cap")
ax1.legend(fontsize=10); ax1.grid(alpha=0.2)

# Gain over QIS
qis_i = NAMES.index("QIS")
for i, name in enumerate(NAMES):
    if name in ("QIS", "Sample"):
        continue
    color, ls = COLORS[name]
    y_nn  = np.nanmean(lvar_results[:, :, i],   axis=1)
    y_qis = np.nanmean(lvar_results[:, :, qis_i], axis=1)
    gain  = (y_qis - y_nn) / y_qis * 100
    valid = np.isfinite(gain)
    ax2.plot(qs[valid], gain[valid], color=color, linestyle=ls,
             linewidth=2, marker="o", markersize=4, label=name)

ax2.axhline(0, color="black", linewidth=1, linestyle="--")
ax2.axvline(1, color="grey", linewidth=1, linestyle=":", alpha=0.6)
ax2.set_xlabel("q = N / T", fontsize=12)
ax2.set_ylabel("L_var gain over QIS (%)", fontsize=12)
ax2.set_title("Improvement over QIS — synchronous large-cap")
ax2.legend(fontsize=10); ax2.grid(alpha=0.2)

plt.suptitle("No-missingness benchmark — synchronous real data (large-cap, N=100)", fontsize=13)
plt.tight_layout()
plt.savefig("images/benchmark_nomiss_qcurve.png", dpi=150)
plt.close()

# ─── Table ────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print(f"{'':>4}  " + "  ".join(f"{n:>8}" for n in NAMES))
print("="*55)
for qi, q in enumerate(Q_VALUES):
    row = np.nanmean(lvar_results[qi], axis=0)
    print(f"q={q:.1f}  " + "  ".join(f"{v:8.5f}" for v in row))
print("="*55)
print(f"\n{'Mean':>4}  " + "  ".join(f"{np.nanmean(lvar_results[:,:,i]):8.5f}"
                                       for i in range(n_est)))
