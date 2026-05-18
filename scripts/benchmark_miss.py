"""
scripts/benchmark_miss.py — Step 2: Missingness benchmark on real asynchronous data.

Estimators:
  Pairwise MLE      — raw baseline; shows PSD failure rate
  QIS(sync window)  — state-of-the-art applied to the shortest synchronous subwindow
  POET(pairwise)    — best classical method from Step 1, applied after PSD repair
  Stambaugh         — classical monotone MLE (Stambaugh 1997)
  NN                — our model; PSD + spectral cleaning + full asynchronous history

Metric: L_var (realized min-variance portfolio variance, 20-day OOS)

Run:  .venv/bin/python scripts/benchmark_miss.py
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

from estimator.QIS          import QIS_batched_numpy
from estimator.pairwise_psd import pairwise_psd, is_psd, pairwise_corr_np
from estimator.poet         import poet_from_corr
from estimator.shaffer      import fit_monotone_regressions, reconstruct_mu_sigma_from_phi
from models.gru_denoiser    import BiGRUSpectralDenoiserTensorFlow
from results.comparaison    import construct_input_seq

os.makedirs("images", exist_ok=True)
np.random.seed(42)
tf.random.set_seed(42)

# ─── Config ──────────────────────────────────────────────────────────────────
DATA        = "data/vanilla_returns_top_3000_with_NaN_dtin_max_1200.joblib"
N           = 80
Q_VALUES    = [0.5, 0.7, 1.0, 1.2, 1.5]
N_STEPS     = 20
T_OOS       = 20
T_PROXY     = 800
SMALL_CAP   = slice(500, 2000)
NN_WEIGHTS  = "models/bigru_weights_realdata.weights.h5"

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

# ─── Load NN ─────────────────────────────────────────────────────────────────
print("Loading NN…")
nn_model = None
if os.path.exists(NN_WEIGHTS):
    nn_model = BiGRUSpectralDenoiserTensorFlow(hidden_size=256)
    nn_model(tf.zeros((1, N, 5)))
    nn_model.load_weights(NN_WEIGHTS)
    print("  loaded")

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


def stambaugh_corr(R_nan):
    N_, _ = R_nan.shape
    obs     = ~np.isnan(R_nan)
    t_first = np.argmax(obs, axis=1)
    order   = np.argsort(t_first)
    try:
        phi = fit_monotone_regressions(R_nan[order], t_first[order])
        _, Sigma = reconstruct_mu_sigma_from_phi(phi)
    except Exception:
        return pairwise_psd(R_nan)
    d = np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    C = Sigma / np.outer(d, d); np.fill_diagonal(C, 1.0)
    C = 0.5 * (C + C.T)
    inv = np.argsort(order)
    Cu = np.empty_like(C); Cu[np.ix_(inv, inv)] = C
    np.fill_diagonal(Cu, 1.0)
    return Cu


def run_nn(R_zero, bad_mask):
    if nn_model is None:
        return None
    rin  = tf.constant(R_zero[None], dtype=tf.float32)
    mask = tf.constant(bad_mask[None], dtype=tf.bool)
    try:
        inp, Q_emp, _, _ = construct_input_seq(rin, mask)
        Q_np = tf.cast(Q_emp, tf.float32)[0].numpy()
        lam  = np.maximum(nn_model(inp, training=False)[0].numpy(), 1e-6)
        C_nn = Q_np @ np.diag(lam) @ Q_np.T
        np.fill_diagonal(C_nn, 1.0)
        return 0.5 * (C_nn + C_nn.T)
    except Exception:
        return None


# ─── Main loop ────────────────────────────────────────────────────────────────

NAMES = ["Pairwise_MLE", "QIS(sync)", "POET(pairwise)", "Stambaugh", "NN"]
n_est = len(NAMES)

lvar_results = np.full((len(Q_VALUES), N_STEPS, n_est), np.nan)
psd_failures = np.zeros((len(Q_VALUES), N_STEPS), dtype=bool)

rng = np.random.default_rng(1)

for qi, q in enumerate(Q_VALUES):
    T_in  = int(N / q)
    T_TOT = T_in + T_OOS + T_PROXY
    print(f"\nq={q:.1f}  T_in={T_in}  N={N}")

    step = 0; attempts = 0
    while step < N_STEPS and attempts < 3000:
        attempts += 1
        t0 = rng.integers(0, n_avail - T_TOT - 1)

        codes_t0 = avail_codes[t0, SMALL_CAP]
        ret_rows = avail_to_ret[t0: t0 + T_TOT]
        if (ret_rows < 0).any():
            continue

        R_cap = ret_mat[np.ix_(ret_rows, codes_t0)]
        sync  = ~np.isnan(R_cap).any(axis=0)
        if sync.sum() < N:
            continue

        chosen  = rng.choice(np.where(sync)[0], N, replace=False)
        R_full  = R_cap[:, chosen].T          # (N, T_TOT)
        R_proxy = R_full[:, :T_PROXY]
        R_oos   = R_full[:, -T_OOS:]
        R_base  = R_full[:, T_PROXY: T_PROXY + T_in].copy().astype(np.float64)

        # Monotone missingness: each stock i missing first τ_i fraction of window
        bad_mask = np.zeros_like(R_base, dtype=bool)
        for i in range(N):
            tau = rng.uniform(0, 0.5)
            t_s = int(tau * T_in)
            if t_s > 0:
                R_base[i, :t_s] = np.nan
                bad_mask[i, :t_s] = True

        # PSD check on raw pairwise
        C_pair = pairwise_corr_np(R_base)
        C_pair = np.where(np.isfinite(C_pair), C_pair, 0.0)
        np.fill_diagonal(C_pair, 1.0)
        psd_failures[qi, step] = not is_psd(C_pair, tol=1e-8)

        # ── Pairwise MLE (raw, may be non-PSD) ──
        lvar_results[qi, step, 0] = min_var_lvar(C_pair + 1e-4*np.eye(N), R_oos)

        # ── QIS on synchronous window ──
        try:
            obs_all = ~np.isnan(R_base).any(axis=0)
            T_sync  = int(obs_all.sum())
            if T_sync >= 2:
                Sv  = QIS_batched_numpy(R_base[:, obs_all][None])[0]
                d   = np.sqrt(np.maximum(np.diag(Sv), 1e-12))
                C_q = Sv / np.outer(d, d); np.fill_diagonal(C_q, 1.0)
                lvar_results[qi, step, 1] = min_var_lvar(C_q, R_oos)
        except Exception: pass

        # ── POET on pairwise+PSD ──
        try:
            C_psd  = pairwise_psd(R_base)
            t_eff  = int(np.median((~np.isnan(R_base)).sum(axis=1)))
            C_poet = poet_from_corr(C_psd, T_eff=t_eff, K=3)
            lvar_results[qi, step, 2] = min_var_lvar(C_poet, R_oos)
        except Exception: pass

        # ── Stambaugh ──
        try:
            lvar_results[qi, step, 3] = min_var_lvar(stambaugh_corr(R_base), R_oos)
        except Exception: pass

        # ── NN ──
        try:
            R_zero = np.where(bad_mask, 0.0, R_base).astype(np.float32)
            C_nn   = run_nn(R_zero, bad_mask)
            if C_nn is not None:
                lvar_results[qi, step, 4] = min_var_lvar(C_nn, R_oos)
        except Exception: pass

        step += 1
        if step % 5 == 0:
            print(f"  {step}/{N_STEPS}  PSD fail {psd_failures[qi,:step].mean()*100:.0f}%",
                  flush=True)

np.save("results/ablation_features/miss_lvar.npy", lvar_results)
np.save("results/ablation_features/miss_psd_failures.npy", psd_failures)

# ─── Plot ─────────────────────────────────────────────────────────────────────
qs = np.array(Q_VALUES)
COLORS = {"Pairwise_MLE":("dimgrey","-"), "QIS(sync)":("darkorange","-"),
          "POET(pairwise)":("#e15759","-"), "Stambaugh":("#59a14f","--"),
          "NN":("steelblue","-")}

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
ax1.set_xlabel("q_eff = N / T_in", fontsize=12)
ax1.set_ylabel("Realized portfolio variance", fontsize=12)
ax1.set_title("Min-variance portfolio variance — asynchronous real data")
ax1.legend(fontsize=10); ax1.grid(alpha=0.2)

# NN gain over QIS(sync)
qis_i = NAMES.index("QIS(sync)")
nn_i  = NAMES.index("NN")
for i, name in enumerate(NAMES):
    if name in ("NN", "Pairwise_MLE"):
        continue
    color, ls = COLORS[name]
    y_nn  = np.nanmean(lvar_results[:, :, nn_i], axis=1)
    y_est = np.nanmean(lvar_results[:, :, i],    axis=1)
    gain  = (y_est - y_nn) / y_est * 100
    valid = np.isfinite(gain)
    ax2.plot(qs[valid], gain[valid], color=color, linestyle=ls,
             linewidth=2, marker="o", markersize=4, label=f"vs {name}")

ax2.axhline(0, color="black", linewidth=1, linestyle="--")
ax2.axvline(1, color="grey", linewidth=1, linestyle=":", alpha=0.6)
ax2.set_xlabel("q_eff = N / T_in", fontsize=12)
ax2.set_ylabel("NN L_var gain (%)", fontsize=12)
ax2.set_title("NN advantage over baselines")
ax2.legend(fontsize=10); ax2.grid(alpha=0.2)

plt.suptitle("Missingness benchmark — asynchronous real data (N=80, miss_frac≤50%)", fontsize=13)
plt.tight_layout()
plt.savefig("images/benchmark_miss_qcurve.png", dpi=150)
plt.close()

# ─── Table ────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'q':>4}  PSD_fail  " + "  ".join(f"{n[:10]:>10}" for n in NAMES))
print("="*65)
for qi, q in enumerate(Q_VALUES):
    row  = np.nanmean(lvar_results[qi], axis=0)
    psd  = psd_failures[qi].mean() * 100
    print(f"q={q:.1f}  {psd:6.0f}%    " + "  ".join(f"{v:10.5f}" for v in row))
print("="*65)
