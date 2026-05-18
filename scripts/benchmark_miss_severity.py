"""
scripts/benchmark_miss_severity.py — Step 2b: Asynchrony severity sweep.

Fixes N=80, q_eff=1.0 (T_in=80), varies miss_frac in [0, 0.1, ..., 0.7].
miss_frac = maximum fraction of history a stock can be missing
(each stock's entry time drawn uniform on [0, miss_frac]).

At miss_frac=0: fully synchronous  → NN should be near Pair+QIS
At miss_frac→0.7: severe asynchrony → QIS must truncate to a short sync window;
                                       NN exploits full pairwise history

This isolates the asynchrony advantage from the noise/q effects.

Run:   .venv/bin/python scripts/benchmark_miss_severity.py
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

from estimator.QIS         import QIS_batched_numpy
from estimator.pairwise_psd import pairwise_psd, is_psd, pairwise_corr_np
from estimator.poet         import poet_from_corr
from estimator.shaffer      import fit_monotone_regressions, reconstruct_mu_sigma_from_phi
from models.gru_denoiser    import BiGRUSpectralDenoiserTensorFlow
from results.comparaison    import construct_input_seq

os.makedirs("images", exist_ok=True)
np.random.seed(42)
tf.random.set_seed(42)

# ─── Config ──────────────────────────────────────────────────────────────────
DATA                = "data/vanilla_returns_top_3000_with_NaN_dtin_max_1200.joblib"
N                   = 50
Q_EFF               = 0.2          # T_in = 250 >> N+2 so QIS is always applicable
T_IN                = int(N / Q_EFF)   # = 250 days estimation window
T_OOS               = 20
T_PROXY             = 800
# At miss_frac=f: mean T_sync ≈ T_IN*(1 - f/2), effective q_qis = N/T_sync
# miss_frac=0.0 → T_sync=250, q_qis=0.20   (QIS excellent)
# miss_frac=0.5 → T_sync=187, q_qis=0.27   (QIS still good)
# miss_frac=0.8 → T_sync=150, q_qis=0.33   (QIS still OK)
# miss_frac=0.95→ T_sync=  6, q_qis=8.3    (QIS collapses)
MISS_FRACS          = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9]
N_STEPS             = 25
SMALL_CAP           = slice(0, 500)   # use large-caps so we can find sync windows easily
NN_HIDDEN           = 64
NN_WEIGHTS          = "models/bigru_weights_realdata.weights.h5"

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
    nn_model = BiGRUSpectralDenoiserTensorFlow(hidden_size=NN_HIDDEN)
    nn_model(tf.zeros((1, 100, 5)))   # build shape-agnostic; N handled dynamically
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
    N_, T_ = R_nan.shape
    obs = ~np.isnan(R_nan)
    t_first = np.argmax(obs, axis=1)
    order = np.argsort(t_first)
    R_s = R_nan[order]; t_s = t_first[order]
    try:
        phi = fit_monotone_regressions(R_s, t_s)
        _, Sigma = reconstruct_mu_sigma_from_phi(phi)
    except Exception:
        return pairwise_psd(R_nan)
    d = np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    C = Sigma / np.outer(d, d); np.fill_diagonal(C, 1.0)
    C = 0.5 * (C + C.T)
    inv = np.argsort(order)
    Cu = np.empty_like(C)
    Cu[np.ix_(inv, inv)] = C
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
        lam  = nn_model(inp, training=False)[0].numpy()
        lam  = np.maximum(lam, 1e-6)
        C_nn = Q_np @ np.diag(lam) @ Q_np.T
        np.fill_diagonal(C_nn, 1.0)
        return 0.5 * (C_nn + C_nn.T)
    except Exception:
        return None


# ─── Main sweep ──────────────────────────────────────────────────────────────

NAMES = ["Pair+QIS(sync)", "POET(pairwise)", "Stambaugh", "NN"]
n_est = len(NAMES)

lvar_results = np.full((len(MISS_FRACS), N_STEPS, n_est), np.nan)
psd_fail     = np.zeros((len(MISS_FRACS), N_STEPS), dtype=bool)
sync_T       = np.zeros((len(MISS_FRACS), N_STEPS))  # actual sync window length

rng = np.random.default_rng(7)
T_TOTAL = T_IN + T_OOS + T_PROXY

for mi, miss_frac in enumerate(MISS_FRACS):
    print(f"\nmiss_frac={miss_frac:.1f}  (max missing={miss_frac*100:.0f}% of T_in={T_IN})")
    step = 0; attempts = 0

    while step < N_STEPS and attempts < 3000:
        attempts += 1
        t0 = rng.integers(0, n_avail - T_TOTAL - 1)

        codes_t0 = avail_codes[t0, SMALL_CAP]
        ret_rows = avail_to_ret[t0: t0 + T_TOTAL]
        if (ret_rows < 0).any():
            continue

        R_cap = ret_mat[np.ix_(ret_rows, codes_t0)]
        sync  = ~np.isnan(R_cap).any(axis=0)
        s_idx = np.where(sync)[0]
        if len(s_idx) < N:
            continue

        chosen = rng.choice(s_idx, N, replace=False)
        R_full = R_cap[:, chosen].T   # (N, T_TOTAL)

        R_proxy = R_full[:, :T_PROXY]
        R_oos   = R_full[:, -T_OOS:]
        R_base  = R_full[:, T_PROXY: T_PROXY + T_IN]

        C_proxy = sample_corr(R_proxy)

        # Introduce monotone missingness at this miss_frac level
        R_nan    = R_base.copy().astype(np.float64)
        bad_mask = np.zeros_like(R_nan, dtype=bool)

        if miss_frac > 0:
            for i in range(N):
                tau = rng.uniform(0, miss_frac)
                t_s = int(tau * T_IN)
                if t_s > 0:
                    R_nan[i, :t_s]    = np.nan
                    bad_mask[i, :t_s] = True

        # PSD check
        C_pair = pairwise_corr_np(R_nan)
        C_pair = np.where(np.isfinite(C_pair), C_pair, 0.0)
        np.fill_diagonal(C_pair, 1.0)
        psd_fail[mi, step] = not is_psd(C_pair, tol=1e-8)

        # ── Pair+QIS (synchronous window only — degrades as T_sync shrinks) ──
        try:
            obs_all  = ~np.isnan(R_nan).any(axis=0)
            T_sync   = int(obs_all.sum())
            sync_T[mi, step] = T_sync
            if T_sync >= 2:   # always apply QIS; it handles T<N via delta0 formula
                R_sync  = R_nan[:, obs_all]
                C_qis_v = QIS_batched_numpy(R_sync[None])[0]
                d = np.sqrt(np.maximum(np.diag(C_qis_v), 1e-12))
                C_qis = C_qis_v / np.outer(d, d); np.fill_diagonal(C_qis, 1.0)
                C_qis = 0.5 * (C_qis + C_qis.T)
                np.fill_diagonal(C_qis, 1.0)
            else:
                C_qis = pairwise_psd(R_nan)
            lvar_results[mi, step, 0] = min_var_lvar(C_qis, R_oos)
        except Exception: pass

        # ── POET(pairwise) ──
        try:
            C_psd2  = pairwise_psd(R_nan)
            obs_cnt = int(np.median((~np.isnan(R_nan)).sum(axis=1)))
            C_poet  = poet_from_corr(C_psd2, T_eff=obs_cnt, K=3)
            lvar_results[mi, step, 1] = min_var_lvar(C_poet, R_oos)
        except Exception: pass

        # ── Stambaugh ──
        try:
            C_stam = stambaugh_corr(R_nan)
            lvar_results[mi, step, 2] = min_var_lvar(C_stam, R_oos)
        except Exception: pass

        # ── NN ──
        try:
            R_zero = np.where(bad_mask, 0.0, R_nan).astype(np.float32)
            C_nn   = run_nn(R_zero, bad_mask)
            if C_nn is not None:
                lvar_results[mi, step, 3] = min_var_lvar(C_nn, R_oos)
        except Exception: pass

        step += 1

    psd_r = psd_fail[mi, :step].mean() * 100
    t_s   = sync_T[mi, :step].mean()
    print(f"  done {step} steps | PSD fail {psd_r:.0f}% | mean sync T={t_s:.0f}/{T_IN}")

np.save("results/ablation_features/miss_severity_lvar.npy",   lvar_results)
np.save("results/ablation_features/miss_severity_psd.npy",    psd_fail)
np.save("results/ablation_features/miss_severity_sync_T.npy", sync_T)

# ─── Plot ─────────────────────────────────────────────────────────────────────
mf_arr = np.array(MISS_FRACS) * 100   # percent

COLORS = {
    "Pair+QIS(sync)": ("darkorange", "-"),
    "POET(pairwise)": ("#e15759",    "-"),
    "Stambaugh":      ("#59a14f",    "--"),
    "NN":             ("steelblue",  "-"),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for i, name in enumerate(NAMES):
    color, ls = COLORS[name]
    y = np.nanmean(lvar_results[:, :, i], axis=1)
    s = np.nanstd(lvar_results[:, :, i],  axis=1)
    valid = np.isfinite(y)
    ax1.plot(mf_arr[valid], y[valid], color=color, linestyle=ls,
             linewidth=2.5, marker="o", markersize=5, label=name)
    ax1.fill_between(mf_arr[valid], (y-s)[valid], (y+s)[valid], alpha=0.1, color=color)

ax1.set_xlabel("Max missing fraction (%)", fontsize=12)
ax1.set_ylabel("Realized portfolio variance", fontsize=12)
ax1.set_title(f"Portfolio variance vs asynchrony severity\n"
              f"(N={N}, q_eff={Q_EFF}, T_in={T_IN})", fontsize=11)
ax1.legend(fontsize=10); ax1.grid(alpha=0.2)

# NN gain over Pair+QIS as asynchrony grows
qis_idx = NAMES.index("Pair+QIS(sync)")
nn_idx  = NAMES.index("NN")
for i, name in enumerate(NAMES):
    if name == "NN":
        continue
    color, ls = COLORS[name]
    y_nn  = np.nanmean(lvar_results[:, :, nn_idx], axis=1)
    y_est = np.nanmean(lvar_results[:, :, i],      axis=1)
    gain  = (y_est - y_nn) / y_est * 100
    valid = np.isfinite(gain)
    ax2.plot(mf_arr[valid], gain[valid], color=color, linestyle=ls,
             linewidth=2.5, marker="o", markersize=5, label=f"vs {name}")

ax2.axhline(0, color="black", linewidth=1, linestyle="--")
ax2.set_xlabel("Max missing fraction (%)", fontsize=12)
ax2.set_ylabel("NN portfolio variance gain (%)", fontsize=12)
ax2.set_title("NN advantage grows with asynchrony severity", fontsize=11)
ax2.legend(fontsize=10); ax2.grid(alpha=0.2)

plt.suptitle(f"Asynchrony severity sweep — N={N}, q_eff={Q_EFF}, T_in={T_IN}", fontsize=13)
plt.tight_layout()
plt.savefig("images/benchmark_miss_severity.png", dpi=150)
plt.close()

# ─── Table ────────────────────────────────────────────────────────────────────
print("\n=== L_var by missingness fraction ===")
print(f"{'miss%':>6}  " + "  ".join(f"{n:>14}" for n in NAMES))
for mi, mf in enumerate(MISS_FRACS):
    row = np.nanmean(lvar_results[mi], axis=0)
    print(f"{mf*100:6.0f}%  " + "  ".join(f"{v:14.5f}" for v in row))

print("\nSaved → images/benchmark_miss_severity.png")
