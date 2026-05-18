"""
Regime 2 benchmark: pure high-q effect, zero missingness.

Large-cap pool (ranks 0–500), N=50 fixed, T_in swept from 60 to 500.
Both QIS and NN see identical fully-finite data.  Any NN advantage is
purely from better spectral estimation at high q — not from missingness.
"""

import sys, os
sys.path.insert(0, '.')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import tensorflow as tf

from estimator.QIS          import QIS_batched_numpy
from estimator.pairwise_psd import pairwise_psd, pairwise_corr_np
from estimator.poet         import poet_from_corr
from estimator.shaffer      import fit_monotone_regressions, reconstruct_mu_sigma_from_phi
from models.gru_denoiser    import BiGRUSpectralDenoiserTensorFlow

os.makedirs('results/ablation_features', exist_ok=True)
os.makedirs('images', exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
DATA = 'data/vanilla_returns_top_3000_with_NaN_dtin_max_1200.joblib'
print('Loading data...')
bundle   = joblib.load(DATA, mmap_mode='r')
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
n_avail      = avail_codes.shape[0]
print(f'  returns {ret_mat.shape}  avail {avail_codes.shape}')

# ── model ─────────────────────────────────────────────────────────────────────
model = BiGRUSpectralDenoiserTensorFlow(hidden_size=64)
model(tf.zeros((1, 80, 5)))
model.load_weights('models/bigru_weights_realdata.weights.h5')
print(f'Parameters: {model.count_params():,}')

# ── helpers ───────────────────────────────────────────────────────────────────
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
    obs     = ~np.isnan(R_nan)
    t_first = np.argmax(obs, axis=1)
    order   = np.argsort(t_first)
    try:
        phi = fit_monotone_regressions(R_nan[order], t_first[order])
        _, Sigma = reconstruct_mu_sigma_from_phi(phi)
    except Exception:
        return pairwise_psd(R_nan)
    d  = np.sqrt(np.maximum(np.diag(Sigma), 1e-12))
    C  = Sigma / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    C  = 0.5 * (C + C.T)
    inv = np.argsort(order)
    Cu  = np.empty_like(C)
    Cu[np.ix_(inv, inv)] = C
    np.fill_diagonal(Cu, 1.0)
    return Cu

def make_input_seq(R_nan, N_, T_):
    nan_mask  = np.isnan(R_nan)
    nan_mask[:, 1:] |= nan_mask[:, :-1]      # bad_next: match training pipeline
    observed  = ~nan_mask
    C_pair    = pairwise_corr_np(np.where(nan_mask, np.nan, R_nan))
    C_pair    = np.where(np.isfinite(C_pair), C_pair, 0.0)
    np.fill_diagonal(C_pair, 1.0)
    C_pair    = 0.5 * (C_pair + C_pair.T) + 1e-6 * np.eye(N_)
    eigvals, eigvecs = np.linalg.eigh(C_pair)
    lam       = eigvals.astype(np.float32)
    pos       = np.linspace(0.0, 1.0, N_, dtype=np.float32)
    has_any   = observed.any(axis=1)
    t_first   = np.argmax(observed, axis=1).astype(np.float32) / T_
    t_first   = np.where(has_any, t_first, 1.0)   # fully-absent stock → T (maximally noisy)
    Tminmean  = (eigvecs ** 2).T @ t_first
    eff_T     = np.maximum(1.0 - Tminmean, 1.0 / T_)
    q_eff     = (N_ / T_) / eff_T
    ipr       = (N_ * np.sum(eigvecs ** 4, axis=0)).astype(np.float32)
    q_s       = np.maximum(q_eff, 1e-6).astype(np.float32)
    z_mp      = (lam - (1.0 + np.sqrt(q_s)) ** 2) / np.sqrt(q_s)
    inp       = np.stack([lam, pos, q_eff.astype(np.float32), ipr, z_mp], axis=1)
    return tf.constant(inp[None], dtype=tf.float32), eigvecs

def run_nn(R_nan, N_, T_):
    try:
        inp, Q_e = make_input_seq(R_nan, N_, T_)
        lam = np.maximum(model(inp, training=False)[0].numpy(), 1e-6)
        C_nn = Q_e @ np.diag(lam) @ Q_e.T
        np.fill_diagonal(C_nn, 1.0)
        return 0.5 * (C_nn + C_nn.T)
    except Exception:
        return None

# ── sweep ─────────────────────────────────────────────────────────────────────
LARGE_CAP  = slice(0, 500)
N_LC       = 50
T_IN_LC    = [60, 80, 100, 150, 200, 300, 500]
T_OOS_LC   = 5
N_STEPS_LC = 80
LC_NAMES   = ['QIS(sync)', 'POET(pairwise)', 'Stambaugh', 'NN']
n_lc       = len(LC_NAMES)
lc_lvar    = np.full((len(T_IN_LC), N_STEPS_LC, n_lc), np.nan)
miss_lc    = np.zeros((len(T_IN_LC), N_STEPS_LC))
rng_lc     = np.random.default_rng(42)

for ti, T_in in enumerate(T_IN_LC):
    T_TOT = T_in + T_OOS_LC
    q_nom = N_LC / T_in
    print(f'\nT_in={T_in}  q={q_nom:.2f}')

    step = 0; attempts = 0
    while step < N_STEPS_LC and attempts < 5000:
        attempts += 1
        t_anchor = rng_lc.integers(T_in - 1, n_avail - T_OOS_LC - 1)
        t0       = t_anchor - T_in + 1
        codes_anchor = avail_codes[t_anchor, LARGE_CAP]
        ret_rows = avail_to_ret[t0: t0 + T_TOT]
        if (ret_rows < 0).any():
            continue
        R_cap     = ret_mat[np.ix_(ret_rows, codes_anchor)]
        R_in_raw  = R_cap[:T_in, :]
        R_oos_raw = R_cap[T_in:, :]
        in_ok     = np.isfinite(R_in_raw).all(axis=0)
        oos_ok    = np.isfinite(R_oos_raw).all(axis=0)
        both_ok   = in_ok & oos_ok
        if both_ok.sum() < N_LC:
            continue
        chosen = rng_lc.choice(np.where(both_ok)[0], N_LC, replace=False)
        R_nan  = R_in_raw[:, chosen].T.astype(np.float64)
        R_oos  = R_oos_raw[:, chosen].T.astype(np.float64)

        miss_lc[ti, step] = (~np.isfinite(R_nan)).mean()

        try:
            Sv  = QIS_batched_numpy(R_nan[None])[0]
            d   = np.sqrt(np.maximum(np.diag(Sv), 1e-12))
            C_q = Sv / np.outer(d, d)
            np.fill_diagonal(C_q, 1.0)
            lc_lvar[ti, step, 0] = min_var_lvar(0.5 * (C_q + C_q.T), R_oos)
        except Exception:
            pass

        try:
            C_psd  = pairwise_psd(R_nan)
            t_eff  = int(np.median(np.isfinite(R_nan).sum(axis=1)))
            C_poet = poet_from_corr(C_psd, T_eff=t_eff, K=3)
            lc_lvar[ti, step, 1] = min_var_lvar(C_poet, R_oos)
        except Exception:
            pass

        try:
            lc_lvar[ti, step, 2] = min_var_lvar(stambaugh_corr(R_nan), R_oos)
        except Exception:
            pass

        try:
            C_nn = run_nn(R_nan, N_LC, T_in)
            if C_nn is not None:
                lc_lvar[ti, step, 3] = min_var_lvar(C_nn, R_oos)
        except Exception:
            pass

        step += 1
        if step % 20 == 0:
            print(f'  {step}/{N_STEPS_LC}  miss={miss_lc[ti,:step].mean()*100:.1f}%', flush=True)

    print(f'  done {step} steps | miss={miss_lc[ti,:step].mean()*100:.2f}%')

np.save('results/ablation_features/highq_lvar.npy', lc_lvar)
np.save('results/ablation_features/highq_miss.npy', miss_lc)

# ── table ─────────────────────────────────────────────────────────────────────
lc_means = np.nanmean(lc_lvar, axis=1)
nn_lci   = LC_NAMES.index('NN')
qis_lci  = LC_NAMES.index('QIS(sync)')

print(f'\n{"T_in":>5}  {"q":>5}  {"miss%":>6}  ' + '  '.join(f'{n:>16}' for n in LC_NAMES) + '  NN gain vs QIS')
print('=' * 100)
for ti, T_in in enumerate(T_IN_LC):
    q    = N_LC / T_in
    row  = lc_means[ti]
    best = int(np.nanargmin(row))
    vals = '  '.join(f'{v:16.5f}{"*" if i == best else " "}' for i, v in enumerate(row))
    gain = (row[qis_lci] - row[nn_lci]) / row[qis_lci] * 100 if np.isfinite(row[qis_lci]) else float('nan')
    print(f'{T_in:5d}  {q:5.2f}  {miss_lc[ti].mean()*100:6.2f}%  {vals}  {gain:+.1f}%')

# ── plot ──────────────────────────────────────────────────────────────────────
qs_lc = np.array([N_LC / T_in for T_in in T_IN_LC])
LC_COLORS = {
    'QIS(sync)':      ('darkorange', '-'),
    'POET(pairwise)': ('#e15759',    '-'),
    'Stambaugh':      ('#59a14f',   '--'),
    'NN':             ('steelblue',  '-'),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for i, name in enumerate(LC_NAMES):
    color, ls = LC_COLORS[name]
    y  = np.nanmean(lc_lvar[:, :, i], axis=1)
    s  = np.nanstd(lc_lvar[:, :, i],  axis=1)
    ok = np.isfinite(y)
    ax1.plot(qs_lc[ok], y[ok], color=color, ls=ls, lw=2, marker='o', ms=4, label=name)
    ax1.fill_between(qs_lc[ok], (y - s)[ok], (y + s)[ok], alpha=0.08, color=color)

ax1.set_xlabel('q = N / T_in')
ax1.set_ylabel('Realized portfolio variance')
ax1.set_title(f'Regime 2: pure high-q, zero missingness\n(N={N_LC}, large-cap 0–500, no NaN/Inf)')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.2)

for i, name in enumerate(LC_NAMES):
    if name == 'NN':
        continue
    color, ls = LC_COLORS[name]
    y_nn  = np.nanmean(lc_lvar[:, :, nn_lci], axis=1)
    y_est = np.nanmean(lc_lvar[:, :, i],       axis=1)
    gain  = (y_est - y_nn) / y_est * 100
    ok    = np.isfinite(gain)
    ax2.plot(qs_lc[ok], gain[ok], color=color, ls=ls, lw=2, marker='o', ms=4, label=f'vs {name}')

ax2.axhline(0, color='black', lw=1, ls='--')
ax2.set_xlabel('q = N / T_in')
ax2.set_ylabel('NN portfolio variance gain (%)')
ax2.set_title('NN advantage at high q\n(identical data for all estimators — pure q effect)')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.2)

plt.suptitle(f'Pure Q-sweep (no missingness) — N={N_LC}, large-cap 0–500', fontsize=13)
plt.tight_layout()
plt.savefig('images/realdata_highq_clean.png', dpi=150)
print('\nSaved: images/realdata_highq_clean.png')
