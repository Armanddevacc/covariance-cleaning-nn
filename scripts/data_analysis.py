"""
scripts/data_analysis.py — Step 0: Synthetic IW vs real small-cap comparison.

Four panels saved to images/step0_*.png:
  A. Spectral density at matched (N, q): IW synthetic vs real vs Marchenko-Pastur law
  B. Top-K eigenvalue stability across rolling windows
  C. Return tail statistics (per-asset skew and excess kurtosis)
  D. Missingness pattern: first-observation time distribution

Run from repo root:
    .venv/bin/python scripts/data_analysis.py
"""

import sys, os
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

os.makedirs("images", exist_ok=True)
np.random.seed(42)

# ─── Config ──────────────────────────────────────────────────────────────────
DATA        = "data/vanilla_returns_top_3000_with_NaN_dtin_max_1200.joblib"
N           = 100           # number of stocks for spectral analysis
Q_VALUES    = [0.5, 1.0, 1.5]
N_IW        = 300           # IW draws per (N, q) for histogram averaging
TOP_K       = 5             # eigenvalues tracked for rolling stability
ROLL_T      = 200           # window length for rolling analysis
ROLL_STEP   = 20            # step between windows
LARGE_CAP   = slice(0, 200) # stock rank range (small index = large cap)
SMALL_CAP   = slice(1000, 3000)  # Russell-2000-tier stocks

# ─── Load data ────────────────────────────────────────────────────────────────
print("Loading data…")
bundle   = joblib.load(DATA, mmap_mode="r")
ret_df   = bundle.returns.copy()
avail_df = bundle.available_stocks.copy()
ret_df.index   = pd.to_datetime(ret_df.index)
avail_df.index = pd.to_datetime(avail_df.index)

# Factorize available_stocks → integer codes pointing into ret_matrix columns
is_num = pd.api.types.is_numeric_dtype(ret_df.columns)
codes, unique_ids = pd.factorize(avail_df.to_numpy().ravel())
if is_num:
    unique_ids = pd.to_numeric(unique_ids)
else:
    ret_df.columns = ret_df.columns.astype(str)
    unique_ids = unique_ids.astype(str)

avail_codes = codes.reshape(avail_df.shape)        # (n_avail, 3000) int codes
ret_mat     = ret_df.reindex(columns=unique_ids).values  # (T_full, n_unique) float64

# Align avail dates → ret_mat row indices
avail_dates = avail_df.index
ret_dates   = ret_df.index
avail_to_ret = ret_dates.get_indexer(avail_dates)  # (n_avail,) int, -1 if not found
assert (avail_to_ret >= 0).all(), "Date alignment failed"

print(f"Returns:  {ret_mat.shape}  {ret_dates[0].date()} — {ret_dates[-1].date()}")
print(f"Avail:    {avail_codes.shape}  {avail_dates[0].date()} — {avail_dates[-1].date()}")

# ─── Helpers ─────────────────────────────────────────────────────────────────

def mp_density(q, n_pts=800):
    """Marchenko-Pastur continuous density for concentration q = N/T."""
    lp = (1 + np.sqrt(q))**2
    lm = (1 - np.sqrt(q))**2 if q <= 1 else 0.0
    x  = np.linspace(max(1e-4, lm * 0.5), lp * 1.3, n_pts)
    inside = (x >= lm) & (x <= lp)
    pdf = np.zeros_like(x)
    pdf[inside] = np.sqrt((lp - x[inside]) * (x[inside] - lm)) / (2 * np.pi * q * x[inside])
    return x, pdf


def iw_eigenvalues(N, q, n_samples, df_min_f=1.5, df_max_f=3.0):
    """Pool eigenvalues of correlation matrices from IW draws."""
    T  = int(N / q)
    all_lam = []
    for _ in range(n_samples):
        df = np.random.randint(int(df_min_f * (N + 2)), int(df_max_f * N))
        Z  = np.random.randn(df, N)
        Sigma = np.linalg.solve(Z.T @ Z, np.eye(N)) * (df - N - 1)
        L  = np.linalg.cholesky(Sigma)
        R  = L @ np.random.randn(N, T)
        S  = np.cov(R)
        d  = np.sqrt(np.maximum(np.diag(S), 1e-12))
        C  = S / np.outer(d, d)
        np.fill_diagonal(C, 1.0)
        all_lam.append(np.linalg.eigvalsh(0.5 * (C + C.T)))
    return np.concatenate(all_lam)


def find_sync_windows(avail_codes, avail_to_ret, ret_mat, T_win, cap_slice,
                      n_win, n_stocks, rng=None):
    """
    Find n_win contiguous windows of T_win days with n_stocks synchronous stocks
    from the given market-cap range.  Returns list of (N, T_win) arrays.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n_avail = avail_codes.shape[0]
    windows  = []
    attempts = 0
    while len(windows) < n_win and attempts < 5000:
        attempts += 1
        t0 = rng.integers(0, n_avail - T_win)
        codes_t0 = avail_codes[t0, cap_slice]         # candidate stock codes
        # Check NaN across the window
        ret_rows  = avail_to_ret[t0: t0 + T_win]      # (T_win,) row indices
        ret_cols  = codes_t0                           # (cap_width,) col indices
        window    = ret_mat[np.ix_(ret_rows, ret_cols)]  # (T_win, cap_width)
        valid     = ~np.isnan(window).any(axis=0)      # (cap_width,)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) >= n_stocks:
            chosen = rng.choice(valid_idx, n_stocks, replace=False)
            windows.append(window[:, chosen].T)        # (n_stocks, T_win)
    return windows


# ─── Panel A: Spectral density at matched (N, q) ──────────────────────────────
print("\n=== Panel A: Spectral density ===")

rng = np.random.default_rng(42)

fig, axes = plt.subplots(1, len(Q_VALUES), figsize=(6 * len(Q_VALUES), 5), sharey=False)

for ax, q in zip(axes, Q_VALUES):
    T_win = int(N / q)
    print(f"  q={q:.1f}  T={T_win}")

    # Real data: large-cap synchronous windows
    wins = find_sync_windows(avail_codes, avail_to_ret, ret_mat,
                             T_win, LARGE_CAP, n_win=40, n_stocks=N, rng=rng)
    if not wins:
        ax.set_title(f"q={q:.1f} — no sync windows found"); continue

    real_lam = []
    for W in wins:
        Wc = W - W.mean(axis=1, keepdims=True)
        S  = Wc @ Wc.T / (T_win - 1)
        d  = np.sqrt(np.maximum(np.diag(S), 1e-12))
        C  = S / np.outer(d, d); np.fill_diagonal(C, 1.0)
        real_lam.append(np.linalg.eigvalsh(0.5 * (C + C.T)))
    real_lam = np.concatenate(real_lam)

    # IW synthetic
    print(f"    generating {N_IW} IW samples…")
    iw_lam = iw_eigenvalues(N, q, N_IW)

    # MP law
    x_mp, pdf_mp = mp_density(q)

    bins = np.linspace(0, max(real_lam.max(), iw_lam.max()) * 1.05, 60)
    ax.hist(iw_lam, bins=bins, density=True, alpha=0.45, color="steelblue",
            label="IW synthetic")
    ax.hist(real_lam, bins=bins, density=True, alpha=0.45, color="darkorange",
            label="Real large-cap")
    ax.plot(x_mp, pdf_mp, color="crimson", linewidth=2, label="MP law")
    ax.set_title(f"q = N/T = {q:.1f}  (T = {T_win})", fontsize=11)
    ax.set_xlabel("Eigenvalue λ")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

axes[0].set_ylabel("Density")
fig.suptitle("Spectral density: IW synthetic vs real large-cap vs Marchenko-Pastur law",
             fontsize=12)
plt.tight_layout()
plt.savefig("images/step0_spectral_density.png", dpi=150)
plt.close()
print("  Saved → images/step0_spectral_density.png")


# ─── Panel B: Rolling eigenvalue stability ────────────────────────────────────
print("\n=== Panel B: Rolling eigenvalue stability ===")

q_roll = 0.5
T_win  = ROLL_T

wins = find_sync_windows(avail_codes, avail_to_ret, ret_mat,
                         T_win, LARGE_CAP, n_win=80, n_stocks=N, rng=rng)
print(f"  Found {len(wins)} rolling windows (T={T_win})")

real_top = []
for W in wins:
    Wc = W - W.mean(axis=1, keepdims=True)
    S  = Wc @ Wc.T / (T_win - 1)
    d  = np.sqrt(np.maximum(np.diag(S), 1e-12))
    C  = S / np.outer(d, d); np.fill_diagonal(C, 1.0)
    lam = np.linalg.eigvalsh(0.5 * (C + C.T))
    real_top.append(lam[-TOP_K:][::-1])           # top-K descending
real_top = np.array(real_top)                     # (n_win, TOP_K)

# IW: same N_IW draws
iw_top = []
for _ in range(len(wins)):
    df   = np.random.randint(int(1.5 * (N + 2)), int(3.0 * N))
    Z    = np.random.randn(df, N)
    Sig  = np.linalg.solve(Z.T @ Z, np.eye(N)) * (df - N - 1)
    L    = np.linalg.cholesky(Sig)
    R    = L @ np.random.randn(N, T_win)
    S    = np.cov(R)
    d    = np.sqrt(np.maximum(np.diag(S), 1e-12))
    C    = S / np.outer(d, d); np.fill_diagonal(C, 1.0)
    lam  = np.linalg.eigvalsh(0.5 * (C + C.T))
    iw_top.append(lam[-TOP_K:][::-1])
iw_top = np.array(iw_top)                         # (n_win, TOP_K)

# Coefficient of variation per eigenvalue rank
cv_real = real_top.std(axis=0) / np.abs(real_top.mean(axis=0))
cv_iw   = iw_top.std(axis=0)   / np.abs(iw_top.mean(axis=0))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = plt.cm.Blues(np.linspace(0.4, 0.95, TOP_K))
for k in range(TOP_K):
    ax1.plot(real_top[:, k], color=colors[k], alpha=0.8, linewidth=1.2,
             label=f"λ_{k+1}")
ax1.set_xlabel("Window index"); ax1.set_ylabel("Eigenvalue")
ax1.set_title(f"Top-{TOP_K} eigenvalues across {len(wins)} rolling windows — real data\n"
              f"(N={N}, T={T_win}, q={q_roll})")
ax1.legend(fontsize=8); ax1.grid(alpha=0.2)

ranks = np.arange(1, TOP_K + 1)
ax2.bar(ranks - 0.2, cv_real, 0.35, color="darkorange", label="Real large-cap")
ax2.bar(ranks + 0.2, cv_iw,   0.35, color="steelblue",  label="IW synthetic")
ax2.set_xlabel("Eigenvalue rank k"); ax2.set_ylabel("Coefficient of variation σ/μ")
ax2.set_title("Eigenvalue stability: lower CV = more stable factor structure")
ax2.set_xticks(ranks)
ax2.legend(fontsize=9); ax2.grid(alpha=0.2, axis="y")

plt.suptitle("Rolling eigenvalue stability — real data vs IW synthetic", fontsize=12)
plt.tight_layout()
plt.savefig("images/step0_rolling_stability.png", dpi=150)
plt.close()
print("  Saved → images/step0_rolling_stability.png")


# ─── Panel C: Return tail statistics ─────────────────────────────────────────
print("\n=== Panel C: Tail statistics ===")

# Use all available data for each large-cap stock
cap_codes = avail_codes[:, LARGE_CAP].ravel()
unique_large = pd.Series(cap_codes).value_counts().head(150).index.to_numpy()

skew_real, kurt_real = [], []
for code in unique_large:
    col_ret = ret_mat[:, code]
    col_ret = col_ret[np.isfinite(col_ret)]
    if len(col_ret) < 60:
        continue
    skew_real.append(float(stats.skew(col_ret)))
    kurt_real.append(float(stats.kurtosis(col_ret)))  # excess kurtosis

# IW synthetic: per-stock tail stats from Gaussian data
skew_iw, kurt_iw = [], []
for _ in range(len(skew_real)):
    df = np.random.randint(int(1.5 * (100 + 2)), int(3.0 * 100))
    sigma_ii = 1.0 + np.random.exponential(0.3)   # IW diagonal ≈ 1
    T_long   = 1500
    r        = np.random.randn(T_long) * np.sqrt(sigma_ii)
    skew_iw.append(float(stats.skew(r)))
    kurt_iw.append(float(stats.kurtosis(r)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

bins_s = np.linspace(-2, 2, 50)
bins_k = np.linspace(-2, 20, 60)

ax1.hist(skew_real, bins=bins_s, density=True, alpha=0.6, color="darkorange", label="Real large-cap")
ax1.hist(skew_iw,   bins=bins_s, density=True, alpha=0.6, color="steelblue",  label="IW synthetic (Gaussian)")
ax1.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax1.set_xlabel("Per-stock skewness"); ax1.set_ylabel("Density")
ax1.set_title(f"Return skewness distribution\n"
              f"Real: μ={np.mean(skew_real):.2f}  IW: μ={np.mean(skew_iw):.2f}")
ax1.legend(fontsize=9); ax1.grid(alpha=0.2)

ax2.hist(kurt_real, bins=bins_k, density=True, alpha=0.6, color="darkorange", label="Real large-cap")
ax2.hist(kurt_iw,   bins=bins_k, density=True, alpha=0.6, color="steelblue",  label="IW synthetic (Gaussian)")
ax2.axvline(0, color="black", linewidth=0.8, linestyle="--", label="Normal baseline")
ax2.set_xlabel("Per-stock excess kurtosis"); ax2.set_ylabel("Density")
ax2.set_title(f"Return excess kurtosis distribution\n"
              f"Real: μ={np.mean(kurt_real):.2f}  IW: μ={np.mean(kurt_iw):.2f}")
ax2.legend(fontsize=9); ax2.grid(alpha=0.2)

plt.suptitle("Return tail statistics: real large-cap vs IW-Gaussian synthetic\n"
             "(real data has heavier tails and negative skew — IW underestimates tail risk)",
             fontsize=11)
plt.tight_layout()
plt.savefig("images/step0_tail_stats.png", dpi=150)
plt.close()
print("  Saved → images/step0_tail_stats.png")


# ─── Panel D: Missingness pattern ────────────────────────────────────────────
print("\n=== Panel D: Missingness pattern ===")

# For each unique stock in the small-cap range, find its first appearance
small_cap_vals = avail_codes[:, SMALL_CAP]        # (n_avail, 1000)
n_avail        = avail_codes.shape[0]

# Vectorized: for each column t, stack (code, t_idx)
t_idx_arr  = np.arange(n_avail)[:, None] * np.ones_like(small_cap_vals)
code_flat  = small_cap_vals.ravel().astype(int)
t_flat     = t_idx_arr.ravel().astype(int)

tmp = pd.DataFrame({"code": code_flat, "t": t_flat})
first_t = tmp.groupby("code")["t"].min().values  # days (0-indexed in avail_codes)
first_obs_frac = first_t / n_avail               # fraction of period elapsed

# Synthetic missing_constant=2 first observation: uniform on [0, 0.5]
synth_first_obs = np.random.uniform(0.0, 0.5, size=len(first_obs_frac))

fig, ax = plt.subplots(figsize=(11, 5))

bins = np.linspace(0, 1, 50)
ax.hist(first_obs_frac, bins=bins, density=True, alpha=0.65, color="darkorange",
        label=f"Real small-cap stocks (n={len(first_obs_frac):,})")
ax.hist(synth_first_obs, bins=bins, density=True, alpha=0.65, color="steelblue",
        label="Synthetic missing_constant=2 (uniform [0, 0.5])")

pct_early = (first_obs_frac < 0.2).mean() * 100
ax.text(0.02, 0.92,
        f"{pct_early:.0f}% of small-cap stocks\nhave first obs in first 20% of period",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

ax.axvline(0.5, color="steelblue", linewidth=1.5, linestyle="--", alpha=0.7,
           label="Synthetic max first-obs (0.5)")
ax.set_xlabel("First observation time (fraction of full period)", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("Missingness pattern: real small-cap stocks vs synthetic missing_constant=2\n"
             "(real missingness is dominated by IPO dates — milder than the synthetic stress test)",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("images/step0_missingness_pattern.png", dpi=150)
plt.close()
print("  Saved → images/step0_missingness_pattern.png")

print("\nDone. Images saved to images/step0_*.png")
