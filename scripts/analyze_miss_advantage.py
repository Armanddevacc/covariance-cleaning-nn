"""
scripts/analyze_miss_advantage.py

Analyzes how the NN advantage over QIS grows with:
  1. Increasing missingness severity (severity sweep: fixed N=50, T_in=250, varying alpha)
  2. Increasing q (q-sweep: fixed N=80, miss_frac<=50%, varying q)

The key driver is q_QIS = N / T_sync (the effective concentration ratio seen by QIS).
As missingness grows, T_sync shrinks while the NN still exploits full T_in pairwise data.

Run: .venv/bin/python scripts/analyze_miss_advantage.py
"""

import os
import sys
sys.path.insert(0, ".")
os.makedirs("images", exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Data ────────────────────────────────────────────────────────────────────

sev_lvar  = np.load("results/ablation_features/miss_severity_lvar.npy")   # (8,25,4)
sev_sync  = np.load("results/ablation_features/miss_severity_sync_T.npy") # (8,25)
miss_lvar = np.load("results/ablation_features/miss_lvar.npy")            # (5,20,5)

MISS_FRACS = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9]
SEV_NAMES  = ["QIS(sync)", "POET", "Stambaugh", "NN"]
N_SEV, T_IN = 50, 250

Q_VALUES   = [0.5, 0.7, 1.0, 1.2, 1.5]
MISS_NAMES = ["Pairwise_MLE", "QIS(sync)", "POET", "Stambaugh", "NN"]
N_MISS     = 80

sev_mean    = np.nanmean(sev_lvar, axis=1)     # (8,4)
sev_stderr  = np.nanstd(sev_lvar, axis=1) / np.sqrt(25)
tsync_mean  = sev_sync.mean(axis=1)            # (8,)
tsync_std   = sev_sync.std(axis=1) / np.sqrt(25)

miss_mean   = np.nanmean(miss_lvar, axis=1)    # (5,5)
miss_stderr = np.nanstd(miss_lvar, axis=1) / np.sqrt(20)

mf_arr  = np.array(MISS_FRACS) * 100
q_qis   = N_SEV / tsync_mean                  # effective q seen by QIS

qis_si = SEV_NAMES.index("QIS(sync)")
nn_si  = SEV_NAMES.index("NN")
po_si  = SEV_NAMES.index("POET")
st_si  = SEV_NAMES.index("Stambaugh")

qis_mi = MISS_NAMES.index("QIS(sync)")
nn_mi  = MISS_NAMES.index("NN")


# ─── Table 1: Severity sweep ─────────────────────────────────────────────────

print("=" * 90)
print("SEVERITY SWEEP  (N=50, q_eff=0.20, T_in=250, T_OOS=20, 25 windows)  ×10⁻⁴")
print("=" * 90)
print(f"{'α%':>5}  {'T_sync':>7}  {'q_QIS':>6}  {'QIS':>8}  {'POET':>8}  {'Stam':>8}  "
      f"{'NN':>8}  {'NN_vs_QIS':>10}  {'NN_vs_POET':>11}")
print("-" * 90)
for mi, mf in enumerate(MISS_FRACS):
    ts    = tsync_mean[mi]
    qq    = q_qis[mi]
    r     = sev_mean[mi] * 1e4
    se    = sev_stderr[mi] * 1e4
    g_qis  = (r[qis_si] - r[nn_si]) / r[qis_si] * 100
    g_poet = (r[po_si]  - r[nn_si]) / r[po_si]  * 100
    print(f"{mf*100:5.0f}%  {ts:7.0f}  {qq:6.2f}  {r[qis_si]:8.2f}  {r[po_si]:8.2f}  "
          f"{r[st_si]:8.2f}  {r[nn_si]:8.2f}  {g_qis:+10.1f}%  {g_poet:+11.1f}%")
print("=" * 90)

# ─── Table 2: Q-sweep with missingness ───────────────────────────────────────

print()
print("=" * 80)
print("Q-SWEEP WITH MISSINGNESS  (N=80, miss_frac ≤ 50%, T_OOS=20, 20 windows)  ×10⁻⁴")
print("=" * 80)
print(f"{'q':>5}  {'T_in':>6}  {'Pairwise':>10}  {'QIS(sync)':>10}  {'POET':>8}  "
      f"{'Stam':>8}  {'NN':>8}  {'NN_vs_QIS':>10}")
print("-" * 80)
for qi, q in enumerate(Q_VALUES):
    T_in_q = int(N_MISS / q)
    r      = miss_mean[qi] * 1e4
    g      = (r[qis_mi] - r[nn_mi]) / r[qis_mi] * 100
    print(f"{q:5.1f}  {T_in_q:6d}  {r[0]:10.2f}  {r[1]:10.4f}  {r[2]:8.4f}  "
          f"{r[3]:8.4f}  {r[4]:8.4f}  {g:+10.1f}%")
print("=" * 80)


# ─── Key insight summary ─────────────────────────────────────────────────────

print()
print("=== NN advantage sorted by q_QIS (severity sweep) ===")
items = [(q_qis[mi], (sev_mean[mi,qis_si]-sev_mean[mi,nn_si])/sev_mean[mi,qis_si]*100, mf)
         for mi, mf in enumerate(MISS_FRACS)]
items.sort()
for qq, gain, mf in items:
    bar = "█" * int(max(gain, 0) / 3)
    print(f"  q_QIS={qq:.2f}  (α={mf*100:.0f}%)  NN gain={gain:+.1f}%  {bar}")


# ─── Figure: 2-panel — severity sweep + q-sweep ──────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Panel A: Severity sweep — L_var vs alpha ──────────────────────────────────
ax = axes[0]
COLORS = {
    "QIS(sync)":  ("darkorange", "-"),
    "POET":       ("#e15759",    "-"),
    "Stambaugh":  ("#59a14f",    "--"),
    "NN":         ("steelblue",  "-"),
}
for i, name in enumerate(SEV_NAMES):
    col, ls = COLORS[name]
    y  = sev_mean[:, i] * 1e4
    se = sev_stderr[:, i] * 1e4
    ax.plot(mf_arr, y, color=col, ls=ls, lw=2.5, marker="o", ms=5, label=name)
    ax.fill_between(mf_arr, y - se, y + se, alpha=0.12, color=col)

# Secondary x-axis showing q_QIS
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks(mf_arr)
ax_top.set_xticklabels([f"{q:.2f}" for q in q_qis], fontsize=8, rotation=45)
ax_top.set_xlabel("$q_{\\mathrm{QIS}} = N / T_{\\mathrm{sync}}$", fontsize=9)

ax.set_xlabel("Max missing fraction α (%)", fontsize=11)
ax.set_ylabel("Realized portfolio variance ($\\times 10^{-4}$)", fontsize=11)
ax.set_title(f"(A)  Severity sweep\nN={N_SEV}, $q_{{\\mathrm{{eff}}}}={N_SEV/T_IN:.2f}$, $T_{{\\mathrm{{in}}}}={T_IN}$", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.2)

# ── Panel B: NN gain vs q_QIS (severity) and q (q-sweep) ─────────────────────
ax = axes[1]

# Severity: NN gain over QIS vs q_QIS
gains_sev = [(q_qis[mi], (sev_mean[mi,qis_si]-sev_mean[mi,nn_si])/sev_mean[mi,qis_si]*100)
             for mi in range(len(MISS_FRACS))]
gains_sev.sort()
qq_sev, g_sev = zip(*gains_sev)

ax.plot(qq_sev, g_sev, color="darkorange", lw=2.5, marker="o", ms=7,
        label="NN vs QIS  (severity sweep, $x$-axis = $q_{QIS}$)")

# Add alpha annotations
for mi in range(len(MISS_FRACS)):
    qq = q_qis[mi]
    g  = (sev_mean[mi,qis_si]-sev_mean[mi,nn_si])/sev_mean[mi,qis_si]*100
    ax.annotate(f"α={int(MISS_FRACS[mi]*100)}%", (qq, g),
                textcoords="offset points", xytext=(5, 4), fontsize=7.5, color="darkorange")

# Q-sweep: NN gain over QIS vs q  (x-axis = nominal q)
gains_miss = []
for qi, q in enumerate(Q_VALUES):
    r = miss_mean[qi] * 1e4
    g = (r[qis_mi]-r[nn_mi])/r[qis_mi]*100
    gains_miss.append((q, g))

q_miss, g_miss = zip(*gains_miss)
ax.plot(q_miss, g_miss, color="steelblue", lw=2, ls="--", marker="s", ms=6,
        label="NN vs QIS  (q-sweep w/ missingness, $x$-axis = $q$)")

ax.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
ax.axvline(1, color="grey", lw=1, ls=":", alpha=0.5, label="$q=1$ ($N=T$)")
ax.set_xlabel("Effective q (= $q_{QIS}$ for severity, = $q$ for q-sweep)", fontsize=10)
ax.set_ylabel("NN portfolio variance gain over QIS (%)", fontsize=11)
ax.set_title("(B)  NN advantage vs effective concentration\n"
             "Peaks when $q_{QIS} \\approx 0.6$–$0.9$, then compresses", fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

plt.suptitle("NN advantage grows with missingness severity — driven by $q_{QIS} = N / T_{\\mathrm{sync}}$",
             fontsize=13)
plt.tight_layout()
plt.savefig("images/miss_advantage_analysis.png", dpi=150, bbox_inches="tight")
print("\nSaved → images/miss_advantage_analysis.png")
plt.close()

# ─── Figure 2: Severity sweep — NN gain vs q_QIS cleanly ─────────────────────

fig, ax = plt.subplots(figsize=(8, 5))

# Plot gain vs q_QIS with error bands
gains_arr = np.array([
    (sev_mean[mi,qis_si]-sev_mean[mi,nn_si])/sev_mean[mi,qis_si]*100
    for mi in range(len(MISS_FRACS))
])

# Propagated std: Var(gain) ≈ (se_qis/qis)^2 + (se_nn/nn)^2 (rough approx)
gains_se = np.array([
    np.sqrt((sev_stderr[mi,qis_si]/sev_mean[mi,qis_si])**2 +
            (sev_stderr[mi,nn_si] /sev_mean[mi,nn_si])**2) * abs(gains_arr[mi])
    for mi in range(len(MISS_FRACS))
])

# Sort by q_QIS
order = np.argsort(q_qis)
qq_plot = q_qis[order]
g_plot  = gains_arr[order]
se_plot = gains_se[order]
mf_plot = [MISS_FRACS[i] for i in order]

ax.fill_between(qq_plot, g_plot - se_plot, g_plot + se_plot, alpha=0.15, color="steelblue")
ax.plot(qq_plot, g_plot, color="steelblue", lw=2.5, marker="o", ms=7, zorder=5)

for i, (qq, g, mf) in enumerate(zip(qq_plot, g_plot, mf_plot)):
    ax.annotate(f"α={int(mf*100)}%\n$q_{{QIS}}$={qq:.2f}",
                (qq, g), textcoords="offset points",
                xytext=(8, -4 if g > 0 else 4), fontsize=8, color="dimgrey")

ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
ax.set_xlabel("$q_{\\mathrm{QIS}} = N / T_{\\mathrm{sync}}$  (effective concentration seen by QIS)", fontsize=12)
ax.set_ylabel("NN portfolio variance gain over QIS (%)", fontsize=12)
ax.set_title(f"NN advantage vs QIS — severity sweep\n"
             f"N={N_SEV}, $q_{{\\mathrm{{eff}}}}={N_SEV/T_IN:.2f}$, $T_{{\\mathrm{{in}}}}={T_IN}$ (NN always uses full window)\n"
             f"NN advantage peaks at $q_{{QIS}} \\approx 0.9$, then compresses as both methods degrade",
             fontsize=11)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("images/miss_advantage_vs_qqis.png", dpi=150)
print("Saved → images/miss_advantage_vs_qqis.png")
plt.close()
