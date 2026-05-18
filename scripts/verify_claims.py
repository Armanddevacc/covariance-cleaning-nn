"""
scripts/verify_claims.py

Systematically verifies every quantitative claim in the article against
computed benchmark data.

Data sources:
  - Synthetic Frobenius sweep: from notebooks/comparison_syntheticdata.ipynb cell output
  - No-miss appendix:          from notebooks/comparison_syntheticdata_nomiss.ipynb cell output
  - Real-data no-miss Table 2: results/ablation_features/nomiss_lvar.npy
  - Real-data severity Table 3: results/ablation_features/miss_severity_lvar.npy

Run: .venv/bin/python scripts/verify_claims.py
"""
import numpy as np

PASS = "  PASS"
FAIL = "  FAIL"
WARN = "  WARN"

issues = []

def check(label, actual, claimed, tol=0.10, unit=""):
    """Report pass if |actual - claimed| / max(|claimed|, 1e-9) <= tol."""
    if claimed == 0:
        rel = abs(actual - claimed)
    else:
        rel = abs(actual - claimed) / abs(claimed)
    status = PASS if rel <= tol else FAIL
    if status == FAIL:
        issues.append(f"{label}: claimed={claimed}{unit}, actual={actual:.4g}{unit}")
    print(f"{status}  {label:60s}  claimed={claimed}{unit}  actual={actual:.4g}{unit}  rel_err={rel*100:.1f}%")
    return status == PASS

def check_bool(label, condition, desc=""):
    status = PASS if condition else FAIL
    if not condition:
        issues.append(f"{label}: {desc}")
    print(f"{status}  {label:60s}  {desc}")
    return condition

def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


# ─── 1. SYNTHETIC BENCHMARK (Section 3.1) ─────────────────────────────────────
section("Section 3.1 — Synthetic Benchmark (N=100, missing_constant=2, 160 samples/q)")

# Data from comparison_syntheticdata.ipynb notebook output (cell-table)
# Model: bigru_weights_syntheticdata.weights.h5
# 12 q values, 10 reps × 16 batch = 160 samples per q
Q_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
EmpMLE = [0.0666, 0.0763, 0.0849, 0.0940, 0.1016, 0.1090, 0.1159, 0.1219, 0.1280, 0.1352, 0.1395, 0.1493]
QIS    = [0.0578, 0.0646, 0.0675, 0.0716, 0.0732, 0.0739, 0.0779, 0.0804, 0.0791, 0.0856, 0.0877, 0.0805]
NN     = [0.0555, 0.0620, 0.0629, 0.0673, 0.0700, 0.0703, 0.0738, 0.0776, 0.0760, 0.0815, 0.0831, 0.0783]
Oracle = [0.0553, 0.0625, 0.0659, 0.0713, 0.0739, 0.0761, 0.0807, 0.0831, 0.0835, 0.0891, 0.0921, 0.0878]

Q_GRID  = np.array(Q_GRID)
EmpMLE  = np.array(EmpMLE)
QIS_s   = np.array(QIS)
NN_s    = np.array(NN)
Oracle_s= np.array(Oracle)

print("\n  Per-q NN vs QIS relative gains:")
for i, q in enumerate(Q_GRID):
    gain = (QIS_s[i] - NN_s[i]) / QIS_s[i] * 100
    print(f"    q={q:.1f}: gain={gain:.1f}%")

# Claim A: NN outperforms QIS at every q
check_bool("A: NN < QIS at ALL tested q values",
           all(NN_s < QIS_s),
           f"min(QIS-NN)={np.min(QIS_s-NN_s):.5f}")

# Claim B: gain 2.6% at q=0.3
gain_q03 = (QIS_s[0] - NN_s[0]) / QIS_s[0] * 100
check("B: NN vs QIS gain at q=0.3", gain_q03, 2.6, tol=0.20, unit="%")

# Claim B: gain 4.6% at q=1.0
idx_10 = list(Q_GRID).index(1.0) if 1.0 in Q_GRID else np.argmin(abs(Q_GRID - 1.0))
gain_q10 = (QIS_s[idx_10] - NN_s[idx_10]) / QIS_s[idx_10] * 100
check("B: NN vs QIS gain at q=1.0", gain_q10, 4.6, tol=0.20, unit="%")

# Claim C: gain over pairwise MLE 10% at q=0.3
gain_emp03 = (EmpMLE[0] - NN_s[0]) / EmpMLE[0] * 100
check("C: NN vs Pairwise gain at q=0.3", gain_emp03, 10.0, tol=0.20, unit="%")

# Claim C: gain over pairwise MLE 46% at q=1.5
idx_15 = np.argmin(abs(Q_GRID - 1.5))
gain_emp15 = (EmpMLE[idx_15] - NN_s[idx_15]) / EmpMLE[idx_15] * 100
check("C: NN vs Pairwise gain at q=1.5", gain_emp15, 46.0, tol=0.05, unit="%")

# Table 1 mean values over q in [0.3, 1.5]
nn_mean    = NN_s.mean()
qis_mean   = QIS_s.mean()
oracle_mean= Oracle_s.mean()
emp_mean   = EmpMLE.mean()

print(f"\n  Mean Frobenius losses: NN={nn_mean:.4f} QIS={qis_mean:.4f} Oracle={oracle_mean:.4f} Emp={emp_mean:.4f}")

check("D Table1: NN mean Frobenius",     nn_mean,     0.075, tol=0.05)
check("D Table1: QIS mean Frobenius",    qis_mean,    0.077, tol=0.05)
check("D Table1: Oracle mean Frobenius", oracle_mean, 0.078, tol=0.05)
check("D Table1: Pairwise mean Frobenius", emp_mean,  0.110, tol=0.05)

# Claims about average gains (using mean values)
avg_gain_vs_qis    = (qis_mean - nn_mean) / qis_mean * 100
avg_gain_vs_pair   = (emp_mean - nn_mean) / emp_mean * 100
avg_gain_vs_oracle = (oracle_mean - nn_mean) / oracle_mean * 100

check("E: Avg NN gain vs QIS",      avg_gain_vs_qis,    1.7, tol=0.30, unit="%")
check("E: Avg NN gain vs Pairwise", avg_gain_vs_pair,  32.0, tol=0.10, unit="%")
check("E: Avg NN gain vs Oracle",   avg_gain_vs_oracle, 3.3, tol=0.30, unit="%")

# Claim F: Oracle outperforms both NN and QIS at q < ~0.5
oracle_beats_nn_below05 = all(
    Oracle_s[i] <= NN_s[i] for i, q in enumerate(Q_GRID) if q < 0.5
)
oracle_beats_qis_below05 = all(
    Oracle_s[i] <= QIS_s[i] for i, q in enumerate(Q_GRID) if q < 0.5
)
check_bool("F: Oracle < NN at q < 0.5",
           oracle_beats_nn_below05,
           str({q: (Oracle_s[i] <= NN_s[i]) for i, q in enumerate(Q_GRID) if q < 0.5}))
check_bool("F: Oracle < QIS at q < 0.5",
           oracle_beats_qis_below05,
           str({q: (Oracle_s[i] <= QIS_s[i]) for i, q in enumerate(Q_GRID) if q < 0.5}))

# Claim G: Oracle underperforms QIS above q~0.6
oracle_above06_vs_qis = {
    q: (Oracle_s[i], QIS_s[i], Oracle_s[i] > QIS_s[i])
    for i, q in enumerate(Q_GRID) if q > 0.6
}
oracle_worse_than_qis_above06 = all(v[2] for v in oracle_above06_vs_qis.values())
check_bool("G: Oracle > QIS at ALL q > 0.6 (crossover before q=0.6)",
           oracle_worse_than_qis_above06,
           str({q: f"Ora={v[0]:.4f} QIS={v[1]:.4f} {'Ora>QIS' if v[2] else 'Ora<=QIS'}"
                for q, v in oracle_above06_vs_qis.items()}))
# Show actual crossover
for i in range(len(Q_GRID) - 1):
    if Oracle_s[i] <= QIS_s[i] and Oracle_s[i+1] > QIS_s[i+1]:
        print(f"    Actual Oracle-QIS crossover: between q={Q_GRID[i]:.1f} and q={Q_GRID[i+1]:.1f}")


# ─── 2. APPENDIX: No-missingness (Appendix A) ─────────────────────────────────
section("Appendix A — No-Missingness Spectral Cleaning")

# Data from comparison_syntheticdata_nomiss.ipynb (cell-table output)
# Model: bigru_weights_nomiss.weights.h5
# N in {80,100,120}, n_steps=10, BATCH=30, averaged over N and steps
# qs = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 5.0]

app_qs     = [0.2,  0.4,  0.6,  0.8,  1.0,  1.2,  1.4,  1.6,  1.8,  2.0,  3.0,  5.0]
app_qis_f  = [0.0404, 0.0522, 0.0587, 0.0638, 0.0703, 0.0707, 0.0745, 0.0762, 0.0796, 0.0789, 0.0829, 0.0896]
app_nn_f   = [0.0528, 0.0594, 0.0622, 0.0666, 0.0715, 0.0720, 0.0756, 0.0767, 0.0797, 0.0784, 0.0818, 0.0882]
# Eigenvalue MSE vs Opt-Oracle λ*
# Cols: Emp, Oracle, QIS, NN
app_qis_mse= [0.00636, 0.01043, 0.01295, 0.01720, 0.03426, 0.02786, 0.03129, 0.03534, 0.03842, 0.04042, 0.04886, 0.05347]
app_nn_mse = [0.12992, 0.09875, 0.05992, 0.05743, 0.05226, 0.05103, 0.05020, 0.04688, 0.04467, 0.03434, 0.03572, 0.03492]
app_oracle_mse=[0.01269,0.03341,0.05675,0.07852,0.10104,0.12098,0.14185,0.16109,0.18358,0.19827,0.27545,0.40068]

app_qs      = np.array(app_qs)
app_qis_f   = np.array(app_qis_f)
app_nn_f    = np.array(app_nn_f)
app_qis_mse = np.array(app_qis_mse)
app_nn_mse  = np.array(app_nn_mse)

# Article Table 4 (Appendix) claims:
art_rows = {
    0.4: dict(qis_f=0.0522, nn_f=0.0594, qis_mse=0.0104, nn_mse=0.0988),
    1.0: dict(qis_f=0.0703, nn_f=0.0715, qis_mse=0.0343, nn_mse=0.0523),
    1.8: dict(qis_f=0.0796, nn_f=0.0797, qis_mse=0.0384, nn_mse=0.0447),
    2.0: dict(qis_f=0.0789, nn_f=0.0784, qis_mse=0.0404, nn_mse=0.0343),
    3.0: dict(qis_f=0.0829, nn_f=0.0818, qis_mse=0.0489, nn_mse=0.0357),
    5.0: dict(qis_f=0.0896, nn_f=0.0882, qis_mse=0.0535, nn_mse=0.0349),
}

print("\n  Table 4 (Appendix) verification:")
for q_val, claimed in art_rows.items():
    idx = np.argmin(abs(app_qs - q_val))
    check(f"App Table: QIS Frob at q={q_val}", app_qis_f[idx], claimed['qis_f'], tol=0.03)
    check(f"App Table: NN Frob at q={q_val}",  app_nn_f[idx],  claimed['nn_f'],  tol=0.03)
    check(f"App Table: QIS EigMSE at q={q_val}", app_qis_mse[idx], claimed['qis_mse'], tol=0.05)
    check(f"App Table: NN EigMSE at q={q_val}",  app_nn_mse[idx],  claimed['nn_mse'],  tol=0.05)

# "QIS eigenvalue MSE at q=0.4 is 0.0104 — three times lower than Oracle (0.033)"
idx_04 = np.argmin(abs(app_qs - 0.4))
check("App: QIS EigMSE q=0.4 ≈ 0.0104",     app_qis_mse[idx_04], 0.0104, tol=0.05)
check("App: Oracle EigMSE q=0.4 ≈ 0.033",   app_oracle_mse[idx_04], 0.033, tol=0.05)
ratio_04 = app_oracle_mse[idx_04] / app_qis_mse[idx_04]
check("App: Oracle/QIS ratio ≈ 3 at q=0.4", ratio_04, 3.0, tol=0.10)

# "NN overtakes QIS around q ≈ 2" — for both Frobenius and EigMSE
print("\n  NN vs QIS EigMSE crossover:")
for i in range(len(app_qs) - 1):
    if app_nn_mse[i] >= app_qis_mse[i] and app_nn_mse[i+1] < app_qis_mse[i+1]:
        print(f"    EigMSE crossover: q in [{app_qs[i]}, {app_qs[i+1]}]")
        check_bool("App: NN overtakes QIS (EigMSE) around q≈2",
                   app_qs[i] >= 1.5 and app_qs[i+1] <= 2.5,
                   f"crossover at [{app_qs[i]}, {app_qs[i+1]}]")
        break

print("  NN vs QIS Frob crossover:")
for i in range(len(app_qs) - 1):
    if app_nn_f[i] >= app_qis_f[i] and app_nn_f[i+1] <= app_qis_f[i+1]:
        print(f"    Frob crossover: q in [{app_qs[i]}, {app_qs[i+1]}]")
        break

# "At q=3.0: NN reduces EigMSE vs QIS by 27%; at q=5.0 by 35%"
idx_30 = np.argmin(abs(app_qs - 3.0))
idx_50 = np.argmin(abs(app_qs - 5.0))
gain_30 = (app_qis_mse[idx_30] - app_nn_mse[idx_30]) / app_qis_mse[idx_30] * 100
gain_50 = (app_qis_mse[idx_50] - app_nn_mse[idx_50]) / app_qis_mse[idx_50] * 100
check("App: NN EigMSE gain vs QIS at q=3.0 ≈ 27%", gain_30, 27.0, tol=0.10, unit="%")
check("App: NN EigMSE gain vs QIS at q=5.0 ≈ 35%", gain_50, 35.0, tol=0.10, unit="%")


# ─── 3. REAL-DATA NO-MISS (Table 2, Section 3.2) ─────────────────────────────
section("Table 2 — Real-Data No-Missingness (N=100, N_STEPS=30)")

nomiss = np.load("results/ablation_features/nomiss_lvar.npy")
# Q_VALUES = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0], NAMES = [Sample,LW,QIS,POET,NN]
NM_Q = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
NM_NAMES = ['Sample', 'LW', 'QIS', 'POET', 'NN']
assert nomiss.shape[0] == len(NM_Q), f"Expected {len(NM_Q)} q values, got {nomiss.shape[0]}"

nm_means = np.nanmean(nomiss, axis=1) * 1e4  # (9, 5), ×10^{-4}

# Article Table 2 values (×10^{-4}):
art_tab2 = {
    0.3:  [2.60, 1.90, 1.70, 2.00, 1.60],
    0.5:  [2.40, 1.80, 1.50, 2.10, 1.50],
    0.7:  [4.00, 2.40, 2.20, 2.40, 2.50],
    1.0:  [133,  2.20, 4.80, 2.40, 2.20],
    1.2:  [5.40, 1.40, 1.20, 1.40, 1.00],
    1.5:  [13.8, 4.70, 4.90, 4.30, 3.30],
    2.0:  [2.80, 1.70, 1.60, 1.70, 1.10],
    2.5:  [1.70, 1.20, 1.00, 1.20, 1.00],
    3.0:  [2.20, 1.50, 1.10, 1.30, 1.40],
}

print("\n  Table 2 cell-by-cell verification (×10^{-4}):")
for qi, q in enumerate(NM_Q):
    if q not in art_tab2:
        continue
    for ei, name in enumerate(NM_NAMES):
        actual_v = nm_means[qi, ei]
        claimed_v = art_tab2[q][ei]
        tol = 0.20 if claimed_v > 10 else 0.15  # looser tol for noisy large values
        check(f"Tab2 q={q} {name}", actual_v, claimed_v, tol=tol, unit="×10⁻⁴")

# Section 3.2 text claims
print("\n  Narrative claims in Section 3.2:")

# "QIS dominates at q ≤ 0.7"
qis_beats_nn_below07 = all(
    nm_means[qi, 2] <= nm_means[qi, 4]
    for qi, q in enumerate(NM_Q) if q <= 0.7
)
check_bool("H: QIS ≤ NN at all q ≤ 0.7 (QIS dominates)",
           qis_beats_nn_below07,
           str({q: f"QIS={nm_means[qi,2]:.2f} NN={nm_means[qi,4]:.2f}"
                for qi, q in enumerate(NM_Q) if q <= 0.7}))

# "NN overtakes from q=1.2 onward"
nn_beats_qis_above12 = all(
    nm_means[qi, 4] <= nm_means[qi, 2]
    for qi, q in enumerate(NM_Q) if q >= 1.2
)
check_bool("I: NN ≤ QIS at all q ≥ 1.2 (NN overtakes from q=1.2)",
           nn_beats_qis_above12,
           str({q: f"NN={nm_means[qi,4]:.2f} QIS={nm_means[qi,2]:.2f}"
                for qi, q in enumerate(NM_Q) if q >= 1.2}))

# "Gains over QIS: 17% at q=1.2 and 32% at q∈{1.5, 2.0}"
qi_12 = NM_Q.index(1.2)
qi_15 = NM_Q.index(1.5)
qi_20 = NM_Q.index(2.0)
gain_12 = (nm_means[qi_12, 2] - nm_means[qi_12, 4]) / nm_means[qi_12, 2] * 100
gain_15 = (nm_means[qi_15, 2] - nm_means[qi_15, 4]) / nm_means[qi_15, 2] * 100
gain_20 = (nm_means[qi_20, 2] - nm_means[qi_20, 4]) / nm_means[qi_20, 2] * 100
check("J: NN gain over QIS at q=1.2 ≈ 17%", gain_12, 17.0, tol=0.15, unit="%")
check("J: NN gain over QIS at q=1.5 ≈ 32%", gain_15, 32.0, tol=0.15, unit="%")
check("J: NN gain over QIS at q=2.0 ≈ 32%", gain_20, 32.0, tol=0.15, unit="%")


# ─── 4. REAL-DATA SEVERITY SWEEP (Table 3, Section 3.2) ──────────────────────
section("Table 3 — Real-Data Severity Sweep (N=50, T_in=250)")

sev_lvar   = np.load("results/ablation_features/miss_severity_lvar.npy")
sev_sync_T = np.load("results/ablation_features/miss_severity_sync_T.npy")

# benchmark_miss_severity.py: MISS_FRACS = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9]
# NAMES = [QIS(sync), POET, Stambaugh, NN]
MISS_FRACS = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9]
SEV_NAMES  = ['QIS', 'POET', 'Stambaugh', 'NN']
N_SYN = 50
T_IN  = 250

sev_means  = np.nanmean(sev_lvar, axis=1) * 1e4  # (8, 4), ×10^{-4}
sev_sync   = np.nanmean(sev_sync_T, axis=1)       # mean T_sync per miss_frac level

# Article Table 3 (×10^{-4}):
art_tab3 = {
    0.0: dict(T_sync=250, q_qis=0.20, QIS=2.0, POET=3.0, Stam=2.0, NN=1.9, gain=3.7),
    0.4: dict(T_sync=152, q_qis=0.33, QIS=2.0, POET=3.0, Stam=2.0, NN=1.5, gain=23.0),
    0.7: dict(T_sync=79,  q_qis=0.63, QIS=5.0, POET=6.0, Stam=12.0, NN=2.4, gain=52.0),
    0.9: dict(T_sync=31,  q_qis=1.63, QIS=1.0, POET=1.0, Stam=42.0, NN=0.9, gain=11.0),
}

print("\n  Table 3 cell verification (×10^{-4}):")
for mf, claimed in art_tab3.items():
    mi = MISS_FRACS.index(mf)
    row = sev_means[mi]
    t_sync = sev_sync[mi]
    actual_gain = (row[0] - row[3]) / row[0] * 100 if row[0] > 0 else np.nan
    actual_qis  = N_SYN / t_sync

    check(f"Tab3 α={int(mf*100)}% T_sync", t_sync, claimed['T_sync'], tol=0.10, unit=" days")
    check(f"Tab3 α={int(mf*100)}% q_QIS", actual_qis, claimed['q_qis'], tol=0.10)
    check(f"Tab3 α={int(mf*100)}% QIS",   row[0], claimed['QIS'],  tol=0.20, unit="×10⁻⁴")
    check(f"Tab3 α={int(mf*100)}% POET",  row[1], claimed['POET'], tol=0.20, unit="×10⁻⁴")
    check(f"Tab3 α={int(mf*100)}% Stam",  row[2], claimed['Stam'], tol=0.20, unit="×10⁻⁴")
    check(f"Tab3 α={int(mf*100)}% NN",    row[3], claimed['NN'],   tol=0.20, unit="×10⁻⁴")
    check(f"Tab3 α={int(mf*100)}% NN gain over QIS", actual_gain, claimed['gain'], tol=0.15, unit="%")

print("\n  Narrative claims for Table 3:")

# "all PSD methods within 4% of one another at α=0%"
mi0 = 0  # α=0%
row0 = sev_means[mi0]
psd_methods = [row0[0], row0[1], row0[3]]  # QIS, POET, NN (exclude Stambaugh as non-PSD issues)
max_spread = (max(psd_methods) - min(psd_methods)) / min(psd_methods) * 100
check_bool("K: PSD methods (QIS,POET,NN) within 4% at α=0%",
           max_spread <= 4.0,
           f"spread={max_spread:.1f}%  QIS={row0[0]:.2f} POET={row0[1]:.2f} NN={row0[3]:.2f}")

# Stambaugh competitive at low missingness, collapses beyond 70%
mi_80 = MISS_FRACS.index(0.8)
stam_80 = sev_means[mi_80, 2]
qis_80  = sev_means[mi_80, 0]
check_bool("L: Stambaugh collapses at α≥80% (Stam >> QIS)",
           stam_80 > 5 * qis_80,
           f"Stam={stam_80:.2f} QIS={qis_80:.2f} (×10⁻⁴)")

# "POET consistently dominated by QIS and NN whenever missingness is substantial"
poet_vs_qis = all(sev_means[mi, 1] >= sev_means[mi, 0]
                  for mi, mf in enumerate(MISS_FRACS) if mf >= 0.4)
poet_vs_nn  = all(sev_means[mi, 1] >= sev_means[mi, 3]
                  for mi, mf in enumerate(MISS_FRACS) if mf >= 0.4)
check_bool("M: POET ≥ QIS at all α ≥ 40%",
           poet_vs_qis,
           str({mf: f"P={sev_means[mi,1]:.2f} QIS={sev_means[mi,0]:.2f}"
                for mi, mf in enumerate(MISS_FRACS) if mf >= 0.4}))
check_bool("M: POET ≥ NN at all α ≥ 40%",
           poet_vs_nn,
           str({mf: f"P={sev_means[mi,1]:.2f} NN={sev_means[mi,3]:.2f}"
                for mi, mf in enumerate(MISS_FRACS) if mf >= 0.4}))

# T_in - T_sync ≈ 171 at α=70%
mi_70 = MISS_FRACS.index(0.7)
t_info_adv = T_IN - sev_sync[mi_70]
check("N: Information advantage T_in - T_sync ≈ 171 at α=70%", t_info_adv, 171.0, tol=0.10, unit=" days")
threefold = T_IN / sev_sync[mi_70]
check("N: T_in/T_sync ≈ 3 (threefold) at α=70%", threefold, 3.0, tol=0.15)


# ─── SUMMARY ──────────────────────────────────────────────────────────────────
section("SUMMARY")

if issues:
    print(f"\n  {len(issues)} claim(s) FAILED or have notable discrepancies:\n")
    for issue in issues:
        print(f"    • {issue}")
else:
    print("\n  All claims verified!")

print(f"\n  Total issues flagged: {len(issues)}")
