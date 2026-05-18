#!/usr/bin/env python3
"""
scripts/ablation_features.py — Feature set ablation for the BiGRU eigenvalue cleaner.

Ablation configs (3 seeds each, EarlyStopping patience=15):
  S1: [λ_emp, pos, q_global, T̃_min, T̃_max, q_eff]        — 6-feature baseline
  S2: [λ_emp, pos, q_eff, IPR]                              — minimal physics-motivated
  S3: [λ_emp, pos, q_eff, IPR, z_MP]                       — + MP z-score
  S4: all 8 features                                         — full control

Metrics per config:
  - Frobenius test loss (mean ± std across seeds)
  - Spearman ρ(λ_pred, λ_true) by quartile of λ_emp
  - Permutation importance per feature

Results → results/ablation_features/
"""

import os, sys, json, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.dataloader import tf_data_generator
from models.gru_denoiser import BiGRUSpectralDenoiserTensorFlow
from models.losses import tf_loss_function_mat

SAVE_DIR = os.path.join(ROOT, "results", "ablation_features")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE      = 20
N_MIN, N_MAX    = 70, 150
Q_MIN, Q_MAX    = 0.3, 1.5
MISSING_CONST   = 2
HIDDEN_SIZE     = 64
MAX_EPOCHS      = 300
STEPS_PER_EPOCH = 5
PATIENCE        = 15
LR              = 1e-4
SEEDS           = [0, 1, 2]
N_VAL_BATCHES   = 40
N_TEST_BATCHES  = 60
N_PERM_REPEATS  = 5

ALL_FEAT_NAMES = ["λ_emp", "pos", "q_global", "T̃_min", "T̃_max", "q_eff", "IPR", "z_MP"]

CONFIGS = {
    "S1": [0, 1, 2, 3, 4, 5],
    "S2": [0, 1, 5, 6],
    "S3": [0, 1, 5, 6, 7],
    "S4": [0, 1, 2, 3, 4, 5, 6, 7],
}


# ── Core helpers ──────────────────────────────────────────────────────────────

def _reconstruct_cov(lam_pred, Q_emp, Sigma_hat_diag):
    sqD = tf.sqrt(tf.linalg.diag(Sigma_hat_diag))
    return tf.matmul(
        tf.matmul(
            tf.matmul(tf.matmul(sqD, Q_emp), tf.linalg.diag(lam_pred)),
            tf.transpose(Q_emp, perm=[0, 2, 1]),
        ),
        sqD,
    )


def _batch_loss(model, batches, feat_cols):
    losses = []
    for b in batches:
        inp = tf.gather(b["input_seq"], feat_cols, axis=2)
        lam_pred = model(inp, training=False)
        Sp = _reconstruct_cov(lam_pred, b["Q_emp"], b["Sigma_hat_diag"])
        losses.append(float(tf_loss_function_mat(b["Sigma_true"], Sp, b["T"])))
    return float(np.mean(losses))


def _lam_true(Sigma_true_np):
    """Covariance (B, N, N) → true correlation eigenvalues, flattened."""
    d = np.sqrt(np.maximum(np.diagonal(Sigma_true_np, axis1=1, axis2=2), 1e-12))
    C = Sigma_true_np / (d[:, :, None] * d[:, None, :])
    C = 0.5 * (C + C.transpose(0, 2, 1))
    return np.linalg.eigvalsh(C).ravel()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(model, test_batches, feat_cols):
    """Returns (mean_loss, spearman_q[4], perm_imp[n_feat])."""
    lam_emp_acc, lam_pred_acc, lam_true_acc, base_losses = [], [], [], []

    for b in test_batches:
        inp = tf.gather(b["input_seq"], feat_cols, axis=2)
        lp = model(inp, training=False)
        Sp = _reconstruct_cov(lp, b["Q_emp"], b["Sigma_hat_diag"])
        base_losses.append(float(tf_loss_function_mat(b["Sigma_true"], Sp, b["T"])))
        lam_emp_acc.append(b["input_seq"][:, :, 0].numpy().ravel())
        lam_pred_acc.append(lp.numpy().ravel())
        lam_true_acc.append(_lam_true(b["Sigma_true"].numpy()))

    baseline = float(np.mean(base_losses))
    lam_emp  = np.concatenate(lam_emp_acc)
    lam_pred = np.concatenate(lam_pred_acc)
    lam_true = np.concatenate(lam_true_acc)

    # Spearman ρ by λ_emp quartile
    q_edges = np.percentile(lam_emp, [0, 25, 50, 75, 100])
    spearman_q = []
    for s in range(4):
        mask = (lam_emp >= q_edges[s]) & (lam_emp < q_edges[s + 1])
        rho = float(spearmanr(lam_pred[mask], lam_true[mask]).statistic) if mask.sum() > 20 else float("nan")
        spearman_q.append(rho)

    # permutation importance
    perm_imp = []
    for fi, global_fi in enumerate(feat_cols):
        scores = []
        for _ in range(N_PERM_REPEATS):
            pl = []
            for b in test_batches:
                inp_np = b["input_seq"].numpy().copy()
                flat = inp_np[:, :, global_fi].ravel()
                np.random.shuffle(flat)
                inp_np[:, :, global_fi] = flat.reshape(inp_np.shape[0], inp_np.shape[1])
                inp_p = tf.gather(tf.constant(inp_np, dtype=tf.float32), feat_cols, axis=2)
                lp2 = model(inp_p, training=False)
                Sp2 = _reconstruct_cov(lp2, b["Q_emp"], b["Sigma_hat_diag"])
                pl.append(float(tf_loss_function_mat(b["Sigma_true"], Sp2, b["T"])))
            scores.append(float(np.mean(pl)))
        perm_imp.append((float(np.mean(scores)) - baseline) / baseline)

    return baseline, spearman_q, perm_imp


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(feat_cols, seed, val_batches):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = BiGRUSpectralDenoiserTensorFlow(hidden_size=HIDDEN_SIZE)
    model(tf.zeros((1, 100, len(feat_cols))))  # build

    lr_sched  = tf.keras.optimizers.schedules.CosineDecay(LR, MAX_EPOCHS)
    optimizer = tf.keras.optimizers.Adam(lr_sched)
    gen       = tf_data_generator(BATCH_SIZE, MISSING_CONST, N_MIN, N_MAX, Q_MIN, Q_MAX)

    best_val, best_weights, no_improve, stopped_at = np.inf, None, 0, MAX_EPOCHS

    for epoch in range(MAX_EPOCHS):
        with tf.GradientTape() as tape:
            train_loss = 0.0
            for _ in range(STEPS_PER_EPOCH):
                inp_seq, Q_emp, Sigma_true, T, Sigma_hat_diag, _ = next(gen)
                inp = tf.gather(tf.cast(inp_seq, tf.float32), feat_cols, axis=2)
                lp  = model(inp, training=True)
                Sp  = _reconstruct_cov(lp,
                                       tf.cast(Q_emp, tf.float32),
                                       tf.cast(Sigma_hat_diag, tf.float32))
                train_loss += (tf_loss_function_mat(
                    tf.cast(Sigma_true, tf.float32), Sp, T) / STEPS_PER_EPOCH)

        grads = tape.gradient(train_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        val_loss = _batch_loss(model, val_batches, feat_cols)

        if val_loss < best_val * (1 - 1e-4):
            best_val   = val_loss
            best_weights = model.get_weights()
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            print(f"    e{epoch+1:3d}  train={float(train_loss):.5f}  "
                  f"val={val_loss:.5f}  best={best_val:.5f}  p={no_improve}/{PATIENCE}")

        if no_improve >= PATIENCE:
            stopped_at = epoch + 1
            print(f"    EarlyStop @ epoch {stopped_at}")
            break

    if best_weights is not None:
        model.set_weights(best_weights)
    return model, stopped_at


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(results):
    print("\n" + "=" * 72)
    print("SUMMARY — mean ± std across seeds")
    print("=" * 72)
    best_loss = min(np.mean([s["test_loss"] for s in r["seeds"]]) for r in results.values())

    for cfg, r in results.items():
        losses  = [s["test_loss"] for s in r["seeds"]]
        sp      = np.array([s["spearman_q"] for s in r["seeds"]])
        pi      = np.array([s["perm_imp"]   for s in r["seeds"]])
        stopped = [s["stopped_epoch"] for s in r["seeds"]]
        m, s    = np.mean(losses), np.std(losses)
        gap     = (m - best_loss) / best_loss * 100
        print(f"\n  {cfg}: {r['feat_names']}")
        print(f"    loss = {m:.5f} ± {s:.5f}  (gap from best: {gap:+.2f}%)")
        print(f"    stopped @ {stopped}")
        print(f"    Spearman Q1-Q4: " +
              "  ".join(f"{np.mean(sp[:, i]):.3f}±{np.std(sp[:, i]):.3f}" for i in range(4)))
        if pi.size > 0:
            print(f"    PermImp: " +
                  "  ".join(f"{r['feat_names'][j]}={np.mean(pi[:, j]):+.3f}" for j in range(pi.shape[1])))

    threshold = best_loss * 1.01
    viable = []
    for cfg, r in results.items():
        mean_loss = np.mean([s["test_loss"] for s in r["seeds"]])
        if mean_loss <= threshold:
            pi = np.array([s["perm_imp"] for s in r["seeds"]])
            all_nonzero = bool(np.all(np.mean(pi, axis=0) > 0))
            viable.append((cfg, len(r["feat_cols"]), all_nonzero, mean_loss))
    if viable:
        viable.sort(key=lambda x: (not x[2], x[1], x[3]))
        best_cfg = viable[0][0]
        print(f"\n  RECOMMENDATION: {best_cfg} — {results[best_cfg]['feat_names']}")
        print(f"    Minimal config within 1% of best with all features having positive perm importance.")
    print("=" * 72)


def _plot_results(results):
    cfgs   = list(results.keys())
    colors = ["#2c7bb6", "#1a9641", "#d7191c", "#984ea3"]
    n      = len(cfgs)

    # Figure 1: loss + Spearman
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    means = [np.mean([s["test_loss"] for s in results[c]["seeds"]]) for c in cfgs]
    stds  = [np.std( [s["test_loss"] for s in results[c]["seeds"]]) for c in cfgs]
    best  = int(np.argmin(means))
    bars  = ax.bar(cfgs, means, yerr=stds, color=colors[:n], capsize=5, width=0.55)
    bars[best].set_edgecolor("gold"); bars[best].set_linewidth(2.5)
    ax.set_ylabel("Frobenius test loss (lower = better)")
    ax.set_title(f"Test loss by feature config\n(mean ± std, seeds={SEEDS})")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + max(stds) * 0.15, f"{m:.4f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    x, w = np.arange(4), 0.2
    for ci, cfg in enumerate(cfgs):
        sp = np.array([s["spearman_q"] for s in results[cfg]["seeds"]])
        m, e = np.mean(sp, axis=0), np.std(sp, axis=0)
        ax.bar(x + ci * w, m, w, yerr=e, label=cfg, color=colors[ci], capsize=3)
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(["Q1\n(low λ_emp)", "Q2", "Q3", "Q4\n(high λ_emp)"])
    ax.set_ylabel("Spearman ρ(λ_pred, λ_true)")
    ax.set_title("Ranking accuracy by λ_emp quartile\n(mean ± std, 3 seeds)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    fig.suptitle(f"Feature ablation — BiGRU eigenvalue cleaner  "
                 f"(N∈[{N_MIN},{N_MAX}], q∈[{Q_MIN},{Q_MAX}], patience={PATIENCE})", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "ablation_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {os.path.join(SAVE_DIR, 'ablation_summary.png')}")

    # Figure 2: permutation importance per config
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ci, (cfg, ax) in enumerate(zip(cfgs, axes)):
        pi = np.array([s["perm_imp"] for s in results[cfg]["seeds"]])
        m, e = np.mean(pi, axis=0), np.std(pi, axis=0)
        fn   = results[cfg]["feat_names"]
        bar_cols = ["#d7191c" if v < 0 else "#2c7bb6" for v in m]
        ax.barh(fn, m, xerr=e, color=bar_cols, capsize=3)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{cfg}: permutation importance\n(mean ± std, {len(SEEDS)} seeds)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("Relative loss change\n(positive = feature helps)", fontsize=8)
    fig.suptitle("Permutation importance by config", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "ablation_perm_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {os.path.join(SAVE_DIR, 'ablation_perm_importance.png')}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # Resumable: skip configs already saved
    partial_path = os.path.join(SAVE_DIR, "results.json")
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            all_results = json.load(f)
        print(f"Resuming — already done: {list(all_results.keys())}")
    else:
        all_results = {}

    # Fixed val + test pools shared across all configs/seeds
    print("\n=== Generating fixed val / test pools (seed=42) ===")
    np.random.seed(42)
    tf.random.set_seed(42)
    gen_pool = tf_data_generator(BATCH_SIZE, MISSING_CONST, N_MIN, N_MAX, Q_MIN, Q_MAX)

    def _collect(n_batches):
        pool = []
        for _ in range(n_batches):
            inp_seq, Q_emp, Sigma_true, T, Sigma_hat_diag, _ = next(gen_pool)
            pool.append({
                "input_seq":      tf.cast(inp_seq, tf.float32),
                "Q_emp":          tf.cast(Q_emp, tf.float32),
                "Sigma_true":     tf.cast(Sigma_true, tf.float32),
                "Sigma_hat_diag": tf.cast(Sigma_hat_diag, tf.float32),
                "T": T,
            })
        return pool

    val_batches  = _collect(N_VAL_BATCHES)
    test_batches = _collect(N_TEST_BATCHES)
    print(f"  val: {N_VAL_BATCHES} batches   test: {N_TEST_BATCHES} batches\n")

    for cfg_name, feat_cols in CONFIGS.items():
        if cfg_name in all_results:
            print(f"Skipping {cfg_name} (already in results.json)")
            continue

        feat_names = [ALL_FEAT_NAMES[i] for i in feat_cols]
        print(f"\n{'='*64}")
        print(f"Config {cfg_name} ({len(feat_cols)} features): {feat_names}")
        print(f"{'='*64}")

        cfg_res = {"feat_cols": feat_cols, "feat_names": feat_names, "seeds": []}

        for seed in SEEDS:
            print(f"\n  Seed {seed}")
            t0 = time.time()
            model, stopped = train_one(feat_cols, seed, val_batches)
            elapsed = time.time() - t0

            loss, spearman_q, perm_imp = compute_metrics(model, test_batches, feat_cols)

            cfg_res["seeds"].append({
                "test_loss":     loss,
                "stopped_epoch": stopped,
                "elapsed_s":     round(elapsed, 1),
                "spearman_q":    spearman_q,
                "perm_imp":      perm_imp,
            })

            print(f"  → loss={loss:.5f}  stopped={stopped}  time={elapsed:.0f}s")
            print(f"     Spearman: " + "  ".join(f"Q{i+1}={v:.3f}" for i, v in enumerate(spearman_q)))
            print(f"     PermImp: " + "  ".join(f"{nm}={v:+.3f}" for nm, v in zip(feat_names, perm_imp)))

        all_results[cfg_name] = cfg_res
        with open(partial_path, "w") as f:
            json.dump(all_results, f, indent=2)

    _print_summary(all_results)
    _plot_results(all_results)
    print(f"\nAll results saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
