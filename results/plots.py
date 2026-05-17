import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_relative_improvement(qs, frob_cov_loss, save_path="images/relative_improvement.png"):
    """
    Relative improvement of NN over QIS, Emp, and Oracle ceiling as a function of q.

    frob_cov_loss : ndarray, shape (len(Ns), len(qs), 5)
        Columns: [NN_miss=0, NN_nomiss=1, Emp_miss=2, QIS=3, Oracle=4].
    """
    qs_arr = np.array(qs)
    means  = np.mean(frob_cov_loss, axis=0)  # (len(qs), 5)

    gain_qis    = (means[:, 3] - means[:, 0]) / means[:, 3] * 100
    gain_emp    = (means[:, 2] - means[:, 0]) / means[:, 2] * 100
    gain_oracle = (means[:, 2] - means[:, 4]) / means[:, 2] * 100  # oracle ceiling

    std_qis = np.std(
        (frob_cov_loss[:, :, 3] - frob_cov_loss[:, :, 0]) / frob_cov_loss[:, :, 3] * 100,
        axis=0,
    )
    std_emp = np.std(
        (frob_cov_loss[:, :, 2] - frob_cov_loss[:, :, 0]) / frob_cov_loss[:, :, 2] * 100,
        axis=0,
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(qs_arr, gain_qis, marker='o', markersize=4, linewidth=2,
            color='darkorange', label='NN gain over QIS')
    ax.fill_between(qs_arr, gain_qis - std_qis, gain_qis + std_qis,
                    alpha=0.15, color='darkorange')

    ax.plot(qs_arr, gain_emp, marker='o', markersize=4, linewidth=2,
            color='steelblue', label='NN gain over Emp')
    ax.fill_between(qs_arr, gain_emp - std_emp, gain_emp + std_emp,
                    alpha=0.15, color='steelblue')

    ax.plot(qs_arr, gain_oracle, marker='s', markersize=3, linewidth=1.5,
            color='crimson', linestyle='--', label='Oracle ceiling (best eigenvalue cleaner)')

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(x=1, color='grey',  linewidth=1,   linestyle=':', label='q = 1')
    ax.set_xlabel('q = N / T')
    ax.set_ylabel('Relative improvement (%)')
    ax.set_title('NN relative gain over baselines — covariance Frobenius loss')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_q_curve(qs, frob_corr_loss, frob_cov_loss, save_path="images/qcurve_with_bands.png"):
    """
    Two-panel q-curve (correlation / covariance) with ±1 std bands across N values.

    Both arrays share the same column layout:
        [NN_miss=0, NN_nomiss=1, Emp_miss=2, QIS=3, Oracle=4]
    """
    qs_arr = np.array(qs)

    means_corr = np.mean(frob_corr_loss, axis=0)
    std_corr   = np.std(frob_corr_loss,  axis=0)
    means_cov  = np.mean(frob_cov_loss,  axis=0)
    std_cov    = np.std(frob_cov_loss,   axis=0)

    # estimators in display order (worst → best), shared across both panels
    ESTIMATORS = [
        # (col, color, label, linestyle)
        (2, 'dimgrey',    'Emp (miss)',   '-'),
        (3, 'darkorange', 'QIS',          '-'),
        (0, 'steelblue',  'NN (miss)',    '-'),
        (1, 'royalblue',  'NN (no-miss)', '--'),
        (4, 'crimson',    'Oracle',       '--'),
    ]

    def _band(ax, x, y, s, color, label, ls):
        ax.plot(x, y, marker='o', markersize=2, linewidth=1.8,
                color=color, linestyle=ls, label=label)
        ax.fill_between(x, y - s, y + s, alpha=0.10, color=color)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for col, color, label, ls in ESTIMATORS:
        _band(ax1, qs_arr, means_corr[:, col], std_corr[:, col], color, label, ls)
        _band(ax2, qs_arr, means_cov[:, col],  std_cov[:, col],  color, label, ls)

    for ax, title, ylabel in [
        (ax1, 'Correlation matrix', 'Frobenius loss'),
        (ax2, 'Covariance matrix',  'Frobenius loss'),
    ]:
        ax.axvline(x=1, color='black', linestyle=':', linewidth=1, alpha=0.4)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(alpha=0.25)

    ax2.set_xlabel('q = N / T')
    plt.suptitle('Frobenius loss vs concentration ratio q', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_q_curve_realdata(qs, port_var_results, save_path="images/qcurve_realdata.png"):
    """
    Single-panel realized portfolio variance vs q for real data.

    Parameters
    ----------
    qs : list[float]
        q = N / T_in values used in the evaluation grid.
    port_var_results : ndarray, shape (len(qs), steps, 3)
        Portfolio variance per step. Columns: [NN, Emp_pairwise, QIS].
    """
    qs_arr = np.array(qs)
    means  = np.mean(port_var_results, axis=1)  # (len(qs), 3)
    stds   = np.std(port_var_results,  axis=1)

    def _band(ax, x, y, s, color, label, ls='-'):
        ax.plot(x, y, marker='o', markersize=3, linewidth=2,
                color=color, linestyle=ls, label=label)
        ax.fill_between(x, y - s, y + s, alpha=0.15, color=color)

    fig, ax = plt.subplots(figsize=(11, 5))
    _band(ax, qs_arr, means[:, 0], stds[:, 0], 'steelblue',  'NN miss')
    _band(ax, qs_arr, means[:, 2], stds[:, 2], 'darkorange', 'QIS')
    _band(ax, qs_arr, means[:, 1], stds[:, 1], 'dimgrey',    'Emp miss')
    ax.axvline(x=1, color='black', linestyle=':', linewidth=1, alpha=0.5, label='q = 1')
    ax.set_yscale('log')
    ax.set_xlabel('q = N / T')
    ax.set_ylabel('Realized portfolio variance (log scale)')
    ax.set_title('Min-variance portfolio realized variance vs q  (real data)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which='both')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_relative_improvement_realdata(
    qs, port_var_results, save_path="images/relative_improvement_realdata.png"
):
    """
    Relative improvement of NN over QIS and empirical pairwise vs q (real data).

    Parameters
    ----------
    qs : list[float]
    port_var_results : ndarray, shape (len(qs), steps, 3)
        Columns: [NN, Emp_pairwise, QIS].
    """
    qs_arr = np.array(qs)
    means  = np.mean(port_var_results, axis=1)  # (len(qs), 3)

    gain_over_qis = (means[:, 2] - means[:, 0]) / means[:, 2] * 100
    gain_over_emp = (means[:, 1] - means[:, 0]) / means[:, 1] * 100

    std_qis = np.std(
        (port_var_results[:, :, 2] - port_var_results[:, :, 0])
        / port_var_results[:, :, 2] * 100,
        axis=1,
    )
    std_emp = np.std(
        (port_var_results[:, :, 1] - port_var_results[:, :, 0])
        / port_var_results[:, :, 1] * 100,
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(qs_arr, gain_over_qis, marker='o', markersize=4, linewidth=2,
            color='darkorange', label='NN gain over QIS')
    ax.fill_between(qs_arr, gain_over_qis - std_qis, gain_over_qis + std_qis,
                    alpha=0.15, color='darkorange')
    ax.plot(qs_arr, gain_over_emp, marker='o', markersize=4, linewidth=2,
            color='steelblue', label='NN gain over pairwise MLE')
    ax.fill_between(qs_arr, gain_over_emp - std_emp, gain_over_emp + std_emp,
                    alpha=0.15, color='steelblue')
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    ax.axvline(x=1, color='grey', linewidth=1, linestyle=':', label='q = 1')
    ax.set_xlabel('q = N / T')
    ax.set_ylabel('Relative improvement in portfolio variance (%)')
    ax.set_title('NN relative gain over baselines — realized portfolio variance  (real data)')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_spectral_cleaning_multiregime(
    model, q_values=(0.5, 0.9, 1.3), N_ex=100,
    save_path="images/spectral_cleaning_regimes.png",
):
    """
    3-panel eigenvalue spectrum across concentration regimes.
    Each panel shows Emp / QIS / NN / True for a fixed (N, q).
    The T_sync annotation reveals why QIS degrades at high q.
    """
    from data.dataloader import tf_data_generator
    from estimator.QIS import tf_QIS_batched

    fig, axes = plt.subplots(1, len(q_values), figsize=(5 * len(q_values), 5), sharey=False)

    for ax, q_ex in zip(axes, q_values):
        np.random.seed(42)
        tf.random.set_seed(42)

        gen = tf_data_generator(
            batch_size=1, missing_constant=2,
            N_min=N_ex, N_max=N_ex, q_min=q_ex, q_max=q_ex,
        )
        input_seq, _, Sigma_true, T, _, R_hat = next(gen)

        lam_emp  = input_seq[0, :, 0].numpy()
        lam_pred = np.sort(model(input_seq, training=False)[0].numpy())

        Sigma_true_np = Sigma_true[0].numpy().astype(np.float64)
        std_true = np.sqrt(np.maximum(np.diag(Sigma_true_np), 1e-12))
        Corr_true = Sigma_true_np / np.outer(std_true, std_true)
        lam_true = np.linalg.eigvalsh(Corr_true)

        T_int = int(T)
        born  = T_int // 2
        R_np  = R_hat[0].numpy()
        R_sync = tf.constant(R_np[:, -born:][None], dtype=tf.float32)
        Sigma_qis = tf_QIS_batched(R_sync)[0].numpy().astype(np.float64)
        std_qis = np.sqrt(np.maximum(np.diag(Sigma_qis), 1e-12))
        Corr_qis = Sigma_qis / np.outer(std_qis, std_qis)
        lam_qis  = np.linalg.eigvalsh(Corr_qis)

        j = np.arange(1, N_ex + 1)
        ax.plot(j, lam_emp,  color='dimgrey',    linewidth=1.2, alpha=0.6, label='Emp (noisy)')
        ax.plot(j, lam_qis,  color='darkorange', linewidth=1.5, linestyle='--', label='QIS')
        ax.plot(j, lam_pred, color='steelblue',  linewidth=2.0, label='NN (cleaned)')
        ax.plot(j, lam_true, color='crimson',    linewidth=1.8, linestyle=':', label='True')

        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_xlabel('Eigenvalue rank j')
        ax.set_title(f'q = {q_ex:.1f}   (T = {T_int},  T_sync = {born})', fontsize=11)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel('Eigenvalue λ_j')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle('Spectral cleaning across concentration regimes', fontsize=13, y=1.10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_q_curve_clean(qs, frob_cov_loss, save_path="images/qcurve_clean.png"):
    """
    Single-panel covariance Frobenius loss vs q with ±1 std bands.

    frob_cov_loss : (len(Ns), len(qs), 5)
        Cols: [NN_miss=0, NN_nomiss=1, Emp_miss=2, QIS=3, Oracle=4]
    """
    qs_arr = np.array(qs)
    means  = np.mean(frob_cov_loss, axis=0)
    stds   = np.std(frob_cov_loss,  axis=0)

    ESTIMATORS = [
        (2, 'dimgrey',    'Emp (pairwise)', '-'),
        (3, 'darkorange', 'QIS',            '-'),
        (0, 'steelblue',  'NN',             '-'),
        (4, 'crimson',    'Oracle',         '--'),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    for col, color, label, ls in ESTIMATORS:
        y = means[:, col]
        s = stds[:, col]
        valid = np.isfinite(y)
        ax.plot(qs_arr[valid], y[valid], marker='o', markersize=4, linewidth=2,
                color=color, linestyle=ls, label=label)
        ax.fill_between(qs_arr[valid], (y - s)[valid], (y + s)[valid],
                        alpha=0.08, color=color)

    ax.axvline(x=1, color='grey', linewidth=1, linestyle=':', alpha=0.7)
    ax.set_xlabel('q = N / T', fontsize=12)
    ax.set_ylabel('Frobenius loss (covariance)', fontsize=12)
    ax.set_title('Covariance estimation error vs concentration ratio', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_nn_advantage(qs, frob_cov_loss, save_path="images/nn_advantage.png"):
    """
    NN % gain over QIS (with ±1 std band and green fill where positive) and Emp.

    frob_cov_loss : (len(Ns), len(qs), 5)
        Cols: [NN_miss=0, NN_nomiss=1, Emp_miss=2, QIS=3, Oracle=4]
    """
    qs_arr = np.array(qs)
    means  = np.mean(frob_cov_loss, axis=0)

    gain_qis = (means[:, 3] - means[:, 0]) / means[:, 3] * 100
    gain_emp = (means[:, 2] - means[:, 0]) / means[:, 2] * 100
    std_qis  = np.std(
        (frob_cov_loss[:, :, 3] - frob_cov_loss[:, :, 0]) / frob_cov_loss[:, :, 3] * 100,
        axis=0,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    valid = np.isfinite(gain_qis)
    ax.plot(qs_arr[valid], gain_qis[valid], marker='o', markersize=4, linewidth=2.5,
            color='darkorange', label='NN gain over QIS')
    ax.fill_between(qs_arr[valid], (gain_qis - std_qis)[valid], (gain_qis + std_qis)[valid],
                    alpha=0.15, color='darkorange')

    positive = valid & (gain_qis > 0)
    if positive.any():
        ax.fill_between(qs_arr, np.where(valid, gain_qis, 0), 0,
                        where=positive, alpha=0.12, color='green', label='NN beats QIS')

    ax.plot(qs_arr, gain_emp, marker='s', markersize=3, linewidth=1.5,
            color='dimgrey', linestyle='--', alpha=0.7, label='NN gain over Emp')

    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.axvline(x=1, color='grey', linewidth=1, linestyle=':', alpha=0.7)

    # annotate crossover q*
    cross_idxs = np.where(np.array(gain_qis) > 0)[0]
    if len(cross_idxs):
        q_cross = qs_arr[cross_idxs[0]]
        ax.axvline(x=q_cross, color='green', linewidth=1.5, linestyle='--', alpha=0.8,
                   label=f'Crossover q* = {q_cross:.2f}')

    # annotate max NN gain
    valid = np.isfinite(gain_qis)
    if valid.any():
        peak_idx = np.argmax(gain_qis[valid])
        peak_idx_global = np.where(valid)[0][peak_idx]
        ax.annotate(
            f'+{gain_qis[peak_idx_global]:.1f}%',
            xy=(qs_arr[peak_idx_global], gain_qis[peak_idx_global]),
            xytext=(qs_arr[peak_idx_global] - 0.18, gain_qis[peak_idx_global] + 1.5),
            fontsize=9, color='darkorange', fontweight='bold',
        )

    ax.set_xlabel('q = N / T', fontsize=12)
    ax.set_ylabel('Relative improvement (%)', fontsize=12)
    ax.set_title('NN advantage over baselines — covariance Frobenius loss', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_eigenvalue_cleaning(model, N_ex=100, q_ex=0.8, save_path="images/eigenvalue_cleaning.png"):
    """
    For a single synthetic example, plot empirical / NN-predicted / QIS / true eigenvalues.

    Parameters
    ----------
    model : BiGRUSpectralDenoiserTensorFlow
        Trained model.
    N_ex : int
        Number of assets.
    q_ex : float
        Concentration ratio N/T.
    """
    from data.dataloader import tf_data_generator
    from estimator.QIS import tf_QIS_batched

    np.random.seed(42)
    tf.random.set_seed(42)

    gen = tf_data_generator(
        batch_size=1, missing_constant=2,
        N_min=N_ex, N_max=N_ex,
        q_min=q_ex, q_max=q_ex,
    )
    input_seq, _, Sigma_true, T, _, R_hat = next(gen)

    # Empirical eigenvalues — feature 0 in input_seq, already sorted ascending
    lam_emp = input_seq[0, :, 0].numpy()

    # NN predicted eigenvalues
    lam_pred = np.sort(model(input_seq, training=False)[0].numpy())

    # True eigenvalues from the ground-truth correlation matrix
    Sigma_true_np = Sigma_true[0].numpy().astype(np.float64)
    std_true = np.sqrt(np.maximum(np.diag(Sigma_true_np), 1e-12))
    Corr_true = Sigma_true_np / np.outer(std_true, std_true)
    lam_true = np.linalg.eigvalsh(Corr_true)

    # QIS on the guaranteed fully-observed window: last T//2 steps have no NaNs
    # (missing_constant=2 ensures every asset is observed for at least T//2 steps)
    R_np = R_hat[0].numpy()
    T_int = int(T)
    born = T_int // 2
    R_sync = tf.constant(R_np[:, -born:][None], dtype=tf.float32)
    Sigma_qis = tf_QIS_batched(R_sync)[0].numpy().astype(np.float64)
    std_qis = np.sqrt(np.maximum(np.diag(Sigma_qis), 1e-12))
    Corr_qis = Sigma_qis / np.outer(std_qis, std_qis)
    lam_qis = np.linalg.eigvalsh(Corr_qis)

    j = np.arange(1, N_ex + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(j, lam_emp,  color='lightgrey',  linewidth=1.5, zorder=1,
            label='Empirical pairwise (noisy)')
    ax.plot(j, lam_pred, color='steelblue',  linewidth=2.0, zorder=3,
            label='NN predicted (cleaned)')
    ax.plot(j, lam_qis,  color='darkorange', linewidth=1.5, linestyle='--', zorder=2,
            label=f'QIS (sync window, T_sync={born})')
    ax.plot(j, lam_true, color='crimson',    linewidth=2.0, zorder=4,
            label='True eigenvalues')

    ax.axhline(y=0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Eigenvalue rank $j$')
    ax.set_ylabel('Eigenvalue $\\lambda_j$')
    ax.set_title(f'Spectral cleaning — N={N_ex}, q={q_ex:.1f}, T={int(T)}')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_eigenvalue_profile(profiles, save_path="images/eigenvalue_profile.png"):
    """
    Mean eigenvalue vs rank j: empirical / NN-predicted / true.

    profiles : dict with keys 'emp', 'pred', 'true', each shape (N,).
               Returned by accumulate_eigenvalue_data().
    """
    N = len(profiles['true'])
    j = np.arange(1, N + 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(j, profiles['emp'],  color='dimgrey',   linewidth=1.5, alpha=0.8,
            label='Empirical pairwise (noisy)')
    ax.plot(j, profiles['pred'], color='steelblue', linewidth=2.0,
            label='NN (cleaned)')
    ax.plot(j, profiles['true'], color='crimson',   linewidth=2.0, linestyle='--',
            label='True')
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Eigenvalue rank j', fontsize=12)
    ax.set_ylabel('Mean eigenvalue λ_j', fontsize=12)
    ax.set_title('Mean eigenvalue profile across N and q — empirical vs NN vs true', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
