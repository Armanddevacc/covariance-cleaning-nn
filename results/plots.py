import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_relative_improvement(qs, frob_cov_loss, save_path="images/relative_improvement.png"):
    """
    Relative improvement of NN over QIS and pairwise MLE as a function of q.

    Parameters
    ----------
    qs : list[float]
        q values used in the evaluation grid.
    frob_cov_loss : ndarray, shape (len(Ns), len(qs), 5)
        Covariance losses. Columns: [NN_miss, NN_nomiss, Emp_miss, Emp_nomiss, QIS].
    """
    qs_arr = np.array(qs)

    means = np.mean(frob_cov_loss, axis=0)  # (len(qs), 5)

    gain_over_qis = (means[:, 4] - means[:, 0]) / means[:, 4] * 100
    gain_over_emp = (means[:, 2] - means[:, 0]) / means[:, 2] * 100

    # spread across N values — indicates how consistent the advantage is
    std_qis = np.std(
        (frob_cov_loss[:, :, 4] - frob_cov_loss[:, :, 0]) / frob_cov_loss[:, :, 4] * 100,
        axis=0,
    )
    std_emp = np.std(
        (frob_cov_loss[:, :, 2] - frob_cov_loss[:, :, 0]) / frob_cov_loss[:, :, 2] * 100,
        axis=0,
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
    ax.axvline(x=1, color='grey', linewidth=1, linestyle=':', label='q = 1  (N = T)')
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

    Parameters
    ----------
    qs : list[float]
    frob_corr_loss : ndarray, shape (len(Ns), len(qs), 4)
        Columns: [NN_miss, NN_nomiss, Emp_miss, Emp_nomiss].
    frob_cov_loss : ndarray, shape (len(Ns), len(qs), 5)
        Columns: [NN_miss, NN_nomiss, Emp_miss, Emp_nomiss, QIS].
    """
    qs_arr = np.array(qs)

    means_corr = np.mean(frob_corr_loss, axis=0)
    std_corr   = np.std(frob_corr_loss,  axis=0)
    means_cov  = np.mean(frob_cov_loss,  axis=0)
    std_cov    = np.std(frob_cov_loss,   axis=0)

    def _band(ax, x, y, s, color, label, ls='-'):
        ax.plot(x, y, marker='o', markersize=3, linewidth=2,
                color=color, linestyle=ls, label=label)
        ax.fill_between(x, y - s, y + s, alpha=0.15, color=color)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # --- Correlation ---
    _band(ax1, qs_arr, means_corr[:, 0], std_corr[:, 0], 'steelblue',  'NN miss')
    _band(ax1, qs_arr, means_corr[:, 1], std_corr[:, 1], 'royalblue',  'NN no-miss', ls='--')
    _band(ax1, qs_arr, means_corr[:, 2], std_corr[:, 2], 'dimgrey',    'Emp miss')
    ax1.axvline(x=1, color='black', linestyle=':', linewidth=1, alpha=0.5, label='q = 1')
    ax1.set_ylabel('Frobenius loss')
    ax1.set_title('Correlation matrix')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    # --- Covariance ---
    _band(ax2, qs_arr, means_cov[:, 0], std_cov[:, 0], 'steelblue',  'NN miss')
    _band(ax2, qs_arr, means_cov[:, 4], std_cov[:, 4], 'darkorange', 'QIS')
    _band(ax2, qs_arr, means_cov[:, 2], std_cov[:, 2], 'dimgrey',    'Emp miss')
    ax2.axvline(x=1, color='black', linestyle=':', linewidth=1, alpha=0.5, label='q = 1')
    ax2.set_xlabel('q = N / T')
    ax2.set_ylabel('Frobenius loss')
    ax2.set_title('Covariance matrix')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    plt.suptitle('Frobenius loss vs concentration ratio q', fontsize=14)
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
