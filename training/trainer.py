import numpy as np
import tensorflow as tf
from models.losses import tf_loss_function_mat, tf_variance_loss


class Trainer_tf:

    def __init__(
        self,
        model,
        data_generator,
        batch_size,
        epochs,
        missing_constant,
        N_min,
        N_max,
        q_min,
        q_max,
        lr=1e-4,
        patience=None,
        val_steps=20,
    ):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.missing_constant = missing_constant
        self.N_min = N_min
        self.N_max = N_max
        self.q_min = q_min
        self.q_max = q_max
        self.patience = patience
        self.val_steps = val_steps
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr, decay_steps=self.epochs
        )
        self.optimizer = tf.keras.optimizers.Adam(lr_schedule)
        self.loss_function = tf_loss_function_mat
        gen_args = (
            self.batch_size,
            self.missing_constant,
            self.N_min,
            self.N_max,
            self.q_min,
            self.q_max,
        )
        self.data_generator = data_generator(*gen_args)
        # Pre-collect a fixed val pool once so val loss is comparable across epochs
        if patience is not None:
            _val_gen = data_generator(*gen_args)
            self._val_pool = []
            print(f"Pre-collecting {val_steps} val batches...")
            for _ in range(val_steps):
                inp, Q, S_true, T, S_diag, _ = next(_val_gen)
                self._val_pool.append((
                    tf.cast(inp, tf.float32),
                    tf.cast(Q, tf.float32),
                    tf.cast(S_true, tf.float32),
                    T,
                    tf.cast(S_diag, tf.float32),
                ))
        else:
            self._val_pool = None
        self.loss_history = []

    def train(self, steps_per_epoch):
        best_val, best_weights, no_improve = np.inf, None, 0

        for epoch in range(self.epochs):
            loss = self._train_step(steps_per_epoch)
            self.loss_history.append(float(loss))

            if self.patience is not None:
                val_loss = self._eval_val()
                if val_loss < best_val * (1 - 1e-4):
                    best_val = val_loss
                    best_weights = self.model.get_weights()
                    no_improve = 0
                else:
                    no_improve += 1
                print(
                    f"Epoch {epoch+1}, Loss: {float(loss):.6f}, Val: {val_loss:.6f}, Best: {best_val:.6f}, P: {no_improve}/{self.patience}"
                )
                if no_improve >= self.patience:
                    print(f"EarlyStop @ epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}, Loss: {float(loss):.6f}")

        if best_weights is not None:
            self.model.set_weights(best_weights)
            print(f"Restored best weights (val={best_val:.6f})")

        return self.loss_history

    def _eval_val(self):
        losses = []
        for inp, Q_emp, Sigma_true, T, _ in self._val_pool:
            lam_pred = self.model(inp, training=False)
            Corr_pred = tf.matmul(
                tf.matmul(Q_emp, tf.linalg.diag(lam_pred)), Q_emp, transpose_b=True
            )
            Corr_true = self._to_corr(Sigma_true)
            losses.append(float(self.loss_function(Corr_true, Corr_pred, T)))
        return float(np.mean(losses))

    def _train_step(self, steps_per_epoch):
        with tf.GradientTape() as tape:
            loss = 0
            for _ in range(steps_per_epoch):
                input_seq, Q_emp, Sigma_true, T, Sigma_hat_diag, R_hat = next(
                    self.data_generator
                )
                Q_emp        = tf.cast(Q_emp,        tf.float32)
                Sigma_true   = tf.cast(Sigma_true,   tf.float32)
                lam_pred     = self.model(input_seq, training=True)
                Corr_pred    = tf.matmul(
                    tf.matmul(Q_emp, tf.linalg.diag(lam_pred)), Q_emp, transpose_b=True
                )
                Corr_true    = self._to_corr(Sigma_true)
                loss += self.loss_function(Corr_true, Corr_pred, T) / steps_per_epoch

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    @staticmethod
    def _to_corr(Sigma):
        """Normalise a batch of covariance matrices to correlation matrices."""
        std = tf.sqrt(tf.maximum(tf.linalg.diag_part(Sigma), 1e-12))
        return Sigma / (std[:, :, None] * std[:, None, :])


from estimator.MLE import tf_cov_pairwise, tf_cov_pairwise_mask


class Trainer_real_data_tf:
    def __init__(
        self,
        model,
        dataset,
        batch_size,
        epochs,
        lr=1e-4,
        patience=None,
        val_dataset=None,
        val_steps=20,
        no_miss=False,
    ):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_steps = val_steps
        self.no_miss = no_miss
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr, decay_steps=epochs
        )
        self.optimizer = tf.keras.optimizers.Adam(lr_schedule, clipnorm=1.0)
        self.dataset = dataset
        self._dataset_iter = iter(dataset)
        # Pre-collect a fixed val pool once so val loss is comparable across epochs
        if patience is not None and val_dataset is not None:
            print(f"Pre-collecting {val_steps} val batches...")
            _val_iter = iter(val_dataset)
            self._val_pool = []
            for _ in range(val_steps):
                (rin, mask), rout_nan = next(_val_iter)
                rout = tf.where(tf.math.is_nan(rout_nan), tf.zeros_like(rout_nan), rout_nan)
                rout = tf.cast(rout, tf.float32)
                input_seq, Q_emp, _, _ = self._construct_input_seq(rin, mask)
                self._val_pool.append((input_seq, tf.cast(Q_emp, tf.float32), rout))
        else:
            self._val_pool = None
        self.loss_history = []

    def train(self, step_per_epoch):
        best_val, best_weights, no_improve = np.inf, None, 0

        for epoch in range(self.epochs):
            loss = self._train_set(step_per_epoch)
            self.loss_history.append(loss)

            if self.patience is not None and self._val_pool is not None:
                val_loss = self._eval_val()
                if val_loss < best_val * (1 - 1e-4):
                    best_val = val_loss
                    best_weights = self.model.get_weights()
                    no_improve = 0
                else:
                    no_improve += 1
                print(
                    f"Epoch {epoch+1}, Loss: {loss:.6f}, Val: {val_loss:.6f}, Best: {best_val:.6f}, P: {no_improve}/{self.patience}"
                )
                if no_improve >= self.patience:
                    print(f"EarlyStop @ epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}, Loss: {loss:.6f}")

        if best_weights is not None:
            self.model.set_weights(best_weights)
            print(f"Restored best weights (val={best_val:.6f})")

        return self.loss_history

    def _eval_val(self):
        losses = []
        for input_seq, Q_emp, rout in self._val_pool:
            lam_pred = self.model(input_seq, training=False)
            losses.append(float(tf_variance_loss(lam_pred, Q_emp, rout)))
        return float(np.mean(losses))

    def _train_set(self, steps_per_epoch):
        # Collect batches first, then accumulate gradients in a single tape
        # (mirrors Trainer_tf._train_step — one optimizer step per epoch).
        batches = []
        for _ in range(steps_per_epoch):
            (rin, mask), rout_nan = next(self._dataset_iter)
            rout = tf.where(tf.math.is_nan(rout_nan), tf.zeros_like(rout_nan), rout_nan)
            rout = tf.cast(rout, tf.float32)
            input_seq, Q_emp, _, _ = self._construct_input_seq(rin, mask)
            batches.append((input_seq, tf.cast(Q_emp, tf.float32), rout))

        with tf.GradientTape() as tape:
            loss = 0.0
            for input_seq, Q_emp, rout in batches:
                lam_pred = self.model(input_seq, training=True)
                loss += tf_variance_loss(lam_pred, Q_emp, rout) / steps_per_epoch

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return float(loss)

    def _construct_input_seq(self, rin, mask):
        B, N, T = tf.shape(rin)

        # no_miss mode: treat every stock as fully observed from t=0
        if self.no_miss:
            mask = tf.zeros_like(mask)

        # mask: True where MISSING — invert for pairwise cov
        observed = ~mask
        Sigma_hat = tf_cov_pairwise_mask(rin, observed)

        Sigma_hat_diag = tf.linalg.diag_part(Sigma_hat)
        Sigma_hat_diag = tf.maximum(Sigma_hat_diag, 1e-8)
        std_pred = tf.sqrt(Sigma_hat_diag)

        corr_hat = Sigma_hat / (std_pred[:, :, None] * std_pred[:, None, :])
        corr_hat = tf.where(tf.math.is_finite(corr_hat), corr_hat, tf.zeros_like(corr_hat))

        eye = tf.eye(N, batch_shape=[B], dtype=corr_hat.dtype)
        corr_hat = corr_hat - tf.linalg.diag(tf.linalg.diag_part(corr_hat)) + eye
        corr_hat = 0.5 * (corr_hat + tf.transpose(corr_hat, perm=[0, 2, 1]))
        corr_hat = corr_hat + 1e-6 * eye

        eigvals, eigvecs = tf.linalg.eigh(corr_hat)
        eigvals = tf.cast(eigvals, tf.float32)
        eigvecs = tf.cast(eigvecs, tf.float32)

        lam_emp = tf.expand_dims(eigvals, axis=-1)
        Q_emp   = eigvecs

        pos  = tf.tile(tf.reshape(tf.linspace(0.0, 1.0, N), (1, N, 1)), [B, 1, 1])
        Q_sq = tf.square(tf.transpose(Q_emp, perm=[0, 2, 1]))  # (B, N, N)

        # T̃_min per stock → project onto eigenmodes → q_eff (mirrors tf_data_generator)
        observed_f = tf.cast(observed, tf.float32)
        has_any = tf.reduce_any(observed, axis=2)  # (B, N) — False for fully-absent stocks
        tmin_raw = tf.cast(tf.argmax(observed_f, axis=2, output_type=tf.int32), tf.float32)
        # argmax returns 0 for all-False rows; override with T so q_eff → ∞ (maximally noisy)
        T_float = tf.cast(T, tf.float32)
        tmin_raw = tf.where(has_any, tmin_raw, T_float)
        Tmin = tf.expand_dims(tmin_raw / T_float, axis=-1)  # (B, N, 1)
        Tminmean = tf.matmul(Q_sq, Tmin)                                 # (B, N, 1)

        T_vec = tf.fill((B, N, 1), tf.cast(T, tf.float32))
        N_vec = tf.fill((B, N, 1), tf.cast(N, tf.float32))

        effective_T_frac = tf.maximum(1.0 - Tminmean, 1.0 / tf.cast(T, tf.float32))
        q_eff = (N_vec / T_vec) / effective_T_frac  # (B, N, 1)

        # IPR_j = N · Σ_i Q_{ij}^4
        ipr = tf.cast(N, tf.float32) * tf.reduce_sum(tf.square(Q_sq), axis=2, keepdims=True)  # (B, N, 1)

        # z_MP_j = (λ_j − (1 + √q_eff)²) / √q_eff
        q_eff_safe = tf.maximum(q_eff, 1e-6)
        z_MP = (lam_emp - tf.square(1.0 + tf.sqrt(q_eff_safe))) / tf.sqrt(q_eff_safe)  # (B, N, 1)

        input_seq = tf.concat(
            [lam_emp, pos, q_eff, ipr, z_MP],
            axis=-1,
        )  # (B, N, 5) — same feature set as tf_data_generator

        return input_seq, Q_emp, Sigma_hat_diag, T
