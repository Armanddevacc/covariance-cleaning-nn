from training.optimizer import get_optimizer
import torch.nn as nn
import torch
from estimator.MLE import torch_cov_pairwise
import numpy as np


class Trainer:
    def __init__(
        self,
        model,
        is_train_on_real_data,
        loss_function,
        data_generator,
        lr,
        weight_decay,
        batch_size,
        epochs,
        N_min,
        N_max,
        T_min,
        T_max,
        log_interval,
        accumulate_steps,
        dataset,
        missing_constant,
    ):
        self.model = model
        self.is_train_on_real_data = is_train_on_real_data
        self.optimizer = get_optimizer(model, lr=lr, weight_decay=weight_decay)
        self.loss_function = loss_function
        self.data_generator = data_generator
        self.batch_size = batch_size
        self.epochs = epochs
        self.N_min = N_min
        self.N_max = N_max
        self.T_min = T_min
        self.T_max = T_max
        self.log_interval = log_interval
        self.accumulate_steps = accumulate_steps
        self.dataset = dataset
        self.missing_constant = missing_constant

        self.loss_history = []

    def train(self):
        print(f"Starting training for {self.epochs} epochs…")

        if not self.is_train_on_real_data:
            generator = self.data_generator(
                self.batch_size,
                N_min=self.N_min,
                N_max=self.N_max,
                T_min=self.T_min,
                T_max=self.T_max,
                missing_constant=self.missing_constant,
            )
        else:
            generator = self.data_generator(
                N_min=self.N_min,
                N_max=self.N_max,
                T_min=self.T_min,
                T_max=self.T_max,
                dataset=self.dataset,
                missing_constant=self.missing_constant,
            )

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            for _ in range(
                self.accumulate_steps
            ):  # accumulate gradients over multiple batches
                input_seq, Q_emp, Mat_oos, T, Sigma_hat_diag, _ = next(generator)

                # Forward through the network to estimate corr matrix eigenvalues
                # In the input_seq we pass :
                #   the eigenvalues of a corr matrix which has Q_emp*Sigma_hat_diag as transition matrix (to cov)
                #   N_vec, T_vec, which are sizes of the different matrix
                #   N_vec / T_vec, which is the ratio of ...
                #   Tmin_mean, Tmax_mean, which are featured engeneered and supposed to help with the prediction given though some variable are missing

                lam_pred = self.model(input_seq)
                cov_pred = (
                    torch.sqrt(torch.diag_embed(Sigma_hat_diag)).float()
                    @ Q_emp
                    @ torch.diag_embed(lam_pred)
                    @ Q_emp.transpose(-1, -2)
                    @ torch.sqrt(torch.diag_embed(Sigma_hat_diag)).float()
                )

                # When we train on real data we don't have the real cov/corr matrix but we have oos data which we can use that way :
                if self.is_train_on_real_data:
                    # in that case Mat_oos was oos return on 10 days; thanks to them we compute the covariance matrix of the assets
                    # in the case where train_on_real_data is false the Mat_oos is already the true covariance matrix (we say true because it is thnak to it that return are generated)
                    Mat_oos = torch_cov_pairwise(Mat_oos)

                # at the end of the day we want the loss between covariance to be small so it is our loss criterion
                loss = self.loss_function(Mat_oos, cov_pred, T)
                (loss / self.accumulate_steps).backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # keeps
            self.optimizer.step()

            self.loss_history.append(loss.item())

            # logging
            if (epoch + 1) % self.log_interval == 0:
                print(f"Epoch {epoch+1}/{self.epochs} — loss: {loss.item():.8f}")

        print("Training complete.")
        return self.loss_history


import tensorflow as tf
from estimator.MLE import tf_cov_pairwise


class Trainer_tf:
    def __init__(
        self,
        model,
        is_train_on_real_data,
        loss_function,
        data_generator,
        lr,
        weight_decay,
        batch_size,
        epochs,
        N_min,
        N_max,
        T_min,
        T_max,
        log_interval,
        accumulate_steps,
        dataset,
        missing_constant,
    ):
        self.model = model
        self.is_train_on_real_data = is_train_on_real_data
        self.loss_function = loss_function
        self.data_generator = data_generator
        self.batch_size = batch_size
        self.epochs = epochs
        self.N_min = N_min
        self.N_max = N_max
        self.T_min = T_min
        self.T_max = T_max
        self.log_interval = log_interval
        self.accumulate_steps = accumulate_steps
        self.dataset = dataset
        self.missing_constant = missing_constant

        # Equivalent to get_optimizer(...)
        self.optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr, weight_decay=weight_decay
        )

        self.loss_history = []

    def train(self):
        print(f"Starting training for {self.epochs} epochs…")

        if not self.is_train_on_real_data:
            generator = self.data_generator(
                self.batch_size,
                N_min=self.N_min,
                N_max=self.N_max,
                T_min=self.T_min,
                T_max=self.T_max,
                missing_constant=self.missing_constant,
            )
        else:
            generator = self.data_generator(
                N_min=self.N_min,
                N_max=self.N_max,
                T_min=self.T_min,
                T_max=self.T_max,
                dataset=self.dataset,
                missing_constant=self.missing_constant,
            )

        # --------- FORCE MODEL BUILD (VERY IMPORTANT) ----------
        input_seq, *_ = next(generator)
        input_seq = tf.cast(input_seq, tf.float32)
        _ = self.model(input_seq, training=True)

        # Re-create generator because we consumed one batch
        if not self.is_train_on_real_data:
            generator = self.data_generator(
                self.batch_size,
                N_min=self.N_min,
                N_max=self.N_max,
                T_min=self.T_min,
                T_max=self.T_max,
                missing_constant=self.missing_constant,
            )
        else:
            generator = self.data_generator(
                N_min=self.N_min,
                N_max=self.N_max,
                T_min=self.T_min,
                T_max=self.T_max,
                dataset=self.dataset,
                missing_constant=self.missing_constant,
            )
        # -------------------------------------------------------

        for epoch in range(self.epochs):

            # TensorFlow does not need model.train()
            vars_ = self.model.trainable_variables
            accum_grads = [tf.zeros_like(v) for v in vars_]

            for _ in range(self.accumulate_steps):
                input_seq, Q_emp, Mat_oos, T, Sigma_hat_diag, _ = next(generator)

                # Convert numpy → tensors if needed
                input_seq = tf.cast(input_seq, dtype=tf.float32)
                Q_emp = tf.cast(Q_emp, dtype=tf.float32)
                Mat_oos = tf.cast(Mat_oos, dtype=tf.float32)
                Sigma_hat_diag = tf.cast(Sigma_hat_diag, dtype=tf.float32)
                T = tf.cast(T, tf.float32)

                with tf.GradientTape() as tape:

                    # Forward through the network to estimate corr matrix eigenvalues
                    # In the input_seq we pass :
                    #   the eigenvalues of a corr matrix which has Q_emp*Sigma_hat_diag as transition matrix (to cov)
                    #   N_vec, T_vec, which are sizes of the different matrix
                    #   N_vec / T_vec, which is the ratio of ...
                    #   Tmin_mean, Tmax_mean, which are featured engeneered and supposed to help with the prediction given though some variable are missing
                    lam_pred = self.model(input_seq, training=True)

                    D_sigma = tf.linalg.diag(Sigma_hat_diag)
                    D_lam = tf.linalg.diag(lam_pred)

                    cov_pred = tf.matmul(
                        tf.matmul(
                            tf.matmul(tf.matmul(D_sigma, Q_emp), D_lam),
                            tf.transpose(Q_emp, perm=[0, 2, 1]),
                        ),
                        D_sigma,
                    )

                    # When we train on real data we don't have the real cov/corr matrix but we have oos data which we can use that way :
                    if self.is_train_on_real_data:
                        # in that case Mat_oos was oos return on 10 days; thanks to them we compute the covariance matrix of the assets
                        # in the case where train_on_real_data is false the Mat_oos is already the true covariance matrix (we say true because it is thnak to it that return are generated)
                        Mat_oos = tf_cov_pairwise(Mat_oos)

                    # eps = 1e-12
                    # diag_oos = tf.linalg.diag_part(Mat_oos)
                    # std_oos = tf.sqrt(tf.maximum(diag_oos, eps))
                    # corr_oos = Mat_oos / (std_oos[:, None, :] * std_oos[:, :, None] + eps)

                    # at the end of the day we want the loss between covariance to be small so it is our loss criterion
                    loss = self.loss_function(Mat_oos, cov_pred, T)
                    loss_scaled = loss / float(self.accumulate_steps)

                grads = tape.gradient(loss_scaled, vars_)
                accum_grads = [
                    ag + (g if g is not None else tf.zeros_like(v))
                    for ag, g, v in zip(accum_grads, grads, vars_)
                ]

            # Equivalent to nn.utils.clip_grad_norm_(..., max_norm=1.0)
            clipped_grads, _ = tf.clip_by_global_norm(accum_grads, 1.0)
            self.optimizer.apply_gradients(zip(clipped_grads, vars_))

            self.loss_history.append(float(loss.numpy()))

            # logging
            if (epoch + 1) % self.log_interval == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs} — loss: {float(loss.numpy()):.8f}"
                )

        print("Training complete.")
        return self.loss_history
