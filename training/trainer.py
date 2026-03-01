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
from estimator.MLE import tf_cov_pairwise  # to be used when out of sample data
from models.losses import tf_loss_function_mat


class Trainer_tf:
    # gradientTape : recuperer les gradients d'une fonctions de pertes par rapport a
    # des poids trainable des couches
    # utiliser ces gradients pour mettre à jours ces poids en les recuperant avec model.trainable_weights

    def __init__(
        self,
        model,
        data_generator,
        batch_size,
        epochs,
        missing_constant,
        N_min,
        N_max,
        T_min,
        T_max,
    ):
        self.model = model
        self.epochs = epochs
        # We need parameters for the data generator
        self.batch_size = batch_size
        self.missing_constant = missing_constant
        self.N_min = N_min
        self.N_max = N_max
        self.T_min = T_min
        self.T_max = T_max
        # first we need an optimizer :
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        # we need a loss function
        self.loss_function = tf_loss_function_mat
        # and a data generator
        self.data_generator = data_generator(
            self.batch_size,
            self.missing_constant,
            self.N_min,
            self.N_max,
            self.T_min,
            self.T_max,
        )
        # save loss over time
        self.loss_history = []

    def train(self, steps_per_epoch):
        """
        The training loop train over epoch * steps_per_epoch * batch number of
        sample matrix
        """
        for epoch in range(self.epochs):

            loss = self._train_step(steps_per_epoch)

            self.loss_history.append(loss)

            print(f"Epoch {epoch+1}, Loss: {loss:.6f}")
        return self.loss_history

    def _train_step(self, steps_per_epoch):
        """ """

        with tf.GradientTape() as tape:
            loss = 0
            for step in range(steps_per_epoch):
                input_seq, Q_emp, Sigma_true, T, Sigma_hat_diag, R_hat = next(
                    self.data_generator
                )
                Q_emp = tf.cast(Q_emp, tf.float32)
                Sigma_hat_diag = tf.cast(Sigma_hat_diag, tf.float32)
                Sigma_true = tf.cast(Sigma_true, tf.float32)

                # input_seq, containts Sigma_emp: (B, N, T)
                lam_pred = self.model(input_seq, training=True)

                # lam_pred: (B, N)
                Sigma_pred = self._reconstruct_cov(lam_pred, Q_emp, Sigma_hat_diag)

                loss += self.loss_function(Sigma_true, Sigma_pred, T) / steps_per_epoch

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def _reconstruct_cov(self, lam_pred, Q_emp, Sigma_hat_diag):

        Sigma_pred = tf.matmul(
            tf.matmul(
                tf.matmul(
                    tf.matmul(tf.sqrt(tf.linalg.diag(Sigma_hat_diag)), Q_emp),
                    tf.linalg.diag(lam_pred),
                ),
                tf.transpose(Q_emp, perm=[0, 2, 1]),
            ),
            tf.sqrt(tf.linalg.diag(Sigma_hat_diag)),
        )
        return Sigma_pred
