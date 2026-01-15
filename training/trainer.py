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
                input_seq, Q_emp, Mat_oos, T = next(generator)

                # forward
                lam_pred = self.model(input_seq)
                Sigma_pred = (
                    Q_emp @ torch.diag_embed(lam_pred) @ Q_emp.transpose(-1, -2)
                )

                # loss
                if self.is_train_on_real_data:
                    Mat_oos = torch_cov_pairwise(Mat_oos)

                eps = 1e-12
                diag_oos = torch.diagonal(Mat_oos, dim1=-2, dim2=-1)
                std_oos = torch.sqrt(torch.clamp(diag_oos, min=eps))
                corr_oos = Mat_oos / (std_oos[:, None, :] * std_oos[:, :, None] + eps)

                diag_pred = torch.diagonal(Sigma_pred, dim1=-2, dim2=-1)
                std_pred = torch.sqrt(torch.clamp(diag_pred, min=eps))
                corr_pred = Sigma_pred / (
                    std_pred[:, None, :] * std_pred[:, :, None] + eps
                )

                loss = self.loss_function(corr_oos, corr_pred, T)
                (loss / self.accumulate_steps).backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # keeps
            self.optimizer.step()

            self.loss_history.append(loss.item())

            # logging
            if (epoch + 1) % self.log_interval == 0:
                print(f"Epoch {epoch+1}/{self.epochs} — loss: {loss.item():.8f}")

        print("Training complete.")
        return self.loss_history
