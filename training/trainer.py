from training.optimizer import get_optimizer
import torch.nn as nn
import torch


class Trainer:
    def __init__(
        self,
        model,
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
    ):
        self.model = model
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

        self.loss_history = []

    def train(self):
        print(f"Starting training for {self.epochs} epochs…")

        generator = self.data_generator(
            self.batch_size,
            N_min=self.N_min,
            N_max=self.N_max,
            T_min=self.T_min,
            T_max=self.T_max,
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

                # loss
                loss = self.loss_function(lam_pred, Q_emp, Mat_oos, T)
                (loss / self.accumulate_steps).backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # keeps
            self.optimizer.step()

            self.loss_history.append(loss.item())

            # logging
            if (epoch + 1) % self.log_interval == 0:
                print(f"Epoch {epoch+1}/{self.epochs} — loss: {loss.item():.6f}")

        print("Training complete.")
        return self.loss_history
