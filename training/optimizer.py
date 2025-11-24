import torch


def get_optimizer(model, lr=1e-4, weight_decay=1e-6):
    return torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )  # lr and weight_decay to be tuned
