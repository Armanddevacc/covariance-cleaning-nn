import torch


def get_optimizer(model):
    return torch.optim.Adam(
        model.parameters(), lr=5e-4, weight_decay=1e-6
    )  # lr and weight_decay to be tuned
