from random import random

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


class TiedLinear(torch.nn.Module):
    """
    Tied linear layer for autoencoder

    For reference, see: https://github.com/openai/sparse_autoencoder/blob/8f74a1cbeb15a6a7e082c812ccc5055045256bb4/sparse_autoencoder/model.py#L87
    """

    def __init__(self, tied_to: torch.nn.Linear):
        super().__init__()
        self.tied_to = tied_to

    def forward(self, x):
        return F.linear(x, self.tied_to.weight.t(), bias=self.tied_to.bias)

    @property
    def weight(self):
        return self.tied_to.weight.t()

    @property
    def bias(self):
        return self.tied_to.bias


def optimizer_to(optim, device):
    """
    From https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    """
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def truncate_seq(seq, max_length):
    offset = int(random() * (len(seq) - max_length))
    return seq[offset:offset + max_length]


def plateau_lr_scheduler(
        optim: torch.optim.Optimizer,
        total_steps: int,
        warmup_pct: float,
        decay_pct: float,
        start_factor=0.1,
        end_factor=0.01
):
    """
    A learning rate scheduler that starts with a warmup, then a plateau, then a decay

    :param optim: the optimizer to use
    :param total_steps: the total number of steps that will be taken
    :param warmup_pct: the percentage of steps to use for warmup
    :param decay_pct: the percentage of steps to use for decay
    :param start_factor: the factor to start with for warmup
    :param end_factor: the factor to end with for decay
    """
    return lr_scheduler.SequentialLR(
        optimizer=optim,
        schedulers=[
            lr_scheduler.LinearLR(optimizer=optim, start_factor=start_factor, end_factor=1,
                                  total_iters=int(total_steps * warmup_pct)),
            lr_scheduler.LinearLR(optimizer=optim, start_factor=1, end_factor=end_factor,
                                  total_iters=int(total_steps * decay_pct)),
        ],
        milestones=[int(total_steps * (1 - decay_pct))]
    )
