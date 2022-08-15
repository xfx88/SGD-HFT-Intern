from typing import Optional, Union

import numpy as np
import math
import torch

class Optimiser:

    def __init__(self, model, optimiser=None, scale_factor=1, warmup_steps=2000, beta1=0.9, beta2=0.98, epsilon=1e-9):

        if optimiser is not None: self.optimiser = optimiser
        else: self.optimiser = torch.optim.AdamW(model.parameters(), lr=0, betas=(beta1, beta2), eps=epsilon)

        self.scale_factor = scale_factor
        self.warmup_steps = math.pow(warmup_steps, -1.5)
        self.current_step = 0
        self.inv_sqrt_d_input = math.pow(model.module.d_input, -0.5)

        self.lrate = lambda step: self.inv_sqrt_d_input * min(math.pow(step, -0.5), step * self.warmup_steps)
        self.rate = None

    def step(self):
        self.current_step += 1
        self.rate = self.scale_factor * self.lrate(self.current_step)
        for i in self.optimiser.param_groups: i['lr'] = self.rate
        self.optimiser.step()


def generate_original_PE(length: int, d_model: int) -> torch.Tensor:
    """Generate positional encoding as described in original paper.  :class:`torch.Tensor`

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length).unsqueeze(1)
    PE[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model))
    PE[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model))

    return PE


def generate_regular_PE(length: int, d_model: int, period: Optional[int] = 64) -> torch.Tensor:
    """Generate positional encoding with a given period.

    Parameters
    ----------
    length:
        Time window length, i.e. K.
    d_model:
        Dimension of the model vector.
    period:
        Size of the pattern to repeat.
        Default is 24.

    Returns
    -------
        Tensor of shape (K, d_model).
    """
    PE = torch.zeros((length, d_model))

    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    PE = torch.sin(pos * 2 * np.pi / period)
    PE = PE.repeat((1, d_model))

    return PE


def generate_local_map_mask(chunk_size: int,
                            attention_size: int,
                            mask_future=False,
                            device: torch.device = 'cpu') -> torch.BoolTensor:
    """Compute attention mask as attention_size wide diagonal.

    Parameters
    ----------
    chunk_size:
        Time dimension size.
    attention_size:
        Number of backward elements to apply attention.
    device:
        torch device. Default is ``'cpu'``.

    Returns
    -------
        Mask as a boolean tensor.
    """
    local_map = np.empty((chunk_size, chunk_size))
    i, j = np.indices(local_map.shape)

    if mask_future:
        local_map[i, j] = (i - j > attention_size) ^ (j - i > 0)
    else:
        local_map[i, j] = np.abs(i - j) > attention_size

    return torch.BoolTensor(local_map).to(device)
