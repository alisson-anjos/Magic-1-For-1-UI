"""
EMA utilities.
"""

from copy import deepcopy
import torch
from torch import nn


class EMA(nn.Module):
    """
    Exponential moving average for nn.Module.
    """

    def __init__(
        self,
        model: nn.Module,
        beta: float = 0.9999,
        update_every: int = 1,
        oncpu: bool = False,
    ):
        super().__init__()
        self.step = 0
        self.beta = beta
        self.update_every = update_every
        self.cur_model = [model]
        self.model_ema = deepcopy(model)
        self.model_ema.requires_grad_(False)
        if oncpu:
            self.model_ema.to("cpu")

    @torch.no_grad()
    def update(self):
        if self.step % self.update_every == 0:
            ema, cur = self.model_ema, self.cur_model[0]
            for ema_params, cur_params in zip(ema.parameters(), cur.parameters()):
                ema_params.lerp_(cur_params.to(ema_params), 1 - self.beta)
            for ema_buffer, cur_buffer in zip(ema.buffers(), cur.buffers()):
                ema_buffer.lerp_(cur_buffer.to(ema_buffer), 1 - self.beta)
        self.step += 1

    def state_dict(self, *args, **kwargs):
        return self.model_ema.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model_ema.load_state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model_ema(*args, **kwargs)
