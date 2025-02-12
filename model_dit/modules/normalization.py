from typing import Tuple

import torch
from torch import nn


def get_norm(
    norm_name: str,
    n_embd: int,
    eps: float = 1e-5,
    num_embeds_ada_norm: int = 0,
) -> nn.Module:
    if norm_name == "LayerNorm":
        return nn.LayerNorm(n_embd, eps=eps)
    elif norm_name == "AdaLayerNormSingle":
        return CogVideoXLayerNormZero(n_embd, n_embd, eps=eps)
    elif norm_name == "RMSNorm":
        return RMSNorm(n_embd, eps=eps)
    else:
        raise ValueError(f"Unknown norm name {norm_name}")


class CogVideoXLayerNormZero(nn.Module):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps=eps)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        return hidden_states, gate[:, None, :], enc_gate[:, None, :]


# from llama
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
