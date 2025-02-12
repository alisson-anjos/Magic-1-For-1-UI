"""
Accumulator for logging.
"""

from functools import partial
from typing import Dict, List, Literal, Union
import torch
from torch import Tensor
from torch.distributed import ReduceOp, all_reduce, is_initialized

from .distributed import get_device


class Accumulator:
    """
    Accumulate values over multiple iterations.
    This is intended for loss logging purposes.
    """

    def __init__(self, mode: Literal["avg", "sum"]):
        self.reset()
        assert mode in ["avg", "sum"]
        self.mode = mode

    def reset(self):
        """
        Reset the accumulator.
        """
        self.sum = {}
        self.num = {}

    @torch.no_grad()
    def add(self, **kwargs: Dict[str, Union[int, float, Tensor, List]]):
        """
        Add value to the accumulator.
        """
        for k, v in kwargs.items():
            sv = sum(v) if isinstance(v, list) else v
            nv = len(v) if isinstance(v, list) else 1
            self.sum[k] = self.sum.get(k, 0) + sv
            self.num[k] = self.num.get(k, 0) + nv

    def get(self) -> Dict[str, Union[float, Tensor]]:
        """
        Get accumulated values.
        """
        if self.mode == "avg":
            return {k: self.sum[k] / self.num[k] for k in self.sum.keys()}
        elif self.mode == "sum":
            return self.sum
        else:
            raise NotImplementedError

    def get_and_reset(self) -> Dict[str, Union[float, Tensor]]:
        """
        Get accumulated value and reset.
        """
        val = self.get()
        self.reset()
        return val


class DistributedAccumulator(Accumulator):
    """
    Accumulate values over multiple iterations and over all GPUs.
    This is intended for loss logging purposes.
    The distributed accumulator must be instantiated on all GPU ranks.
    The method "get" and "get_and_reset" must be invoked on all GPU ranks.
    """

    def get(self) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Get accumulated values from all ranks.
        """
        # Instead of all_reduce on values individually,
        # we merge values to a big tensor and only reduce once together.
        result = {k: [self.sum[k], self.num[k]] for k in self.sum.keys()}.items()
        device = get_device()
        tensor = torch.cat([torch.as_tensor(v, device=device).view(-1) for _, v in result])
        if is_initialized():
            all_reduce(tensor=tensor, op=ReduceOp.SUM)  # sum plus sum, num plus num
        if self.mode == "sum":
            tensor = tensor[torch.arange(0, len(tensor), 2)]
        elif self.mode == "avg":
            tensor = (
                tensor[torch.arange(0, len(tensor), 2)] / tensor[torch.arange(1, len(tensor), 2)]
            )
        else:
            raise NotImplementedError("Other mode is not supported.")
        # Split back to original shapes.
        tensor = tensor.split([v.numel() if torch.is_tensor(v) else 1 for _, v in result])
        result = {
            k: t.reshape_as(v) if torch.is_tensor(v) else t.item()
            for t, (k, v) in zip(tensor, result)
        }
        # remove inf values
        for key in result.keys():
            if result[key] == float("inf"):
                del result[key]
        return result


# Backward compatibility.
AverageAccumulator = partial(Accumulator, mode="avg")
DistributedAverageAccumulator = partial(DistributedAccumulator, mode="avg")
