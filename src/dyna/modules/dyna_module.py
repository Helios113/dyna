import math
from abc import ABC, abstractmethod

import torch
from jaxtyping import Float
from torch import Tensor


def log_mean(
    x: Float[Tensor, "batch seq_len k_experts n_experts"], dim: int = 0
) -> Float[Tensor, "batch seq_len k_experts"]:
    """Compute log of mean along specified dimension."""
    return x.logsumexp(dim) - math.log(x.shape[dim])


def entropy_l(
    log_probs: Float[Tensor, "batch seq_len k_experts"],
) -> Float[Tensor, "batch seq_len"]:
    """Compute entropy from log probabilities."""
    return -(log_probs * log_probs.exp()).sum(-1)


def entropy_reg(
    sel: Float[Tensor, "batch seq_len k_experts n_experts"], dim: int
) -> Float[Tensor, ""]:
    """Compute entropy regularization term."""
    sel = torch.nn.functional.log_softmax(sel, dim=-1)
    sel = log_mean(sel, dim)
    return -entropy_l(sel).mean()


class DynaModule(torch.nn.Module, ABC):
    @abstractmethod
    def get_reg_loss(self) -> torch.Tensor:
        pass

    @abstractmethod
    def clear_selection_history(self):
        pass

    @abstractmethod
    def reset_parameters(self, ffn_scale: float, attn_scale: float) -> None:
        pass

    @property
    def device(self):
        return next(self.parameters()).device
