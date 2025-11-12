from collections.abc import Callable

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from dyna.config import (
    CROSS_ENTROPY_IGNORE_INDEX,
)
from dyna.model.base import DynaConfig, DynaPretrainedModel


def calc_entropy(
    chunks: Float[Tensor, "batch seq_len vocab"],
    temperature: float,
) -> Float[Tensor, "batch seq_len"]:
    """Calculate entropy of logits.

    Args:
        chunks (Float[Tensor, "batch seq vocab"]): Logits tensor.
        temperature (float): Temperature for softmax.

    Returns:
        Float[Tensor, "batch seq"]: Entropy tensor.
    """
    # entropy = 0
    # for logits in chunks:
    logits = chunks
    probs = torch.softmax(logits / temperature, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    entropy = -torch.sum(probs * log_probs, dim=-1).detach()
    return entropy


class PassThroughTransformer(DynaPretrainedModel):
    """Transformer model with configurable behavior."""

    def __init__(self, config: DynaConfig):
        """Initialize DynaFormer model.

        Args:
            config (DynaConfig): Configuration object for the model.
        """
        super().__init__(config)
        # entropy calculation head
        self._temp_lm_head: Callable[[torch.Tensor], torch.Tensor] | None = None

    def reset_parameters(self) -> None:
        self._seq_len = []
        self._latent_vectors = []
        self._residual_magnitudes = []
        self._exit_logits = []
        self._expert_sel = []

    def _collect_regularization_loss(self) -> torch.Tensor:
        return 0

    def _clear_selection_history(self):
        pass

    def forward(
        self,
        x: Float[Tensor, "batch seq d_model"],
        attention_mask: Bool[Tensor, "batch 1 seq seq"],
        sequence_length: Int[Tensor, "batch seq"],
        e: Float[Tensor, "batch seq d_model"] | None = None,
        input_ids: Int[Tensor, "batch seq"] | None = None,
    ) -> tuple[Float[Tensor, "batch seq d_model"], Float[Tensor, "batch seq 1"] | None]:
        if input_ids is not None:
            _labels = torch.roll(input_ids, shifts=-1)
            _labels[:, -1] = CROSS_ENTROPY_IGNORE_INDEX

        # logging data containers
        self._expert_sel.append([])
        self._exit_logits.append([])
        self._latent_vectors.append([])
        self._seq_len.append([])
        self._residual_magnitudes.append([])

        if self.gather_stats:
            self.gather_stats_func(x, ((None, None), None))

        return x, None

    def gather_stats_func(
        self,
        x: Float[Tensor, "batch seq d_model"],
        expert_sel: tuple[
            tuple[
                Int[Tensor, "batch seq expert_heads attn_experts"] | None,
                Int[Tensor, "batch seq expert_heads attn_experts"] | None,
            ],
            Int[Tensor, "batch seq ffn_experts"] | None,
        ],
    ) -> None:
        assert isinstance(self._temp_lm_head, Callable)
        self._latent_vectors[-1].append(
            calc_entropy(
                self._temp_lm_head(x.detach()),
                1.0,
            )
        )
        if expert_sel[0] is not None:
            self._expert_sel[-1].append(expert_sel)
        self._residual_magnitudes[-1].append(torch.norm(x.detach(), dim=-1).cpu())
