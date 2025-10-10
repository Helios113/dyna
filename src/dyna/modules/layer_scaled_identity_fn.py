import atexit

import torch
from beartype import beartype

# from composer.callbacks
# Add jaxtyping imports
from jaxtyping import Float, Int
from torch import Tensor
from torch.nn import Module
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

# Constants
CROSS_ENTROPY_IGNORE_INDEX = -100


def cleanup_distributed():
    """Clean up distributed training resources."""
    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
        except Exception as e:
            print(f"Warning: Error during distributed cleanup: {e}")


def setup_distributed_cleanup():
    """Register cleanup function to run at program exit."""
    atexit.register(cleanup_distributed)


@beartype
def get_targets(labels: Int[Tensor, "batch seq"]) -> Int[Tensor, "batch seq"]:
    """Shift labels for causal language modeling."""
    targets = torch.roll(labels, shifts=-1)
    targets[:, -1] = CROSS_ENTROPY_IGNORE_INDEX
    return targets


@beartype
def compute_loss_from_logits(
    outputs: CausalLMOutputWithPast,
    shift_labels: bool,
    labels: Int[Tensor, "batch seq"],
    loss_fn: Module,
) -> Float[Tensor, ""]:
    """Compute cross-entropy loss from logits and labels."""
    targets = get_targets(labels) if shift_labels else labels

    losses = loss_fn(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        targets.view(-1),
    )

    if torch.all(targets == loss_fn.ignore_index):
        loss = losses.sum()
    else:
        loss = losses.sum() / (targets != loss_fn.ignore_index).sum()

    return loss


@beartype
def round_up_to_multiple_of_256(n: torch.Tensor) -> int:
    """Return the smallest number divisible by 256 that is >= n."""
    if n <= 0:
        return 256

    return int(((n - 1) // 256 + 1) * 256)


# class LayerScaledIdentityFn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, layer_idx: int, total_layers: int):
#         # Save values needed for backward
#         ctx.layer_idx = layer_idx
#         ctx.total_layers = total_layers
#         return input.clone()  # return unchanged

#     @staticmethod
#     def backward(ctx, grad_output):
#         # Compute scaling factor
#         scale = (ctx.total_layers - ctx.layer_idx) ** 2
#         grad_input = grad_output * scale
#         # None for layer_idx and total_layers (no gradients for ints)
#         return grad_input, None, None


class LayerScaledIdentityFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, total_layers: int):
        # Save values needed for backward
        ctx.total_layers = total_layers
        return input.clone()  # return unchanged

    @staticmethod
    def backward(ctx, grad_output):
        # Compute scaling factor
        grad_input = grad_output * ctx.total_layers
        return grad_input, None


def layer_scaled_identity(x, total_layers: int):
    return LayerScaledIdentityFn.apply(x, total_layers)
