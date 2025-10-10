import torch
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger


class CleanMetrics(Callback):
    """Callback to compute and log Shannon entropy of language model predictions.

    Computes H(p) = -sum(p(x_i) * log(p(x_i))) for each position in the sequence,
    then averages across the batch and logs to wandb. Also tracks per-layer entropy.
    """

    def __init__(
        self,
        interval: str | int = "1ba",
    ):
        super().__init__()
        self.interval = (
            Time.from_timestring(interval)
            if isinstance(interval, str)
            else Time(interval, TimeUnit.BATCH)
        )

    def find_transformer_module(self, module: torch.nn.Module):
        # Recursively search for a child named 'transformer'
        for name, child in module.named_children():
            if name == "transformer":
                return child
            result = self.find_transformer_module(child)
            if result is not None:
                return result
        return None

    def batch_start(self, state: State, logger: Logger) -> None:
        transformer = self.find_transformer_module(state.model)

        if (
            state.timestamp.get(self.interval.unit) + Time.from_timestring("1ba")
        ) % self.interval.value == 0:
            transformer.gather_stats = True
        else:
            transformer.gather_stats = False

        transformer._latent_vectors.clear()
        transformer._exit_logits = []
        transformer._seq_len = []
        transformer._expert_sel.clear()

        # Also clear sel_hist for all layers (attention and ffn)
        for layer in getattr(transformer, "layers", []):
            if hasattr(layer, "attention") and hasattr(layer.attention, "sel_hist"):
                layer.attention.sel_hist = []
            if hasattr(layer, "ffn") and hasattr(layer.ffn, "sel_hist"):
                layer.ffn.sel_hist = []
