import torch
import torch.nn.functional as F
from typing import Any, Dict
import wandb
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from matplotlib.ticker import ScalarFormatter

class CleanMetrics(Callback):
    """
    Callback to compute and log Shannon entropy of language model predictions.

    Computes H(p) = -sum(p(x_i) * log(p(x_i))) for each position in the sequence,
    then averages across the batch and logs to wandb. Also tracks per-layer entropy.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def batch_start(self, state: State, logger: Logger) -> None:
        """
        Called at the end of each batch to compute and log entropy.

        Args:
            state: Composer training state
            logger: Composer logger instance
        """
        transformer = state.model.model.transformer
        transformer._latent_vectors = []
        transformer._exit_logits = []
        transformer._seq_len = []
        transformer._expert_sel = []
        # Also clear sel_hist for all layers (attention and ffn)
        for layer in getattr(transformer, "layers", []):
            if hasattr(layer, "attention") and hasattr(layer.attention, "sel_hist"):
                layer.attention.sel_hist = []
            if hasattr(layer, "ffn") and hasattr(layer.ffn, "sel_hist"):
                layer.ffn.sel_hist = []
