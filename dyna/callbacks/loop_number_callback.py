import math
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


class LoopNumberCallback(Callback):
    """
    Callback to compute and log Shannon entropy of language model predictions.

    Computes H(p) = -sum(p(x_i) * log(p(x_i))) for each position in the sequence,
    then averages across the batch and logs to wandb. Also tracks per-layer entropy.
    """

    def __init__(
        self,
        log_interval: str = "100ba",
        total_training_duration: str = "100ba",
        upper_bound: int = 6,
        lower_bound: int = 2,
        warm_up=0.2,
        cool_down=0.2,
        log_key: str = "loop_number",
    ):

        self.log_interval = (
            Time.from_timestring(log_interval)
            if isinstance(log_interval, str)
            else Time(log_interval, TimeUnit.BATCH)
        )

        self.T = (
            Time.from_timestring(total_training_duration)
            if isinstance(total_training_duration, str)
            else Time(total_training_duration, TimeUnit.BATCH)
        )
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.warm_up = warm_up
        self.cool_down = cool_down
        self.log_key = log_key  # Initialize log_key
        # Storage for per-layer entropy data
        # self.entropy_data = []  # Add this back for the plot

    def _should_log(self, state: State) -> bool:
        """Determine if it's time to log based on the log_interval."""
        if isinstance(self.log_interval, Time):
            return (
                state.timestamp.get(self.log_interval.unit)
                % self.log_interval.value
                == 0
            )
        return False

    def batch_start(self, state: State, logger: Logger) -> None:
        """
        Called at the end of each batch to compute and log entropy.
        """
        if not self._should_log(state):
            return
        metrics_dict = {}
        G = self.T.value*(1-self.warm_up-self.cool_down)
        g = (self.upper_bound - self.lower_bound-1)/G
        slope = self.warm_up*float(self.T.value)*g+self.upper_bound-g*float(state.timestamp.batch)
        state.model.model.transformer.min_loop_layers = math.floor(max(self.lower_bound,min(self.upper_bound, slope)))
        metrics_dict["metrics/"+self.log_key] = state.model.model.transformer.min_loop_layers
        logger.log_metrics(metrics_dict)
