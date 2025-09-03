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

class ResidualMagnitudeCallback(Callback):
    """
    Callback to compute and log Shannon entropy of language model predictions.

    Computes H(p) = -sum(p(x_i) * log(p(x_i))) for each position in the sequence,
    then averages across the batch and logs to wandb. Also tracks per-layer entropy.
    """

    def __init__(
        self,
        log_interval: str = "100ba",
        figsize: tuple[int, int] = (12, 8),
        log_key: str = "residual_magnitude"
    ):
       
        self.log_interval = (
            Time.from_timestring(log_interval)
            if isinstance(log_interval, str)
            else Time(log_interval, TimeUnit.BATCH)
        )
        self.last_batch_logged = -1
        self.figsize = figsize
        self.log_key = log_key  # Initialize log_key

    def _should_log(self, state: State) -> bool:
        """Determine if it's time to log based on the log_interval."""
        if isinstance(self.log_interval, Time):
            return (
                state.timestamp.batch != self.last_batch_logged
                and state.timestamp.get(self.log_interval.unit)
                % self.log_interval.value
                == 0
            )
        return False

    def batch_end(self, state: State, logger: Logger) -> None:
        """
        Called at the end of each batch to compute and log entropy.
        """
        if  not self._should_log(state):
            return
        
        metrics_dict = {}
        step = str(state.timestamp.batch)
        tmp = state.model.model.transformer._residual_magnitudes
        residual_magnitudes = []
        for elem in tmp:
            for i, sample in enumerate(elem):                
                if i == len(residual_magnitudes):
                    residual_magnitudes.append(sample)
                else:
                    residual_magnitudes[i] = torch.cat((residual_magnitudes[i],sample))

        print(len(residual_magnitudes), "residual magnitudes", flush=True)
        print(residual_magnitudes[0].shape, "residual magnitudes[0] shape", flush=True)
        
        metrics_dict["metrics/residual_magnitude"] = self._fig_to_wandb_image(self._create_magnitude_plot(residual_magnitudes, step))
    
        logger.log_metrics(metrics_dict)

        # Update last logged batch
        self.last_batch_logged = state.timestamp.batch
        state.model.model.transformer._residual_magnitudes = []

    def state_dict(self) -> Dict[str, Any]:
        """Return callback state for checkpointing."""
        return {
            "last_batch_logged": self.last_batch_logged,
            "log_key": self.log_key,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        self.last_batch_logged = state_dict.get("last_batch_logged", -1)
    def _fig_to_wandb_image(self, fig: plt.Figure) -> wandb.Image:
        """Convert matplotlib figure to wandb Image."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Create wandb image from PIL Image
        from PIL import Image
        pil_img = Image.open(buf)
        img = wandb.Image(pil_img)
        plt.close(fig)
        return img
    def _create_magnitude_plot(self, data: list[torch.Tensor], step) -> plt.Figure:
        """Plot mean and ±1 std of a list of entropy tensors using seaborn."""

        if not data:
            # Create empty plot if no data
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.set_title("Entropy Trend (no data)", fontsize=14)
            return fig

        # Only move to CPU when necessary for plotting, keep computation on GPU
        means = [d.mean().cpu().item() for d in data]
        stds = [d.std().cpu().item() for d in data]
        maxs = [mean+std for mean, std in zip(means, stds)]
        mins = [mean-std for mean, std in zip(means, stds)]
        # x = np.arange(len(means))
        fig, ax = plt.subplots(figsize=self.figsize)
        # Plot mean line
        ax.plot(means, color='blue', label='Mean Entropy')
        ax.ticklabel_format(style='plain', axis='both') 
        # Fill ±1 std area
        ax.fill_between(range(len(mins)), mins, maxs, color='blue', alpha=0.1, label='±1 Std Dev')
        # plt.ylim(min(means)*0.9, max(means)*1.1)  # Adjust as neededs
        # Axis and formatting
        ax.set_title(f"Residual Magnitude at step {step}", fontsize=14)
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel("L2 norm", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        sns.despine()
        fig.tight_layout()

        return fig

