from io import BytesIO
from typing import Any, cast

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from matplotlib.figure import Figure

from dyna.model import DynaLM


class ShannonEntropyCallback(Callback):
    """Callback to compute and log Shannon entropy of language model predictions.

    Computes H(p) = -sum(p(x_i) * log(p(x_i))) for each position in the sequence, then
    averages across the batch and logs to wandb. Also tracks per-layer entropy.
    """

    def __init__(
        self,
        log_interval: str | int = "100ba",
        epsilon: float = 1e-3,
        figsize: tuple[int, int] = (12, 8),
        log_key: str = "shannon_entropy",
    ):
        """Initialize the Shannon entropy callback.

        Args:
            log_interval: Logging frequency specified as a time string
            epsilon: Small value to add for numerical stability. Default: 1e-10
            figsize: Figure size for entropy plots. Default: (12, 8)
            log_key: Key for logging entropy metrics. Default: "shannon_entropy"
        """
        self.log_interval = (
            Time.from_timestring(log_interval)
            if isinstance(log_interval, str)
            else Time(log_interval, TimeUnit.BATCH)
        )
        self.epsilon = (
            torch.tensor(epsilon, dtype=torch.float16)
            if torch.cuda.is_available()
            else epsilon
        )
        self.last_batch_logged = -1
        self.figsize = figsize
        self.log_key = log_key  # Initialize log_key
        # Storage for per-layer entropy data
        # self.entropy_data = []  # Add this back for the plot
        self.block_indices = []
        self.total_blocks_so_far = 0
        self.wandb_table_data = None

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
        """Called at the end of each batch to compute and log entropy."""
        if not self._should_log(state):
            return

        metrics_dict = {}
        step = str(state.timestamp.batch)
        model: DynaLM = cast(DynaLM, state.model.model)
        batch_latentes = model.transformer._latent_vectors

        # Keep everything on GPU
        batch_entropy = []
        for elem in batch_latentes:
            for i, sample in enumerate(elem):
                if i == len(batch_entropy):
                    batch_entropy.append(sample)

                else:
                    batch_entropy[i] = torch.cat((batch_entropy[i], sample))
        if batch_entropy:
            metrics_dict["metrics/shanon_entropy"] = batch_entropy[-1].mean().item()
            try:
                metrics_dict["entropy/shanon_entropy"] = self._fig_to_wandb_image(
                    self._create_entropy_plot(batch_entropy, step)
                )
            except Exception as e:
                print(f"Error creating entropy plot: {e}, skipping plot logging.")

        logger.log_metrics(metrics_dict)

        # Update last logged batch
        self.last_batch_logged = state.timestamp.batch
        model.transformer._latent_vectors = []

    def state_dict(self) -> dict[str, Any]:
        """Return callback state for checkpointing."""
        return {
            "last_batch_logged": self.last_batch_logged,
            "log_key": self.log_key,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        self.last_batch_logged = state_dict.get("last_batch_logged", -1)
        self.log_key = state_dict.get("log_key", self.log_key)

    def _fig_to_wandb_image(self, fig: Figure) -> wandb.Image:
        """Convert matplotlib figure to wandb Image."""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)

        # Create wandb image from PIL Image
        from PIL import Image

        pil_img = Image.open(buf)
        img = wandb.Image(pil_img)
        plt.close(fig)
        return img

    def _create_entropy_plot(self, data: list[torch.Tensor], step) -> Figure:
        """Plot mean and ±1 std of a list of entropy tensors using seaborn."""
        if not data:
            # Create empty plot if no data
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.set_title("Entropy Trend (no data)", fontsize=14)
            return fig

        # Only move to CPU when necessary for plotting, keep computation on GPU
        means = [d.mean().cpu().item() for d in data]
        stds = [d.std().cpu().item() for d in data]
        maxs = [mean + std for mean, std in zip(means, stds, strict=False)]
        mins = [mean - std for mean, std in zip(means, stds, strict=False)]
        # x = np.arange(len(means))
        fig, ax = plt.subplots(figsize=self.figsize)
        # Plot mean line
        ax.plot(means, color="blue", label="Mean Entropy")
        ax.ticklabel_format(style="plain", axis="both")
        # Fill ±1 std area
        ax.fill_between(
            range(len(mins)), mins, maxs, color="blue", alpha=0.1, label="±1 Std Dev"
        )
        # plt.ylim(min(means)*0.9, max(means)*1.1)  # Adjust as neededs
        # Axis and formatting
        ax.set_title(f"Entropy Trend at step {step}", fontsize=14)
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel("Entropy", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        sns.despine()
        fig.tight_layout()

        return fig
