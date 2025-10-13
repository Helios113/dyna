from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from matplotlib.ticker import ScalarFormatter


class LayerUsageMonitor(Callback):
    """Logs the average number of layers used per batch.

    This callback logs the average number of layers used in the MoEUT model
    during training and evaluation.
    It checks if microbatching is used and reports appropriately.

    Args:
        log_interval (Union[str, int]): Logging frequency. Default: "1ba"
    """

    def __init__(
        self, log_interval: str | int = "100ba", figsize: tuple[int, int] = (12, 8)
    ):
        """Initialize the LayerUsageMonitor callback.

        Args:
            log_interval (Union[str, int]): Logging frequency. Default: "100ba"
            figsize (tuple[int, int]): Figure size for plotting. Default: (12, 8)
        """
        super().__init__()
        self.log_interval = (
            Time.from_timestring(log_interval)
            if isinstance(log_interval, str)
            else Time(log_interval, TimeUnit.BATCH)
        )
        # Store the layer usage data between batches
        self.layer_usage_data = []
        self.block_indices = []
        # Track total blocks processed so far
        self.total_blocks_so_far = 0
        self.last_batch_logged = -1
        self.figsize = figsize

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
        """Log layer usage information at the end of each batch."""
        if not state.model.training or not self._should_log(state):
            # Always clear to avoid memory leak, even if not logging
            state.model.model.transformer._seq_len = []
            return

        transformer = state.model.model.transformer
        seq_len = transformer._seq_len
        _tau = 0
        # _tau = transformer.tau.item()

        # Store layer usage for epoch statistics - keep everything on GPU
        avg_layers = 0
        seq_lens = []

        # Keep everything on GPU and avoid unnecessary copying
        for elem in seq_len:
            avg_layers += len(elem)
            for i, sample in enumerate(elem):
                # Ensure sample stays on GPU as tensor
                if not isinstance(sample, torch.Tensor):
                    sample = torch.tensor(sample, device=state.model.device)
                elif sample.device != state.model.device:
                    sample = sample.to(state.model.device)

                if i == len(seq_lens):
                    seq_lens.append(sample.clone())
                else:
                    # Use torch.cat instead of numpy append to stay on GPU
                    seq_lens[i] = torch.cat([seq_lens[i], sample.flatten()])

        avg_layers /= len(seq_len) if len(seq_len) > 0 else 1

        metrics_dict = {
            "metrics/tau": _tau,
            "metrics/avg_layers": avg_layers,
        }

        # Only convert to CPU when absolutely necessary for plotting
        if seq_lens:
            metrics_dict["seq_length/seq_length"] = self._fig_to_wandb_image(
                self._create_entropy_plot(seq_lens)
            )

        logger.log_metrics(metrics_dict)
        self.last_batch_logged = state.timestamp.batch

        # Always clear after use to avoid memory leak
        transformer._seq_len = []

    def _fig_to_wandb_image(self, fig: plt.Figure) -> wandb.Image:
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

    def _create_entropy_plot(self, data: list[torch.Tensor]) -> plt.Figure:
        """Plot mean and ±1 std of a list of entropy tensors using seaborn."""
        # Only move to CPU when necessary for plotting, keep computation on GPU
        if not data:
            # Create empty plot if no data
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.set_title("sequence lengths (no data)", fontsize=14)
            return fig

        means_gpu = torch.stack([d.mean() for d in data])
        stds_gpu = torch.stack([d.std() for d in data])

        # Convert to CPU numpy only for matplotlib
        means = means_gpu.cpu().numpy()
        stds = stds_gpu.cpu().numpy()
        x = np.arange(len(means))

        # Clean up GPU tensors immediately
        del means_gpu, stds_gpu

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot mean line
        sns.lineplot(
            x=x,
            y=means,
            ax=ax,
            marker="o",
            color="royalblue",
            linewidth=2.0,
            label="Mean Entropy",
        )
        ax.set_yscale("linear")

        # Fill ±1 std area
        ax.fill_between(
            x, means - stds, means + stds, color="blue", alpha=0.1, label="±1 Std Dev"
        )

        # Force plain (non-scientific) y-axis formatting
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax.ticklabel_format(style="plain", axis="y")  # avoid 1.0e+03 notation

        # Axis and formatting
        ax.set_title("sequence lengths", fontsize=14)
        ax.set_xlabel("local step", fontsize=12)
        ax.set_ylabel("seq_len", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

        sns.despine()
        fig.tight_layout()

        return fig
