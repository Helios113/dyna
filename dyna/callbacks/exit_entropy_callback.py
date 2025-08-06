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

class ExitEntropyCallback(Callback):
    """
    Callback to compute and log Shannon entropy of language model predictions.

    Computes H(p) = -sum(p(x_i) * log(p(x_i))) for each position in the sequence,
    then averages across the batch and logs to wandb. Also tracks per-layer entropy.
    """

    def __init__(
        self,
        log_interval: str = "1ba",
        epsilon: float = 1e-10,
        figsize: tuple[int, int] = (12, 8),
    ):
        """
        Initialize the Shannon entropy callback.

        Args:
            log_interval: Logging frequency in batches or as a time string. Default: "1ba"
            log_key: Key to use when logging to wandb. Default: "metrics/shannon_entropy"
            epsilon: Small value to add for numerical stability. Default: 1e-10
        """
        self.log_interval = (
            Time.from_timestring(log_interval)
            if isinstance(log_interval, str)
            else Time(log_interval, TimeUnit.BATCH)
        )
        self.epsilon = epsilon
        self.last_batch_logged = -1
        self.figsize = figsize
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

    def _compute_binary_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute binary entropy from logits.

        Args:
            logits: Model exit logits of shape [batch_size, seq_len]

        Returns:
            entropy: Average binary entropy across batch and sequence, scalar tensor
        """
        # Convert logits to probabilities using sigmoid (for binary classification)
        probs = torch.sigmoid(logits)
        
        # Clip probabilities for numerical stability
        probs = torch.clamp(probs, min=self.epsilon, max=1.0-self.epsilon)
        
        # Binary entropy formula: -p*log(p) - (1-p)*log(1-p)
        entropy_per_position = -probs * torch.log(probs) - (1-probs) * torch.log(1-probs)
        
        # Average across all dimensions
        mean_entropy = torch.mean(entropy_per_position, dim=1)
        fin_entropy = entropy_per_position[-1]
        
        
        return mean_entropy, fin_entropy

    def batch_end(self, state: State, logger: Logger) -> None:
        """
        Called at the end of each batch to compute and log entropy.
        """
        if not state.model.training or not self._should_log(state):
            # Always clear to avoid memory leak, even if not logging
            state.model.model.transformer._exit_logits = []
            return
        
        metrics_dict = {}
        
        batch_latentes = state.model.model.transformer._exit_logits
        batch_entropy = []
        batch_last_token_entrp = []
        
        # Keep everything on GPU
        data_proc = []
        for elem in batch_latentes:
            for i, sample in enumerate(elem):
                # Ensure tensors stay on GPU
                if not isinstance(sample, torch.Tensor):
                    sample = torch.tensor(sample, device=state.model.device)
                elif sample.device != state.model.device:
                    sample = sample.to(state.model.device)
                    
                if i == len(data_proc):
                    data_proc.append(sample)
                else:
                    data_proc[i] = torch.cat([data_proc[i], sample])
                    
        for elem in data_proc:
            entropy, last_token_entrp = self._compute_binary_entropy(elem)
            batch_entropy.append(entropy)
            batch_last_token_entrp.append(last_token_entrp)
            
        if batch_entropy:
            # Keep on GPU until final aggregation
            entropy_tensor = torch.stack(batch_entropy) if len(batch_entropy) > 1 else batch_entropy[0]
            last_token_tensor = torch.stack(batch_last_token_entrp) if len(batch_last_token_entrp) > 1 else batch_last_token_entrp[0]
            
            metrics_dict["metrics/exit_entropy"] = entropy_tensor.mean().item()
            metrics_dict["metrics/last_token_exit_entropy"] = last_token_tensor.mean().item()
            
            metrics_dict["entropy/exit_entropy"] = self._fig_to_wandb_image(self._create_entropy_plot(batch_entropy))
            metrics_dict["entropy/last_token_exit_entropy"] = self._fig_to_wandb_image(self._create_entropy_plot(batch_last_token_entrp))
        
        logger.log_metrics(metrics_dict)

        # Update last logged batch
        self.last_batch_logged = state.timestamp.batch

        # Always clear after use to avoid memory leak
        state.model.model.transformer._exit_logits = []

    def state_dict(self) -> Dict[str, Any]:
        """Return callback state for checkpointing."""
        return {
            "last_batch_logged": self.last_batch_logged,
            "log_key": self.log_key,
            "epsilon": self.epsilon,
            "total_blocks_so_far": self.total_blocks_so_far,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        self.last_batch_logged = state_dict.get("last_batch_logged", -1)
        self.log_key = state_dict.get("log_key", self.log_key)
        self.epsilon = state_dict.get("epsilon", self.epsilon)
        self.total_blocks_so_far = state_dict.get("total_blocks_so_far", 0)
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


    def _create_entropy_plot(self, data: list[torch.Tensor]) -> plt.Figure:
        """Plot mean and ±1 std of a list of entropy tensors using seaborn."""

        if not data:
            # Create empty plot if no data
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.set_title("Entropy Trend (no data)", fontsize=14)
            return fig

        # Only move to CPU when necessary for plotting, keep computation on GPU
        means_gpu = torch.stack([d.mean() for d in data]).to(torch.float32)
        stds_gpu = torch.stack([d.std() for d in data]).to(torch.float32)
        
        # Convert to CPU numpy only for matplotlib
        means = means_gpu.cpu().numpy()
        stds = stds_gpu.cpu().numpy()
        x = np.arange(len(means))

        # Clean up GPU tensors immediately
        del means_gpu, stds_gpu

        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot mean line
        sns.lineplot(x=x, y=means, ax=ax, marker='o', color='royalblue', linewidth=2.0, label='Mean Entropy')

        # Fill ±1 std area
        ax.fill_between(x, means - stds, means + stds, color='blue', alpha=0.1, label='±1 Std Dev')
        ax.set_yscale('linear')

        # Force plain (non-scientific) y-axis formatting
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
        ax.ticklabel_format(style='plain', axis='y')  # avoid 1.0e+03 notation

        # Axis and formatting
        ax.set_title("Entropy Trend", fontsize=14)
        ax.set_xlabel("Index", fontsize=12)
        ax.set_ylabel("Entropy", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

        sns.despine()
        fig.tight_layout()

        return fig


