import torch
import torch.nn.functional as F
from typing import Any, Dict
import wandb
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
import numpy as np

class ExitEntropyCallback(Callback):
    """
    Callback to compute and log Shannon entropy of language model predictions.

    Computes H(p) = -sum(p(x_i) * log(p(x_i))) for each position in the sequence,
    then averages across the batch and logs to wandb. Also tracks per-layer entropy.
    """

    def __init__(
        self,
        log_interval: str = "1ba",
        log_key: str = "metrics/exit_entropy",
        log_key_batch: str = "metrics/batch_exit_entropy",
        log_key_seq: str = "metrics/seq_exit_entropy",  # Add this back
        epsilon: float = 1e-10,
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
        self.log_key = log_key
        self.log_key_batch = log_key_batch
        self.log_key_seq = log_key_seq  # Add this back
        self.epsilon = epsilon
        self.last_batch_logged = -1

        # Storage for per-layer entropy data
        self.entropy_data = []  # Add this back for the plot
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
        mean_entropy = torch.mean(entropy_per_position)
        
        return mean_entropy

    def batch_end(self, state: State, logger: Logger) -> None:
        """
        Called at the end of each batch to compute and log entropy.

        Args:
            state: Composer training state
            logger: Composer logger instance
        """
        if not state.model.training or not self._should_log(state):
            return
        
        metrics_dict = {}
        
        batch_logits = state.model.model.transformer._exit_logits
        batch_entropy = []
        entropy = None
        
        for elem in batch_logits:
            entropy = self._compute_binary_entropy(elem).item()
            batch_entropy.append(entropy)
            self.block_indices.append(self.total_blocks_so_far)
            self.total_blocks_so_far += 1
        metrics_dict[self.log_key] = entropy
        self.entropy_data.extend(batch_entropy)  # Add this back for the plot
        

        # Create a numpy array for the current batch
        batch_size = len(batch_entropy)
        batch_data = np.zeros((batch_size, 4))  # 4 columns: entropy, local_steps, global_steps, batch_index
        batch_data[:, 0] = batch_entropy  # entropy
        batch_data[:, 1] = np.arange(batch_size)  # local_steps
        batch_data[:, 2] = np.arange(self.total_blocks_so_far - batch_size, self.total_blocks_so_far)  # global_steps
        batch_data[:, 3] = state.timestamp.batch  # batch_index
        
        # Create wandb table from the numpy data
        if self.wandb_table_data is None:
            table_data = [[row[0], int(row[1]), int(row[2]), int(row[3])] for row in batch_data]
            self.wandb_table_data = wandb.Table(
                data=table_data,
                columns=["entropy", "local_steps", "global_steps", "batch_index"]
            )
        else:
            for row in batch_data:
                self.wandb_table_data.add_data(row[0], int(row[1]), int(row[2]), int(row[3]))
        
        metrics_dict[self.log_key_batch] = self.wandb_table_data
        
        # Add entropy plot
        metrics_dict[self.log_key_seq] = wandb.plot.line_series(
            xs=self.block_indices,
            ys=[self.entropy_data],
            keys=["entropy"],
            title="Per-Layer Exit Entropy",
            xname="Block Index",
        )

        logger.log_metrics(metrics_dict)

        # Update last logged batch
        self.last_batch_logged = state.timestamp.batch

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