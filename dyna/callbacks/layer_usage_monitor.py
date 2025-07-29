import torch
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from typing import Any, Dict, Optional, Union
import wandb

class LayerUsageMonitor(Callback):
    """Logs the average number of layers used per batch.

    This callback logs the average number of layers (layer_index_abs) used in the MoEUT model
    during training and evaluation. It checks if microbatching is used and reports appropriately.

    Args:
        log_interval (Union[str, int]): Logging frequency in batches or as a time string. Default: "1ba"
    """

    def __init__(self, log_interval: Union[str, int] = "1ba"):
        super().__init__()
        self.log_interval = Time.from_timestring(log_interval) if isinstance(log_interval, str) else Time(log_interval, TimeUnit.BATCH)
        # Store the layer usage data between batches
        self.layer_usage_data = []
        self.block_indices = []
        # Track total blocks processed so far
        self.total_blocks_so_far = 0
        self.last_batch_logged = -1
        self.metric_defined = False
        self.seq_len_data = []
    def _should_log(self, state: State) -> bool:
        """Determine if it's time to log based on the log_interval."""
        if isinstance(self.log_interval, Time):
            return state.timestamp.batch != self.last_batch_logged and state.timestamp.get(self.log_interval.unit) % self.log_interval.value == 0
        return False

    def batch_end(self, state: State, logger: Logger) -> None:
        """Log layer usage information at the end of each batch."""
        if not state.model.training or not self._should_log(state):
            return
     

        transformer = state.model.model.transformer
            
        # Check if layer_index_abs is tracked by the model
        if hasattr(transformer, "_layer_index_abs"):
            layer_usage = transformer._layer_index_abs
            _tau = transformer.tau.item()
            # Store layer usage for epoch statistics
            self.layer_usage_data.append(layer_usage)
            # Log other metrics with regular steps (default behavior)
            
            # Access sequence length evolution if available
            if hasattr(transformer, "_seq_len_evolve"):
                seq_len_evolve = transformer._seq_len_evolve
                
                # Log sequence length with custom step metric
                for i, seq_len in enumerate(seq_len_evolve):
                    print(seq_len)
                    self.seq_len_data.append(seq_len)
                    self.block_indices.append(self.total_blocks_so_far)
                    self.total_blocks_so_far += 1

                metrics_dict = {
                    'layer/tau': _tau,
                    'layer/layers_activated': layer_usage,
                    'layer/total_blocks': self.total_blocks_so_far,
                    "layer/seq_len": wandb.plot.line_series(
                            xs=self.block_indices,
                            ys=[self.seq_len_data],
                            keys=["seq_len"],
                            title="Sequence Length in transformer depth",
                            xname="Block Index",
                        )
                }
                logger.log_metrics(metrics_dict)
    
            # Update last logged batch
            self.last_batch_logged = state.timestamp.batch
