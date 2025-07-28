import torch
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
from composer.loggers.wandb_logger import WandBLogger
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
        # Track total blocks processed so far
        self.total_blocks_so_far = 0
        self.last_batch_logged = -1
        self.metric_defined = False
    def _should_log(self, state: State) -> bool:
        """Determine if it's time to log based on the log_interval."""
        if isinstance(self.log_interval, Time):
            return state.timestamp.batch != self.last_batch_logged and state.timestamp.get(self.log_interval.unit) % self.log_interval.value == 0
        return False

    def batch_end(self, state: State, logger: Logger) -> None:
        """Log layer usage information at the end of each batch."""
        if not state.model.training or not self._should_log(state):
            return
        if self.metric_defined == False:
            wandb.run.define_metric(step_metric = "layer_usage/block_index", name = "block_index")
            self.metric_defined = True
        

        # Access the transformer model
        if hasattr(state.model, "model") and hasattr(state.model.model, "transformer"):
            transformer = state.model.model.transformer
            
            # Check if layer_index_abs is tracked by the model
            if hasattr(transformer, "_layer_index_abs"):
                layer_usage = transformer._layer_index_abs
                _tau = transformer.tau.item()
                # Store layer usage for epoch statistics
                self.layer_usage_data.append(layer_usage)
                
                # Access sequence length evolution if available
                if hasattr(transformer, "_seq_len_evolve"):
                    seq_len_evolve = transformer._seq_len_evolve
                    
                    # Log sequence length at each block point on the total blocks axis
                    for i, seq_len in enumerate(seq_len_evolve):
                        block_index = self.total_blocks_so_far + i + 1  # Calculate absolute block index
                        
                        # Also log using a consistent metric name for better plotting
                        logger.log_metrics({
                            'layer_usage/seq_len_vs_blocks': i,
                            'layer_usage/block_index': block_index
                        })
                    
                    # Update total blocks so far
                    self.total_blocks_so_far += layer_usage
                
                # Log metrics dictionary
                metrics_dict = {
                    'layer_usage/tau': _tau,
                    'layer_usage/average_layers': layer_usage,
                    'layer_usage/total_blocks': self.total_blocks_so_far
                }
                logger.log_metrics(metrics_dict)
                
                # Update last logged batch
                self.last_batch_logged = state.timestamp.batch
                