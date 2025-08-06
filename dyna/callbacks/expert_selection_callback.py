import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
import wandb
from composer.core import Callback, State, Time, TimeUnit
from composer.loggers import Logger
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


class ExpertSelectionCallback(Callback):
    """
    Callback to visualize expert selection patterns for both attention and FFN layers.
    Creates heatmaps showing which experts are selected across different layers and time steps.
    """

    def __init__(
        self,
        ffn_experts,
        attn_experts,
        log_interval: str = "1ba",
        log_key_prefix: str = "expert_selection",
        max_samples_per_batch: int = 32,
        figsize: Tuple[int, int] = (12, 8),
    ):
        """
        Initialize the expert selection visualization callback.

        Args:
            log_interval: Logging frequency in batches or as a time string. Default: "10ba"
            log_key_prefix: Prefix for logging keys. Default: "expert_selection"
            max_samples_per_batch: Maximum number of samples to visualize per batch
            figsize: Figure size for matplotlib plots
        """
        self.log_interval = (
            Time.from_timestring(log_interval)
            if isinstance(log_interval, str)
            else Time(log_interval, TimeUnit.BATCH)
        )
        self.log_key_prefix = log_key_prefix
        self.max_samples_per_batch = max_samples_per_batch
        self.figsize = figsize
        self.last_batch_logged = -1
        self.ffn_experts = ffn_experts
        self.attn_experts = attn_experts
        self.run_data = None
        self.samples_eaten = 1

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

    def _collect_expert_selections(self, model) -> Dict[str, torch.Tensor]:
        """Collect expert selection data from all layers."""
        
        attn_o_counts = []
        attn_v_counts = []
        ffn_counts = []
        
        data = model.transformer._expert_sel
        device = next(model.parameters()).device
        
        for _, batch in enumerate(data):
            for idx, elem in enumerate(batch):
                (attn_v, attn_o), ffn = elem
                if attn_v is None:
                    continue
                    
                # Keep everything on GPU - move tensors to device if needed
                if attn_o.device != device:
                    attn_o = attn_o.to(device)
                if attn_v.device != device:
                    attn_v = attn_v.to(device)
                if ffn.device != device:
                    ffn = ffn.to(device)
                
                # Ensure tensors are integers for bincount, keep on GPU
                attn_o_int = attn_o.flatten().int()
                attn_v_int = attn_v.flatten().int()
                ffn_int = ffn.flatten().int()
                
                # Use GPU operations throughout
                attn_o_count = torch.bincount(attn_o_int, minlength=self.attn_experts).to(torch.float32)
                attn_v_count = torch.bincount(attn_v_int, minlength=self.attn_experts).to(torch.float32)
                ffn_count = torch.bincount(ffn_int, minlength=self.ffn_experts).to(torch.float32)
                
                if idx == len(attn_o_counts):
                    attn_o_counts.append(attn_o_count)
                    attn_v_counts.append(attn_v_count)
                    ffn_counts.append(ffn_count)
                else:
                    attn_o_counts[idx] = attn_o_counts[idx] + attn_o_count
                    attn_v_counts[idx] = attn_v_counts[idx] + attn_v_count
                    ffn_counts[idx] = ffn_counts[idx] + ffn_count
                                                      
        expert_data = {
            "attn_o": torch.stack(attn_o_counts, dim=0),  # [num_layers, num_attn_experts]
            "attn_v": torch.stack(attn_v_counts, dim=0),  # [num_layers, num_attn_experts]
            "ffn": torch.stack(ffn_counts, dim=0)         # [num_layers, num_ffn_experts]
        }
        
        return expert_data

    def _create_expert_heatmap(
        self, 
        selection_data: torch.Tensor, 
    ) -> plt.Figure:
        """Create a heatmap showing expert selection patterns and a heatmap for mean expert selection."""
        
        # Keep computation on GPU until final conversion
        heatmap_data_gpu = selection_data.clone()
        mean_probs_gpu = selection_data.mean(dim=0)
        
        # Compute load balance on GPU
        sum_probs = torch.sum(mean_probs_gpu)
        normalized_probs = mean_probs_gpu / sum_probs
        uniform_dist = torch.ones_like(mean_probs_gpu) / mean_probs_gpu.shape[0]
        
        load_balance = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(normalized_probs, dim=0),
            uniform_dist,
            reduction="sum"
        )
        
        # Only convert to CPU for plotting
        heatmap_data = heatmap_data_gpu.detach().cpu()
        mean_probs = mean_probs_gpu.detach().cpu()
        
        # Clean up GPU tensors
        del heatmap_data_gpu, mean_probs_gpu, normalized_probs, uniform_dist
        
        # Compute common vmin and vmax for the heatmap
        common_vmin = heatmap_data.min()
        common_vmax = heatmap_data.max()
        
        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.3)
        
        # Main heatmap (per layer)
        ax1 = fig.add_subplot(gs[0])
        sns.heatmap(
            heatmap_data,
            ax=ax1,
            cmap='viridis',
            cbar=True,
            vmin=common_vmin,
            vmax=common_vmax,
            xticklabels=[f'Expert {i}' for i in range(heatmap_data.shape[1])],
            yticklabels=[f'Layer {i}' for i in range(heatmap_data.shape[0])],
            fmt='.3f'
        )
        ax1.set_title("Expert Selection by Layer")
        ax1.set_xlabel('')
        ax1.set_ylabel('Layer')
        
        # Line plot for mean expert selection
        ax2 = fig.add_subplot(gs[1])
        x = range(len(mean_probs))
        ax2.plot(x, mean_probs, marker='o', linestyle='-', color='blue')
        ax2.set_xlabel('Expert Index')
        ax2.set_ylabel('Mean Prob')
        ax2.set_title('Mean Expert Selection Across Layers')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'E{i}' for i in x])
        
        return fig, load_balance

    # Plot the expert selections
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

    def batch_end(self, state: State, logger: Logger) -> None:
        """
        Called at the end of each batch to visualize expert selections.

        Args:
            state: Composer training state
            logger: Composer logger instance
        """
        if not state.model.training or not self._should_log(state):
            return

        # Collect expert selection data
        selections = self._collect_expert_selections(state.model.model)
        device = next(state.model.parameters()).device
        
        if self.run_data is None:
            self.run_data = {}
            for key in selections.keys():
                self.run_data[key] = torch.sum(selections[key], dim=0)
        else:
            for key in self.run_data:
                if key in selections:
                    self.run_data[key] = self.run_data[key] + torch.sum(selections[key], dim=0)

        # Create visualizations
        heat_map_attn_o, load_balance_attn_o = self._create_expert_heatmap(selections["attn_o"])
        heat_map_attn_v, load_balance_attn_v = self._create_expert_heatmap(selections["attn_v"])
        heat_map_ffn, load_balance_ffn = self._create_expert_heatmap(selections["ffn"])
    
        # Compute total load balances on GPU
        sum_attn_o = torch.sum(self.run_data["attn_o"])
        sum_attn_v = torch.sum(self.run_data["attn_v"])
        sum_ffn = torch.sum(self.run_data["ffn"])
        
        load_balance_attn_o_tot = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(self.run_data["attn_o"]/sum_attn_o, dim=0),
            torch.ones_like(self.run_data["attn_o"])/self.run_data["attn_o"].shape[0],
            reduction="sum"
        )
        load_balance_attn_v_tot = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(self.run_data["attn_v"]/sum_attn_v, dim=0),
            torch.ones_like(self.run_data["attn_v"])/self.run_data["attn_v"].shape[0],
            reduction="sum"
        )
        load_balance_ffn_tot = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(self.run_data["ffn"]/sum_ffn, dim=0),
            torch.ones_like(self.run_data["ffn"])/self.run_data["ffn"].shape[0],
            reduction="sum"
        )
        
        metrics_dict = {}
        metrics_dict[f"{self.log_key_prefix}/ffn_layer"] = self._fig_to_wandb_image(heat_map_ffn)
        metrics_dict[f"{self.log_key_prefix}/attn_o_layer"] = self._fig_to_wandb_image(heat_map_attn_o)
        metrics_dict[f"{self.log_key_prefix}/attn_v_layer"] = self._fig_to_wandb_image(heat_map_attn_v)
        
        metrics_dict[f"metrics/load_balance_attn_o"] = load_balance_attn_o.item()
        metrics_dict[f"metrics/load_balance_attn_v"] = load_balance_attn_v.item()
        metrics_dict[f"metrics/load_balance_ffn"] = load_balance_ffn.item()
        
        metrics_dict[f"metrics/load_balance_attn_total_o"] = load_balance_attn_o_tot.item()
        metrics_dict[f"metrics/load_balance_attn_total_v"] = load_balance_attn_v_tot.item()
        metrics_dict[f"metrics/load_balance_total_ffn"] = load_balance_ffn_tot.item()
        
        logger.log_metrics(metrics_dict)

        # Update last logged batch
        self.last_batch_logged = state.timestamp.batch
        state.model.model.transformer._expert_sel = []

    def state_dict(self) -> Dict[str, Any]:
        """Return callback state for checkpointing."""
        return {
            "last_batch_logged": self.last_batch_logged,
            "log_key_prefix": self.log_key_prefix,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state from checkpoint."""
        self.last_batch_logged = state_dict.get("last_batch_logged", -1)
        self.log_key_prefix = state_dict.get("log_key_prefix", self.log_key_prefix)
        self.last_batch_logged = state_dict.get("last_batch_logged", -1)
        self.log_key_prefix = state_dict.get("log_key_prefix", self.log_key_prefix)
