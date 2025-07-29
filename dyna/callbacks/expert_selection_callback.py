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
import base64
from dyna.model.model import SigmaMoE, SwitchHeadCore


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
        for i in data:
            (attn_o, attn_v), ffn = i
            # check that o and v are correct order
            print(attn_o)
            print(attn_o.flatten() if attn_o is not None else torch.tensor(0))
            
            # Ensure tensors are integers for bincount, then convert to float
            attn_o_int = attn_o.flatten().int() if attn_o is not None else torch.zeros(self.attn_experts, dtype=torch.int)
            attn_v_int = attn_v.flatten().int() if attn_v is not None else torch.zeros(self.attn_experts, dtype=torch.int)
            ffn_int = ffn.flatten().int() if ffn is not None else torch.zeros(self.ffn_experts, dtype=torch.int)
            
            attn_o_counts.append(torch.bincount(attn_o_int, minlength=self.attn_experts).cpu().to(torch.float32))
            attn_v_counts.append(torch.bincount(attn_v_int, minlength=self.attn_experts).cpu().to(torch.float32))
            ffn_counts.append(torch.bincount(ffn_int, minlength=self.ffn_experts).cpu().to(torch.float32))
        
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
        """Create a heatmap showing expert selection patterns."""
        # Ensure tensor is on CPU
        selection_data = selection_data.cpu()
        
        # Convert to probabilities
        probs = F.softmax(selection_data, dim=1)
        
        # Average across batch and sequence dimensions, keep layer and expert dims
        if len(probs.shape) == 4:  # [batch, seq, heads, experts] or [time, batch, seq, experts]
            # Average over batch and sequence
            avg_probs = probs.mean(dim=(0, 1))  # [heads/time, experts] or [experts]
        elif len(probs.shape) == 3:  # [batch, seq, experts]
            avg_probs = probs.mean(dim=(0, 1))  # [experts]
        else:
            avg_probs = probs.mean(dim=0)  # Fallback
        
        # Ensure we have a 2D tensor for heatmap
        if len(avg_probs.shape) == 1:
            avg_probs = avg_probs.unsqueeze(0)
        
        # Convert to numpy
        heatmap_data = avg_probs.detach().cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            ax=ax,
            cmap='viridis',
            cbar=True,
            xticklabels=[f'Expert {i}' for i in range(heatmap_data.shape[1])],
            yticklabels=[f'Head/Step {i}' for i in range(heatmap_data.shape[0])],
            annot=heatmap_data.shape[1] <= 20,  # Only annotate if not too many experts
            fmt='.3f'
        )
        
        ax.set_title("test")
        ax.set_xlabel('Expert Index')
        ax.set_ylabel('Head/Time Step')
        
        plt.tight_layout()
        return fig

    # def _create_layer_comparison_plot(
    # Plot the entropy of the expert selection process
    #     self, 
    #     selections_by_layer: List[Dict], 
    #     title: str
    # ) -> plt.Figure:
    #     """Create a plot comparing expert usage across layers."""
    #     if not selections_by_layer:
    #         return None
            
    #     fig, ax = plt.subplots(figsize=self.figsize)
        
    #     layer_indices = []
    #     expert_entropies = []
        
    #     for layer_data in selections_by_layer:
    #         selection_tensor = layer_data['data']
    #         layer_idx = layer_data['layer_idx']
            
    #         # Convert to probabilities
    #         probs = F.softmax(selection_tensor, dim=-1)
            
    #         # Compute entropy across experts (higher = more diverse usage)
    #         log_probs = F.log_softmax(selection_tensor, dim=-1)
    #         entropy = -(probs * log_probs).sum(dim=-1).mean()
            
    #         layer_indices.append(layer_idx)
    #         expert_entropies.append(entropy.item())
        
    #     ax.plot(layer_indices, expert_entropies, 'o-', linewidth=2, markersize=8)
    #     ax.set_xlabel('Layer Index')
    #     ax.set_ylabel('Expert Selection Entropy')
    #     ax.set_title(title)
    #     ax.grid(True, alpha=0.3)
        
    #     plt.tight_layout()
    #     return fig

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
        if self.run_data is None:
            self.run_data = selections
        else:
            for key in self.run_data:
                if key in selections:
                    self.run_data[key] = torch.cat([self.run_data[key], selections[key]], dim=0)

        # batched
        heat_map_attn_o = self._create_expert_heatmap(selections["attn_o"])
        heat_map_attn_v = self._create_expert_heatmap(selections["attn_v"])
        heat_map_ffn = self._create_expert_heatmap(selections["ffn"])
        
        # Total
        print(self.run_data["ffn"].shape)
        print(self.run_data["attn_o"].shape)
        
        heat_map_attn_o_total = self._create_expert_heatmap(self.run_data["attn_o"])
        heat_map_attn_v_total = self._create_expert_heatmap(self.run_data["attn_v"])
        heat_map_ffn_total = self._create_expert_heatmap(self.run_data["ffn"])
        
        metrics_dict = {}
        metrics_dict[f"{self.log_key_prefix}/ffn_layer"] = self._fig_to_wandb_image(heat_map_ffn)
        metrics_dict[f"{self.log_key_prefix}/attn_o_layer"] = self._fig_to_wandb_image(heat_map_attn_o)
        metrics_dict[f"{self.log_key_prefix}/attn_v_layer"] = self._fig_to_wandb_image(heat_map_attn_v)
        
        # # Log all metrics
        # if metrics_dict:
        logger.log_metrics(metrics_dict)

        # Update last logged batch
        self.last_batch_logged = state.timestamp.batch

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
