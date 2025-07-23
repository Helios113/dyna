import sys
import os
import yaml

# Add project root to sys.path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model_config import ModelConfig
from model import MoEUTConfig, MoEUTLM
from utils import build_full_concrete_config
import hydra
from omegaconf import DictConfig, OmegaConf

def count_parameters(cfg):
    # Embedding
    vocab_size = cfg.vocab_size
    d_model = cfg.d_model
    embedding = vocab_size * d_model

    # Output head (with bias)
    lm_head = d_model * vocab_size + vocab_size

    # Transformer layers (group_size unique layers, repeated)
    n_layers = cfg.n_layers
    group_size = cfg.group_size
    n_heads = cfg.n_heads
    d_head = cfg.d_head or (d_model // n_heads)
    ff_n_experts = getattr(cfg, "n_ffn_experts", getattr(cfg, "ff_n_experts", None))
    att_n_experts = getattr(cfg, "n_att_experts", getattr(cfg, "att_n_experts", None))
    ff_expert_size = cfg.ff_expert_size
    ff_k = getattr(cfg, "ff_k", 8)
    att_k = getattr(cfg, "att_k", 2)

    # --- Per-layer parameter count (matches MoEUTLayer) ---
    # Attention (SwitchHeadRope)
    att_q = d_model * d_head * n_heads
    att_k = d_model * d_head * n_heads
    att_sel_v = n_heads * att_n_experts * d_model
    att_sel_o = n_heads * att_n_experts * d_model
    att_v = n_heads * att_n_experts * d_model * d_head
    att_o = n_heads * att_n_experts * d_head * d_model
    att_total = att_q + att_k + att_sel_v + att_sel_o + att_v + att_o

    # FFN (SigmaMoE)
    ff_keys = ff_n_experts * d_model * ff_expert_size
    ff_values = ff_n_experts * ff_expert_size * d_model
    ff_expert_sel = ff_n_experts * d_model
    ffn_total = ff_keys + ff_values + ff_expert_sel

    # 4 RMSNorms per layer (each d_model, no bias)
    rmsnorm = 4 * d_model

    per_layer = att_total + ffn_total + rmsnorm

    # Only group_size unique layers, repeated n_layers times
    transformer = per_layer * group_size

    # Add out_norm (RMSNorm) parameter count
    out_norm = d_model

    total = embedding + lm_head + transformer + out_norm

    return {
        "embedding": embedding,
        "lm_head": lm_head,
        "transformer": transformer,
        "out_norm": out_norm,
        "total": total
    }

def print_param_breakdown(model):
    print("\nParameter breakdown by module (real):")
    for name, module in model.named_modules():
        if name == "":
            continue
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        if param_count > 0:
            print(f"{name:<40} {param_count:>12,}")

@hydra.main(version_base=None, config_path="../configuration", config_name="MoA")
def main(cfg: DictConfig):
    # Build full concrete config (merges model, trainer, data, etc.)
    full_cfg = build_full_concrete_config(cfg)
    # Use model_config from DictConfig (OmegaConf object)
    model_cfg = full_cfg.model_config
    # Convert to MoEUTConfig (inherits from PretrainedConfig)
    # OmegaConf objects can be passed as dicts using **
    hf_cfg = MoEUTConfig(**model_cfg)
    # Instantiate model
    model = MoEUTLM(hf_cfg)
    # Count parameters (real)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embedding_params = sum(p.numel() for n, p in model.named_parameters() if 'embedding' in n)
    lm_head_params = sum(p.numel() for n, p in model.named_parameters() if 'lm_head' in n)
    transformer_params = sum(p.numel() for p in model.transformer.parameters())

    # Get d_model from config for out_norm
    d_model = model_cfg.d_model
    # out_norm is a single RMSNorm layer with d_model parameters
    out_norm_params = sum(p.numel() for n, p in model.named_parameters() if 'out_norm' in n)

    # Analytical count
    param_counts = count_parameters(OmegaConf.to_container(model_cfg, resolve=True))

    print("Parameter count comparison:")
    print(f"{'Component':<15} {'Real':>15} {'Analytical':>15}")
    print(f"{'-'*45}")
    print(f"{'Embedding':<15} {embedding_params:>15,} {param_counts['embedding']:>15,}")
    print(f"{'LM Head':<15} {lm_head_params:>15,} {param_counts['lm_head']:>15,}")
    print(f"{'Transformer':<15} {transformer_params:>15,} {param_counts['transformer']:>15,}")
    print(f"{'out_norm':<15} {out_norm_params:>15,} {param_counts['out_norm']:>15,}")
    print(f"{'-'*45}")
    print(f"{'Total':<15} {total_params:>15,} {param_counts['total']:>15,}")
    print(f"{'Trainable':<15} {trainable_params:>15,}")

    print_param_breakdown(model)

if __name__ == "__main__":
    main()
    print(f"{'Component':<15} {'Real':>15} {'Analytical':>15}")
    print(f"{'-'*45}")
    print(f"{'Embedding':<15} {embedding_params:>15,} {param_counts['embedding']:>15,}")
    print(f"{'LM Head':<15} {lm_head_params:>15,} {param_counts['lm_head']:>15,}")
    print(f"{'Transformer':<15} {transformer_params:>15,} {param_counts['transformer']:>15,}")
    print(f"{'out_norm':<15} {out_norm_params:>15,} {param_counts['out_norm']:>15,}")
    print(f"{'-'*45}")
    print(f"{'Total':<15} {total_params:>15,} {param_counts['total']:>15,}")
    print(f"{'Trainable':<15} {trainable_params:>15,}")

    print_param_breakdown(model)

if __name__ == "__main__":
    main()
