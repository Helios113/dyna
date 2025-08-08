import sys
import os
import yaml
import torch
from transformers import AutoTokenizer
from torchinfo import summary

# Add project root to sys.path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model.model import DynaConfig, ComposerDynaModel
from dyna.utils.utils import build_full_concrete_config
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../configuration", config_name="Transformer")
def main(cfg: DictConfig):
    # Build full concrete config (merges model, trainer, data, etc.)
    full_cfg = build_full_concrete_config(cfg)
    # Use model_config from DictConfig (OmegaConf object)
    model_cfg = full_cfg.model_config
    
    # Convert OmegaConf to raw dict to avoid enum type conflicts
    model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)
    
    # Convert to MoEUTConfig (inherits from PretrainedConfig)
    hf_cfg = DynaConfig(**model_cfg_dict)
    # Instantiate model and move to CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    model = ComposerDynaModel(hf_cfg, tokenizer).to(device)
    
    # Get batch size and sequence length from config
    batch_size = getattr(model_cfg, "batch_size", 6)
    seq_length = getattr(model_cfg, "max_seq_len", 1024)

    # Create input shape for torchinfo
    input_size = (batch_size, seq_length)
    data = torch.randint(0, hf_cfg.vocab_size, input_size, device=device)
    print("=" * 80)
    print("MODEL SUMMARY WITH TORCHINFO")
    print("=" * 80)
    
    print(f"Total DataSize: {data.numel()*8/1000000}MB")

   
    # print("eval1")
    
    # output = model({"input_ids": torch.randint(0, hf_cfg.vocab_size, input_size, device=device)})
    # with torch.no_grad():
    #     print("eval")
    #     output = model({"input_ids": torch.randint(0, hf_cfg.vocab_size, input_size, device=device)})
    # model.train()
    # print("train2")
    # output = model({"input_ids": torch.randint(0, hf_cfg.vocab_size, input_size, device=device)})
    # loss = model.loss(output,{"input_ids": torch.randint(0, hf_cfg.vocab_size, input_size, device=device)})
    # loss.backward()
    # for name, param in model.named_parameters(recurse=True):
    #     print(name, param.grad is not None)
    # Use torchinfo summary for comprehensive model profiling
    model_summary = summary(
        model,
        input_data=[{"input_ids": torch.randint(0, hf_cfg.vocab_size, input_size, device=device)}],
        depth=3,  # Show 3 levels of nesting
        col_names=["input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"],
        row_settings=["var_names"],
        verbose=1,
        device=device
    )
    print(model_summary)
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Extract key statistics from the summary
    total_params = model_summary.total_params
    trainable_params = model_summary.trainable_params
    total_mult_adds = model_summary.total_mult_adds
    
    
    # Compute-optimal token count (20 tokens per parameter)
    compute_optimal_tokens = total_params * 20
    print(f"\nCompute-optimal token count (20 per parameter): {compute_optimal_tokens:,}")
    
    # Compute number of steps for training
    tokens_per_step = batch_size * seq_length
    if tokens_per_step > 0:
        steps = (compute_optimal_tokens + tokens_per_step - 1) // tokens_per_step
        print(f"Steps needed for batch_size={batch_size}, seq_length={seq_length}: {steps:,}")
    else:
        print("Batch size or sequence length is zero; cannot compute steps.")
    
if __name__ == "__main__":
    main()