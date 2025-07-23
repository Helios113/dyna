import torch
from old_model import MoEUTLM_old
import profiles
import random
import numpy as np
from transformers import AutoTokenizer
from model_config import ModelConfig, TrainerConfig
from model import ComposerMoEUT, MoEUTConfig
from composer import Trainer
from streaming import StreamingDataset
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from composer.optim import DecoupledAdamW
from data.stream_text_data import StreamingTextDataset

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="configuration", config_name="MoA")
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    
    # max_seq_len -- VERY Important
    
    
    # CONFIGS
    # Model Config
    model_schema = OmegaConf.structured(ModelConfig)
    model_config = cfg.model_config
    model_config = OmegaConf.merge(model_schema, model_config)

    tokens = torch.randint(0, model_config.vocab_size, (2, model_config.max_seq_len)).cuda()

    conf = MoEUTConfig(model_config)
    seed_everything()
    model = ComposerMoEUT(
        config=conf,
        tokenizer=None,
        additional_train_metrics=None,
        loss_fn=None,
    ).cuda()
    seed_everything()
    model_old = MoEUTLM_old(model_config.vocab_size,model_config.d_model, model_config.n_layers, model_config.n_heads,
                            model_config.ff_n_experts, model_config.att_n_experts, d_head=model_config.d_head,
                            group_size=model_config.group_size, ff_k=model_config.ff_k, att_k=model_config.att_k,
                            ff_expert_dropout=model_config.ff_expert_dropout, att_expert_dropout=model_config.att_expert_dropout,
                            ff_expert_size=model_config.ff_expert_size).cuda()
    
    
    a = model({"input_ids": tokens})
    b = model_old(tokens)
    
    print(a.logits.shape, b.outputs.shape)
    print(torch.norm(a.logits-b.outputs))
    print(torch.allclose(a.logits, b.outputs, atol=1e-6))
if __name__ == "__main__":
    main()
