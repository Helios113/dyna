import random

import hydra
import numpy as np
from transformers import AutoTokenizer
from model.modules.model_config import DynaConfig, TrainerConfig
from model.model import ComposerMoEUT, MoEUTConfig
from composer import Trainer
from streaming import StreamingDataset
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from tools.old_model import MoEUTLM_old


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
    model_schema = OmegaConf.structured(DynaConfig)
    model_config = cfg.model_config
    model_config = OmegaConf.merge(model_schema, model_config)

    seed_everything()
    model_old = MoEUTLM_old(
        model_config.vocab_size,
        model_config.d_model,
        model_config.n_layers,
        model_config.n_heads,
        model_config.n_experts_ffn,
        model_config.n_experts_attn,
        d_head=model_config.d_head,
        group_size=model_config.group_size,
        ff_k=model_config.ff_k,
        att_k=model_config.att_k,
        ff_expert_dropout=model_config.ff_expert_dropout,
        att_expert_dropout=model_config.att_expert_dropout,
        ff_expert_size=model_config.ff_expert_size,
    ).cuda()

    print(
        "Old model parameters:",
        sum(p.numel() for p in model_old.parameters() if p.requires_grad),
    )

    # Print embedding, head, and transformer parameters
    embedding_params = sum(
        p.numel()
        for n, p in model_old.named_parameters()
        if "emb" in n and p.requires_grad
    )
    head_params = sum(
        p.numel()
        for n, p in model_old.named_parameters()
        if "head" in n and p.requires_grad
    )
    transformer_params = sum(
        p.numel()
        for n, p in model_old.named_parameters()
        if ("transformer" in n or "blocks" in n) and p.requires_grad
    )

    print(f"Embedding parameters: {embedding_params}")
    print(f"Head parameters: {head_params}")
    print(f"Transformer parameters: {transformer_params}")


if __name__ == "__main__":
    main()
