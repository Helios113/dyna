import random
from transformers import AutoTokenizer
from composer.loggers import WandBLogger
from dyna.model.model import ComposerDynaModel, DynaConfig
from composer import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf
from dyna.utils.utils import (
    make_wandb_run_name,
    get_callbacks,
    get_data_loader,
    build_full_concrete_config,
)
from beartype import beartype
import torch
import streaming

streaming.base.util.clean_stale_shared_memory()

@hydra.main(version_base=None, config_path="configuration", config_name="eval_config")
@beartype
def main(cfg: DictConfig):
    cfg = build_full_concrete_config(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    run_name = make_wandb_run_name(cfg.model_config, cfg.eval_config)
    wandb_logger = WandBLogger(
        project="dyna-eval", 
        log_artifacts=False, 
        name=f"eval-{run_name}", 
        init_kwargs={"config": OmegaConf.to_container(cfg, resolve=True)}
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    tokenizer.pad_token = tokenizer.eos_token

    # Create model
    conf = DynaConfig(**cfg.model_config)
    model = ComposerDynaModel(config=conf, tokenizer=tokenizer)
    
    # Get evaluation dataloader
    eval_dataloader = get_data_loader(
        cfg.data_config,
        tokenizer=tokenizer,
        device_train_batch_size=cfg.eval.device_eval_batch_size,
    )

    loggers = [wandb_logger]
    callbacks = get_callbacks(cfg.callbacks) if hasattr(cfg, 'callbacks') else []

    trainer = Trainer(
        model=model,
        train_dataloader=None,  # No training
        eval_dataloader=eval_dataloader,
        callbacks=callbacks,
        loggers=loggers,
        **cfg.eval_config,
    )

    # Run evaluation
    trainer.eval()


if __name__ == "__main__":
    main()
