from transformers import AutoTokenizer
from model_config import ModelConfig, TrainerConfig
from composer.loggers import WandBLogger
from model import ComposerMoEUT, MoEUTConfig
from composer import Trainer
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from composer.optim import DecoupledAdamW
from utils import make_wandb_run_name, get_callbacks, get_data_loader
@hydra.main(version_base=None, config_path="configuration", config_name="MoA")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # CONFIGS
    # Model Config
    model_schema = OmegaConf.structured(ModelConfig)
    model_config = OmegaConf.merge(model_schema, cfg.model_config)
    assert isinstance(model_config, DictConfig), "model_config should be a DictConfig instance"
    
    # Trainer Config
    trainer_schema = OmegaConf.structured(TrainerConfig)
    trainer_config = OmegaConf.merge(trainer_schema, cfg.trainer_config)
    assert isinstance(trainer_config, DictConfig), "trainer_config should be a DictConfig instance"
    
    run_name = make_wandb_run_name(model_config)

    wandb_logger = WandBLogger(project="dyna", log_artifacts=False, name=run_name)
    # We don't need a tokenizer becuase all of our data is pre-tokenized.
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    
    
    conf = MoEUTConfig(model_config)

    model = ComposerMoEUT(
        config=conf,
        tokenizer=tokenizer,
        additional_train_metrics=None,
        loss_fn=None,
    )
    
    
    # Make dataloader

    # max_seq_len -- VERY Important
    train_dataloader = get_data_loader(
        cfg,
        tokenizer,
        device_train_batch_size=2,  # Set to 2 for testing purposes
    )
    
    # Make optimizer
    optimizer = DecoupledAdamW(model.parameters())
    
    scheduler = None
    eval_dataloader = None
    
    loggers = [wandb_logger]
    callbacks = get_callbacks(cfg.callbacks)
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        callbacks=callbacks,
        optimizers=optimizer,
        schedulers=scheduler,
        loggers=loggers,
        **trainer_config,
    )
    
    trainer.fit()

if __name__ == "__main__":
    main()
