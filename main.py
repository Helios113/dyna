from transformers import AutoTokenizer
from composer.loggers import WandBLogger
from dyna.model.model import ComposerMoEUT, MoEUTConfig
from composer import Trainer
from composer.callbacks import ActivationMonitor
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from composer.optim import DecoupledAdamW
from dyna.utils.utils import (
    make_wandb_run_name,
    get_callbacks,
    get_data_loader,
    build_full_concrete_config,
    get_scheduler,
)


@hydra.main(version_base=None, config_path="configuration", config_name="MoA")
def main(cfg: DictConfig):
    cfg = build_full_concrete_config(cfg)

    run_name = make_wandb_run_name(cfg.model_config)

    wandb_logger = WandBLogger(project="dyna", log_artifacts=False, name=run_name)
    # We don't need a tokenizer becuase all of our data is pre-tokenized.

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    # Instead of passing the DictConfig directly, unpack it as kwargs
    conf = MoEUTConfig(**cfg.model_config)

    model = ComposerMoEUT(
        config=conf,
        tokenizer=tokenizer
    )

    train_dataloader = get_data_loader(
        cfg.data_config,
        tokenizer,
        device_train_batch_size=cfg.train.device_train_batch_size,
    )

    # Make optimizer
    optimizer = DecoupledAdamW(model.parameters())

    scheduler = get_scheduler(cfg.scheduler_config)
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
        **cfg.trainer_config,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
