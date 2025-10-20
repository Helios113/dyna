import os

import hydra
import torch
import torch.distributed as dist
from beartype import beartype
from composer import Trainer
from composer.algorithms import GradientClipping
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from omegaconf import DictConfig, OmegaConf
from streaming.base.util import clean_stale_shared_memory
from transformers import AutoTokenizer

from dyna.config import DynaConfig
from dyna.model import ComposerDynaModel
from dyna.utils import (
    build_full_concrete_config,
    get_callbacks,
    get_data_loader,
    get_scheduler,
    make_wandb_run_name,
)


def trace_handler(p):
    p.export_chrome_trace(
        "/nfs-share/pa511/code_bases/dyna_project/dyna/torch_traces_sel/trace_new"
        + str(p.step_num)
        + ".json"
    )


def safe_clean_stale_shared_memory():
    # only rank 0 (main process) initializes
    if not dist.is_initialized() or dist.get_rank() == 0:
        return clean_stale_shared_memory()


@beartype
@hydra.main(version_base=None, config_path="configs", config_name="MoA_moeut_160M")
def main(cfg: DictConfig):
    safe_clean_stale_shared_memory()

    cfg = build_full_concrete_config(cfg)
    print(OmegaConf.to_yaml(cfg))

    tmpdir = os.getenv("TMPDIR") or "/tmp_000"
    unique = tmpdir.split("_")[-1]
    run_name = make_wandb_run_name(cfg.model_config, cfg.trainer_config, unique)
    cfg.trainer_config.save_filename = run_name + "-ba{batch}.pt"
    wandb_logger = WandBLogger(
        project="dyna",
        log_artifacts=False,
        name=run_name,
        init_kwargs={"config": OmegaConf.to_container(cfg, resolve=True), "id": unique},
    )

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")

    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    # Instead of passing the DictConfig directly, unpack it as kwargs
    conf = DynaConfig(**cfg.model_config)

    torch.manual_seed(42)
    model = ComposerDynaModel(config=conf, tokenizer=tokenizer)

    train_dataloader = get_data_loader(
        cfg.data_config,
        tokenizer=tokenizer,
        device_train_batch_size=cfg.train.device_train_batch_size,
    )
    # Make optimizer
    optimizer = DecoupledAdamW(model.parameters(), lr=cfg.optimizer_config.lr)
    scheduler = get_scheduler(cfg.scheduler_config)
    eval_dataloader = None

    loggers = [wandb_logger]
    callbacks = get_callbacks(cfg.callbacks)

    clipping_type = cfg.optimizer_config.clipping_type
    gc = GradientClipping(clipping_type=clipping_type, clipping_threshold=1)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        callbacks=callbacks,
        optimizers=optimizer,
        schedulers=scheduler,
        parallelism_config={"fsdp_config": cfg.get("fsdp_config", {})},
        loggers=loggers,
        algorithms=[gc],
        **cfg.trainer_config,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
