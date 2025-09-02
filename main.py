import random
from transformers import AutoTokenizer
from composer.loggers import WandBLogger
from dyna.model.model import ComposerDynaModel, DynaConfig
from composer import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf
from composer.optim import DecoupledAdamW
from composer.profiler import JSONTraceHandler, cyclic_schedule
from composer.profiler.profiler import Profiler
from dyna.utils.utils import (
    make_wandb_run_name,
    get_callbacks,
    get_data_loader,
    build_full_concrete_config,
    get_scheduler,
)
from beartype import beartype
from torch.profiler import profile, ProfilerActivity, record_function
from torch.profiler import schedule
import torch

def trace_handler(p):
    p.export_chrome_trace(
        "/nfs-share/pa511/code_bases/dyna_project/dyna/torch_traces_sel/trace_new"
        + str(p.step_num)
        + ".json"
    )
from composer.algorithms import GradientClipping
from composer.trainer import Trainer

import streaming
streaming.base.util.clean_stale_shared_memory()
@hydra.main(version_base=None, config_path="configuration", config_name="MoA_moeut")
@beartype
def main(cfg: DictConfig):
    cfg = build_full_concrete_config(cfg)
    print(OmegaConf.to_yaml(cfg))
    run_name = make_wandb_run_name(cfg.model_config, cfg.trainer_config)
    cfg.trainer_config.save_filename = run_name+"-ba{batch}.pt"
    wandb_logger = WandBLogger(project="dyna", log_artifacts=False, name=run_name, init_kwargs={"config": OmegaConf.to_container(cfg, resolve=True)})
    # We don't need a tokenizer becuase all of our data is pre-tokenized.
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    # Instead of passing the DictConfig directly, unpack it as kwargs
    conf = DynaConfig(**cfg.model_config)
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
    
    composer_trace_dir = "composer_profiler"
    torch_trace_dir = "torch_traces_new"
    
    
    clipping_type = 'norm' # can also be 'adaptive' or 'value'
    gc = GradientClipping(clipping_type=clipping_type, clipping_threshold=1)


    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        callbacks=callbacks,
        optimizers=optimizer,
        schedulers=scheduler,
        loggers=loggers,
        algorithms=[gc],
        **cfg.trainer_config,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
