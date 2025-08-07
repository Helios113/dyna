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
import torch


@hydra.main(version_base=None, config_path="configuration", config_name="MoA_moeut")
@beartype
def main(cfg: DictConfig):
    cfg = build_full_concrete_config(cfg)

    run_name = make_wandb_run_name(cfg.model_config)

    wandb_logger = WandBLogger(project="dyna", log_artifacts=False, name=run_name)
    # We don't need a tokenizer becuase all of our data is pre-tokenized.

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    # Instead of passing the DictConfig directly, unpack it as kwargs
    conf = DynaConfig(**cfg.model_config)

    model = ComposerDynaModel(config=conf, tokenizer=tokenizer)

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
    composer_trace_dir = "composer_profiler"
    torch_trace_dir = "torch_profiler"

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        callbacks=callbacks,
        optimizers=optimizer,
        schedulers=scheduler,
        loggers=loggers,
        # profiler=Profiler(
        #     trace_handlers=[
        #         JSONTraceHandler(folder=composer_trace_dir, overwrite=True)
        #     ],
        #     schedule=cyclic_schedule(
        #         wait=1,
        #         warmup=1,
        #         active=4,
        #         repeat=0,
        #     ),
        #     torch_prof_folder=torch_trace_dir,
        #     torch_prof_overwrite=True,
        #     torch_prof_memory_filename=None,
        #     # torch_prof_with_stack=True,
        #     # torch_prof_record_shapes=True,
        #     # torch_prof_profile_memory=True,
        # ),
        **cfg.trainer_config,
    )

    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     with_stack=True
    # ) as prof:
    #     trainer.fit()
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30), flush=True)

    trainer.fit()


if __name__ == "__main__":
    main()
