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
    torch_trace_dir = "torch_traces_new"

    # trainer = Trainer(
    #     model=model,
    #     train_dataloader=train_dataloader,
    #     eval_dataloader=eval_dataloader,
    #     callbacks=callbacks,
    #     optimizers=optimizer,
    #     schedulers=scheduler,
    #     loggers=loggers,
    #     profiler=Profiler(
    #         trace_handlers=[
    #             JSONTraceHandler(folder=composer_trace_dir, overwrite=True)
    #         ],
    #         schedule=cyclic_schedule(
    #             wait=5,
    #             warmup=0,
    #             active=1,
    #             repeat=0,
    #         ),
    #         torch_prof_folder=torch_trace_dir,
    #         torch_prof_overwrite=True,
    #         torch_prof_memory_filename="memory_trace{batch}.html",
    #         torch_prof_with_stack=True,
    #         torch_prof_record_shapes=True,
    #         torch_prof_profile_memory=True
    #     ),
    #     **cfg.trainer_config,
    # )

    # trainer.fit()
    my_schedule = schedule(skip_first=1, wait=1, warmup=0, active=1, repeat=2)

    model = model.cuda()
    torch.cuda.memory._record_memory_history()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=my_schedule,
        record_shapes=True,
        profile_memory=True,
        # on_trace_ready=trace_handler,
        with_stack = True,
        with_flops = True,
        with_modules = True,
    ) as prof:
        for idx, batch in enumerate(train_dataloader.dataloader):
            batch["input_ids"] = batch["input_ids"].to("cuda")
            print(batch["input_ids"].shape)
            out = model(batch)
            prof.step()
            if idx == 6:
                break
    torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    


if __name__ == "__main__":
    main()
