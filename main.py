import os
from collections.abc import Generator, Iterator
from copy import copy
from typing import Any, cast

import hydra
import torch
import torch.distributed as dist
from composer import Trainer
from composer.algorithms import GradientClipping
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from omegaconf import DictConfig, OmegaConf
from streaming.base.util import clean_stale_shared_memory
from transformers import AutoTokenizer

from dyna.config import DynaConfig, SweepConfig
from dyna.model import ComposerDynaModel
from dyna.utils import (
    build_full_concrete_config,
    create_param_groups_with_conditional_wd,
    get_callbacks,
    get_current_git_short_hash,
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


def generate_param(cfg: DictConfig) -> Iterator[tuple[str, Any]]:
    # Returns an executre config for training

    sweeps = cfg.sweeps
    for i in sweeps:
        min_val = sweeps[i].min_val
        max_val = sweeps[i].max_val
        step_size = sweeps[i].step_size
        param = sweeps[i].name
        count = 0
        local_steps = torch.arange(min_val, max_val, step_size)
        print(local_steps)
        while count < len(local_steps):
            print("val updated to", local_steps[count].item())
            yield (param, local_steps[count].item())
            count += 1


def rebase_config(cfg_subtree: DictConfig) -> DictConfig:
    # Convert to YAML string (preserves interpolations)
    yaml_str = OmegaConf.to_yaml(cfg_subtree, resolve=False)

    # Create a fresh DictConfig from the YAML
    # This makes it a new root with no parent references
    rebased = cast(DictConfig, OmegaConf.create(yaml_str))

    return rebased


def handle_params(sweep_config, execute_config) -> Generator[DictConfig]:
    for i in generate_param(sweep_config):
        run_confg = copy.deepcopy(execute_config)
        param_name, param_value = i

        OmegaConf.update(run_confg, param_name, param_value, merge=False)

        yield run_confg


def execute_train(cfg: DictConfig):
    safe_clean_stale_shared_memory()
    print(get_current_git_short_hash())

    cfg = build_full_concrete_config(cfg)
    print(OmegaConf.to_yaml(cfg))
    os.environ["S3_ENDPOINT_URL"] = "http://128.232.115.19:9000"

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

    conf = DynaConfig(**cfg.model_config)
    torch.manual_seed(42)
    model = ComposerDynaModel(config=conf, tokenizer=tokenizer)
    # condition_model(
    #     model,
    #     ["model.embedding.weight", "model.lm_head.weight", "model.out_norm.weight"],
    #     "s3://loop-llm/dyna/1_baseTransformer_22oct25_X1Q0lJ_dim~768_d_ffn~3072_n_l~12_n_h~12_d_hd~64_ee~False_mode~transformer_norm~pre_rescale~none-ba3171.pt",
    # )
    print("model_structure  ", model, flush=True)
    train_dataloader = get_data_loader(
        cfg.data_config,
        tokenizer=tokenizer,
        device_train_batch_size=cfg.train.device_train_batch_size,
    )
    # Make optimizer
    # params = create_param_groups_with_conditional_wd(
    #     model,
    #     ["attn_pre", "attn_post", "ffn_pre", "ffn_post", "out_norm"],
    #     frozen_param_names=[
    #         "model.embedding.weight",
    #         "model.lm_head.weight",
    #         "model.out_norm.weight",
    #     ],
    # )

    params = create_param_groups_with_conditional_wd(
        model,
        # ["attn_pre", "attn_post", "ffn_pre", "ffn_post", "out_norm"],
        [],
        frozen_param_names=[],
    )
    optimizer = DecoupledAdamW(params, lr=cfg.optimizer_config.lr)
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


@hydra.main(version_base=None, config_path="configs", config_name="MoA_moeut_160M")
def main(cfg: DictConfig):
    # Model Config
    sweep_schema = OmegaConf.structured(SweepConfig)
    # Set the parent to None to make it act as root
    execute_config = rebase_config(cfg.execute_config)

    sweep_config = OmegaConf.merge(sweep_schema, cfg.sweep_config)
    for i in handle_params(sweep_config, execute_config):
        print(i)
        execute_train(i)
        # print(i)


if __name__ == "__main__":
    main()
