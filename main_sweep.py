import contextlib
import os
from typing import cast

import hydra
import torch
import wandb
from composer import Trainer
from composer.algorithms import GradientClipping
from composer.loggers import WandBLogger
from composer.optim import DecoupledAdamW
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from wandb.sdk.wandb_run import Run

from dyna.config import DynaConfig
from dyna.model import ComposerDynaModel
from dyna.utils import (
    build_full_concrete_config,
    create_param_groups,
    get_callbacks,
    get_current_git_short_hash,
    get_data_loader,
    get_scheduler,
    make_wandb_run_name,
)


def safe_clean_stale_shared_memory():
    """Clean stale shared memory objects more aggressively."""
    try:
        from streaming.base.util import clean_stale_shared_memory

        clean_stale_shared_memory()

        # Additional cleanup for PyTorch shared memory
        import os
        import subprocess

        # Clean up stale torch shared memory objects
        try:
            # List all shared memory objects and clean up torch ones
            result = subprocess.run(["ls", "/dev/shm"], capture_output=True, text=True)
            if result.returncode == 0:
                for item in result.stdout.split():
                    if item.startswith("torch_"):
                        with contextlib.suppress(OSError, FileNotFoundError):
                            os.unlink(f"/dev/shm/{item}")
        except Exception:
            pass  # Best effort cleanup

    except Exception as e:
        print(f"Warning: Could not clean shared memory: {e}")


def rebase_config(cfg_subtree: DictConfig) -> DictConfig:
    # Convert to YAML string (preserves interpolations)
    yaml_str = OmegaConf.to_yaml(cfg_subtree, resolve=False)

    # Create a fresh DictConfig from the YAML
    # This makes it a new root with no parent references
    rebased = cast(DictConfig, OmegaConf.create(yaml_str))

    return rebased


def execute_train(cfg: DictConfig, wandb_run: Run | None = None):
    print("commit hash:", get_current_git_short_hash())
    os.environ["S3_ENDPOINT_URL"] = "http://128.232.115.19:9000"

    if wandb_run is not None:
        wandb_logger = WandBLogger()
    else:
        tmpdir = os.getenv("TMPDIR") or "/tmp_000"
        unique = tmpdir.split("_")[-1]
        run_name = make_wandb_run_name(cfg.model_config, cfg.trainer_config, unique)
        wandb_logger = WandBLogger(
            project="dyna",
            log_artifacts=False,
            name=run_name,
            init_kwargs={
                "config": OmegaConf.to_container(cfg, resolve=True),
                "id": unique,
            },
        )
    cfg = build_full_concrete_config(cfg)
    print(OmegaConf.to_yaml(cfg))
    # assert wandb.run is not None
    # cfg.trainer_config.save_filename = wandb.run.name + "-ba{batch}.pt"
    cfg.trainer_config.save_filename = "test" + "-ba{batch}.pt"

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    conf = DynaConfig(**cfg.model_config)
    torch.manual_seed(42)
    model = ComposerDynaModel(config=conf, tokenizer=tokenizer)
    print(model, flush=True)
    # condition_model(
    #     model,
    #     ["model.embedding.weight", "model.lm_head.weight", "model.out_norm.weight"],
    #     "s3://loop-llm/dyna/1_baseTransformer_22oct25_X1Q0lJ_dim~768_d_ffn~3072_n_l~12_n_h~12_d_hd~64_ee~False_mode~transformer_norm~pre_rescale~none-ba3171.pt",
    # )
    safe_clean_stale_shared_memory()
    # train_dataloader = build_text_dataloader() for future
    train_dataloader = get_data_loader(
        cfg.data_config,
        tokenizer=tokenizer,
        device_train_batch_size=cfg.train.device_train_batch_size,
    )
    params = create_param_groups(
        model,
        cfg.optimizer_config.lr,
        cfg.optimizer_config.eps,
        base_depth=cfg.model_config.base_depth,
        current_depth=cfg.model_config.current_depth,
        base_width=cfg.model_config.base_width,
        current_width=cfg.model_config.current_width,
        cp_alpha=cfg.model_config.cp_alpha,
        default_wd=cfg.optimizer_config.weight_decay,
    )

    # optimizer = DecoupledAdamW(model.parameters(), lr=cfg.optimizer_config.lr, eps=cfg.optimizer_config.eps, weight_decay=cfg.optimizer_config.weight_decay)
    optimizer = DecoupledAdamW(params)
    
    scheduler = get_scheduler(cfg.scheduler_config)
    eval_dataloader = None

    loggers = [wandb_logger]
    callbacks = get_callbacks(cfg.callbacks)

    clipping_type = cfg.optimizer_config.clipping_type
    grad_clipping = GradientClipping(clipping_type=clipping_type, clipping_threshold=1)

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        callbacks=callbacks,
        optimizers=optimizer,
        schedulers=scheduler,
        parallelism_config={"fsdp_config": cfg.get("fsdp_config", {})},
        loggers=loggers,
        algorithms=[grad_clipping],
        **cfg.trainer_config,
    )

    trainer.fit()

    del trainer
    del grad_clipping
    del callbacks
    del params
    del scheduler
    del optimizer
    del eval_dataloader
    del train_dataloader
    del model
    del tokenizer


def train_with_wandb_sweep(cfg):
    print("Starting training with WandB sweep", flush=True)

    # Initialize wandb run
    tmpdir = os.getenv("TMPDIR") or "/tmp_000"
    unique = tmpdir.split("_")[-1]
    run_name = make_wandb_run_name(cfg.model_config, cfg.trainer_config, unique)
    run: Run = wandb.init(
        project="dyna",
        name=run_name,
        config=cast(dict, OmegaConf.to_container(cfg, resolve=True)),
        id=unique,
    )
    # Update config with sweep parameters from wandb
    for key, value in wandb.config.items():
        print(f"Setting {key} to {value}")
        OmegaConf.update(cfg, key, value, merge=False)

    # Resolve interpolations and execute
    resolved_config = cast(
        DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    )

    execute_train(resolved_config, wandb_run=run)
    torch._C._cuda_clearCublasWorkspaces()
    import gc

    gc.collect()
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    run.finish()
    wandb.finish()


@hydra.main(version_base=None, config_path="configs", config_name="MoA_moeut_160M")
def main(cfg: DictConfig):
    # Check if this is a wandb sweep run
    if cfg.get("sweep_config", False):
        # if False:
        sweep_config = cast(dict, OmegaConf.to_container(cfg.sweep_config))
        prior_runs = []  # List of prior run IDs to avoid duplicates
        sweep_id = wandb.sweep(sweep_config, project="dyna", prior_runs=prior_runs)

        def sweep_wrapper():
            fresh_cfg = rebase_config(cfg.execute_config)
            train_with_wandb_sweep(fresh_cfg)

        wandb.agent(
            sweep_id,
            function=sweep_wrapper,
            project="dyna",
            count=None,
        )
    else:
        execute_train(rebase_config(cfg.execute_config))


if __name__ == "__main__":
    main()
