"""Evaluation script for Dyna models based on LLM Foundry eval.py.

This script provides functionality to load and evaluate Dyna models,
following the pattern from mosaicml/llm-foundry.

Usage with Hydra:
    python eval.py                                    # Uses default config
    python eval.py --config-name=eval_example         # Uses specific config
    python eval.py device=cpu precision=fp32          # Override parameters
"""

import logging
import time
from typing import Any

import hydra
import torch
from composer import Trainer
from composer.core import Callback
from composer.loggers import WandBLogger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from dyna.config import DynaConfig
from dyna.model import ComposerDynaModel

log = logging.getLogger(__name__)


def build_tokenizer(
    tokenizer_name: str,
    tokenizer_kwargs: dict[str, Any],
) -> PreTrainedTokenizerBase:
    """Build a tokenizer from HuggingFace.

    Args:
        tokenizer_name: Name or path of the tokenizer
        tokenizer_kwargs: Additional kwargs for tokenizer

    Returns:
        PreTrainedTokenizerBase: The loaded tokenizer
    """
    log.info(f"Building tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_composer_model(
    model_config: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    init_device: str = "cpu",
) -> ComposerDynaModel:
    """Build a ComposerDynaModel.

    Args:
        model_config: Configuration for the model
        tokenizer: Tokenizer to use with the model
        init_device: Device to initialize the model on

    Returns:
        ComposerDynaModel: The initialized model
    """
    log.info("Building Dyna model...")

    # Create DynaConfig from model_config
    dyna_config = DynaConfig(**model_config)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Build the model
    model = ComposerDynaModel(config=dyna_config, tokenizer=tokenizer)

    return model


def build_callbacks(callback_configs: dict[str, Any] | None) -> list[Callback]:
    """Build callbacks from config.

    Args:
        callback_configs: Dictionary of callback configurations

    Returns:
        list[Callback]: List of initialized callbacks
    """
    callbacks = []
    if callback_configs:
        # Import here to avoid circular dependencies
        from dyna.utils import get_callbacks

        callbacks = get_callbacks(callback_configs)
    return callbacks


def build_loggers(logger_configs: dict[str, Any] | None) -> list[LoggerDestination]:
    """Build loggers from config.

    Args:
        logger_configs: Dictionary of logger configurations

    Returns:
        list[LoggerDestination]: List of initialized loggers
    """
    loggers = []
    if logger_configs:
        for name, logger_cfg in logger_configs.items():
            if name.lower() == "wandb":
                loggers.append(WandBLogger(**logger_cfg))
            # Add more logger types as needed
    return loggers


def evaluate_model(
    model_name: str,
    model_config: dict[str, Any],
    tokenizer_config: dict[str, Any],
    load_path: str | None = None,
    evaluators: list | None = None,
    callbacks: list[Callback] | None = None,
    loggers: list[LoggerDestination] | None = None,
    precision: str = "amp_bf16",
    seed: int = 42,
    dist_timeout: float | int = 300.0,
    run_name: str | None = None,
    device: str = "gpu",
) -> Trainer:
    """Evaluate a single model.

    Args:
        model_name: Name identifier for the model
        model_config: Model configuration dictionary
        tokenizer_config: Tokenizer configuration dictionary
        load_path: Path to checkpoint to load
        evaluators: List of evaluators for evaluation
        callbacks: List of callbacks
        loggers: List of loggers
        precision: Training precision
        seed: Random seed
        dist_timeout: Distributed timeout
        run_name: Name for the run
        device: Device to use

    Returns:
        Trainer: The trainer object after evaluation
    """
    log.info(f"Evaluating model: {model_name}")

    # Build tokenizer
    tokenizer_name = tokenizer_config.get("name", "HuggingFaceTB/SmolLM2-1.7B")
    tokenizer_kwargs = tokenizer_config.get("kwargs", {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    # Build model
    init_device = model_config.pop("init_device", device)
    model = build_composer_model(model_config, tokenizer, init_device)

    # Build trainer
    log.info(f"Building trainer for {model_name}...")
    trainer = Trainer(
        run_name=run_name or model_name,
        seed=seed,
        model=model,
        callbacks=callbacks or [],
        loggers=loggers or [],
        precision=precision,
        load_path=load_path,
        load_weights_only=True if load_path else False,
        progress_bar=True,
        log_to_console=True,
        dist_timeout=dist_timeout,
        device=device,
    )

    # Run evaluation if evaluators provided
    if evaluators:
        log.info(f"Starting eval for {model_name}...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()
        trainer.eval(eval_dataloader=evaluators)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time
        log.info(f"Ran {model_name} eval in: {elapsed_time:.2f} seconds")

    return trainer


def evaluate(cfg: DictConfig) -> list[Trainer]:
    """Main evaluation function.

    Args:
        cfg: Configuration for evaluation

    Returns:
        list[Trainer]: List of trainers (one per model evaluated)
    """
    # Initialize distributed
    dist.initialize_dist(get_device(None), timeout=cfg.get("dist_timeout", 300.0))

    # Set up logging
    python_log_level = cfg.get("python_log_level", "INFO")
    logging.basicConfig(
        format=f"%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
        level=getattr(logging, python_log_level.upper()),
        force=True,
    )

    # Set seed
    seed = cfg.get("seed", 42)
    reproducibility.seed_all(seed)

    # Build loggers
    logger_configs = cfg.get("loggers", {})
    loggers = build_loggers(logger_configs)

    # Build callbacks
    callback_configs = cfg.get("callbacks", {})
    callbacks = build_callbacks(callback_configs)

    # Get model configurations
    # Support both single model and multiple models
    if "model" in cfg and "models" not in cfg:
        # Single model format (training style)
        model_configs = [
            {
                "model_name": cfg.get(
                    "model_name", cfg.model.get("name", "dyna_model")
                ),
                "model": cfg.model,
                "tokenizer": cfg.tokenizer,
                "load_path": cfg.get("load_path", None),
            }
        ]
    elif "models" in cfg:
        # Multiple models format (eval style)
        model_configs = cfg.models
    else:
        raise ValueError("Configuration must contain either 'model' or 'models' key")

    # Evaluate each model
    trainers = []
    for model_cfg in model_configs:
        model_name = model_cfg.get("model_name", "model")
        model_config = dict(model_cfg["model"])
        tokenizer_config = model_cfg["tokenizer"]
        load_path = model_cfg.get("load_path", None)

        trainer = evaluate_model(
            model_name=model_name,
            model_config=model_config,
            tokenizer_config=tokenizer_config,
            load_path=load_path,
            evaluators=None,  # Add evaluator building here if needed
            callbacks=callbacks,
            loggers=loggers,
            precision=cfg.get("precision", "amp_bf16"),
            seed=seed,
            dist_timeout=cfg.get("dist_timeout", 300.0),
            run_name=cfg.get("run_name", None),
            device=cfg.get("device", "gpu"),
        )

        trainers.append(trainer)
        log.info(f"Completed evaluation for {model_name}")

    return trainers


@hydra.main(version_base=None, config_path="configs", config_name="eval_example")
def main(cfg: DictConfig) -> list[Trainer]:
    """Main entry point for evaluation script using Hydra.

    Args:
        cfg: Configuration loaded by Hydra from configs directory

    Returns:
        list[Trainer]: List of trainers from evaluation

    Examples:
        # Use default config
        python eval.py

        # Use specific config file
        python eval.py --config-name=my_eval_config

        # Override parameters
        python eval.py device=cpu precision=fp32
        python eval.py load_path=/path/to/checkpoint.pt

        # Multiple overrides
        python eval.py device=cpu precision=fp32 seed=123
    """
    log.info("Starting Dyna model evaluation")
    log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Run evaluation
    trainers = evaluate(cfg)

    log.info(f"Evaluation complete. Evaluated {len(trainers)} model(s).")

    return trainers


if __name__ == "__main__":
    main()
