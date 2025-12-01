"""Evaluation script for Dyna models based on LLM Foundry eval.py.

This script provides functionality to load and evaluate Dyna models,
following the pattern from mosaicml/llm-foundry.

Usage with Hydra:
    python eval.py                                    # Uses default config
    python eval.py --config-name=eval_example         # Uses specific config
    python eval.py device=cpu precision=fp32          # Override parameters
"""

import logging
import os
import time
from typing import Any

import hydra
import pandas as pd
import torch
from composer import Trainer
from composer.core import Callback
from composer.loggers import WandBLogger
from composer.loggers.logger_destination import LoggerDestination
from composer.utils import dist, get_device, reproducibility
from llmfoundry.utils.builders import (
    build_tokenizer as llm_build_tokenizer,
)
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from dyna.config import DynaConfig
from dyna.model import ComposerDynaModel
from dyna.utils import build_full_concrete_config

log = logging.getLogger(__name__)


def build_tokenizer(
    tokenizer_name: str,
    tokenizer_kwargs: dict[str, Any],
) -> PreTrainedTokenizerBase:
    """Build a tokenizer from HuggingFace or llmfoundry.

    Args:
        tokenizer_name: Name or path of the tokenizer
        tokenizer_kwargs: Additional kwargs for tokenizer

    Returns:
        PreTrainedTokenizerBase: The loaded tokenizer
    """
    log.info(f"Building tokenizer: {tokenizer_name}")
    try:
        # Try using llmfoundry builder first
        tokenizer = llm_build_tokenizer(tokenizer_name, tokenizer_kwargs)
    except Exception:
        # Fall back to direct HuggingFace loading
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


def build_icl_evaluators(
    icl_tasks: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    device_eval_batch_size: int,
    icl_seq_len: int,
    destination_dir: str = "./eval_data",
):
    """Build ICL evaluators as a generator to avoid memory issues.

    Args:
        icl_tasks: List of validated ICL task configurations
        tokenizer: Tokenizer for the model
        device_eval_batch_size: Batch size for evaluation
        icl_seq_len: Maximum sequence length
        destination_dir: Directory to cache eval data

    Yields:
        Evaluator: Individual evaluator objects
    """
    from composer import Evaluator
    from llmfoundry.eval.datasets import get_icl_task_dataloader

    os.makedirs(destination_dir, exist_ok=True)

    for icl_cfg in icl_tasks:
        log.info(f"Building evaluator for {icl_cfg['label']}")

        # Set defaults
        if "max_seq_len" not in icl_cfg:
            icl_cfg["max_seq_len"] = icl_seq_len
        if "batch_size" not in icl_cfg:
            icl_cfg["batch_size"] = device_eval_batch_size

        # Set default metrics based on task type
        if "metric_names" not in icl_cfg:
            if icl_cfg["icl_task_type"] == "language_modeling":
                icl_cfg["metric_names"] = ["InContextLearningLMAccuracy"]
            elif (
                icl_cfg["icl_task_type"] == "multiple_choice"
                or icl_cfg["icl_task_type"] == "schema"
            ):
                icl_cfg["metric_names"] = ["InContextLearningMultipleChoiceAccuracy"]
            else:
                icl_cfg["metric_names"] = []

        # Build the dataloader
        icl_cfg = OmegaConf.to_container(icl_cfg)
        label = icl_cfg.pop("label")
        dataset_uri = icl_cfg.pop("dataset_uri")
        icl_task_type = icl_cfg.pop("icl_task_type")
        batch_size = icl_cfg.pop("batch_size")
        metric_names = icl_cfg.pop("metric_names")
        has_categories = icl_cfg.pop("has_categories", False)
        icl_cfg["pad_tok_id"] = tokenizer.pad_token_id
        # Remaining kwargs for the dataset
        kwargs = icl_cfg

        dataloader = get_icl_task_dataloader(
            icl_task_type=icl_task_type,
            dataset_uri=dataset_uri,
            tokenizer=tokenizer,
            batch_size=batch_size,
            has_categories=has_categories,
            destination_path=os.path.join(destination_dir, f"{label}.jsonl"),
            kwargs=kwargs,
        )

        # Create evaluator
        if isinstance(dataloader, dict):
            # Handle categorical datasets
            for category, dl in dataloader.items():
                yield Evaluator(
                    label=f"{label}/{category}",
                    dataloader=dl,
                    metric_names=metric_names,
                )
        else:
            yield Evaluator(
                label=label,
                dataloader=dataloader,
                metric_names=metric_names,
            )


def calculate_markdown_results(
    logger_keys: list[str],
    trainer: Trainer,
    benchmark_to_taxonomy: dict[str, str],
    model_name: str,
) -> pd.DataFrame:
    """Calculate and format evaluation results as a markdown table.

    Args:
        logger_keys: List of metric keys from evaluation
        trainer: Composer trainer with evaluation results
        benchmark_to_taxonomy: Mapping of benchmark names to taxonomy categories
        model_name: Name of the evaluated model

    Returns:
        DataFrame with formatted results
    """
    results = {}

    for key in logger_keys:
        # dl_name is either 2-tuple (benchmark_name, num_fewshot)
        # or 3-tuple (benchmark_name, num_fewshot, subcategory)
        parts = key.split("/")
        dl_name, metric_name = parts[1:-1], parts[-1]
        if "Accuracy" not in metric_name:
            continue

        metric = trainer.state.eval_metrics.get("/".join(dl_name), {}).get(
            metric_name, None
        )

        if metric is None:
            continue
        if dl_name[1] not in results:
            results[dl_name[1]] = {}

        if dl_name[0] not in results[dl_name[1]]:
            results[dl_name[1]][dl_name[0]] = {}

        if metric_name not in results[dl_name[1]][dl_name[0]]:
            results[dl_name[1]][dl_name[0]][metric_name] = []

        results[dl_name[1]][dl_name[0]][metric_name].append(
            {
                "val": metric.compute(),
                "subcat": dl_name[-1] if len(dl_name) == 3 else "no_subcat",
            }
        )

    df = pd.DataFrame(
        columns=[
            "Category",
            "Benchmark",
            "Subtask",
            "Accuracy",
            "Number few shot",
            "Model",
        ],
    )

    for num_shot in results:
        for benchmark in results[num_shot]:
            for metric in results[num_shot][benchmark]:
                subscores = results[num_shot][benchmark][metric]
                if len(subscores) == 1:
                    row = {
                        "Category": benchmark_to_taxonomy.get(benchmark, ""),
                        "Benchmark": benchmark,
                        "Subtask": None,
                        "Accuracy": subscores[0]["val"],
                        "Number few shot": num_shot,
                        "Model": model_name,
                    }
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                else:
                    row = {
                        "Category": benchmark_to_taxonomy.get(benchmark, ""),
                        "Benchmark": benchmark,
                        "Subtask": "Average",
                        "Accuracy": sum(s["val"] for s in subscores) / len(subscores),
                        "Number few shot": num_shot,
                        "Model": model_name,
                    }
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    for sub in subscores:
                        row = {
                            "Category": benchmark_to_taxonomy.get(benchmark, ""),
                            "Benchmark": None,
                            "Subtask": sub["subcat"],
                            "Accuracy": sub["val"],
                            "Number few shot": num_shot,
                            "Model": model_name,
                        }
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def evaluate_model(
    model_name: str,
    model_config: dict[str, Any],
    tokenizer_config: str,
    load_path: str | None = None,
    icl_tasks: list[dict[str, Any]] | None = None,
    eval_gauntlet_config: dict[str, Any] | None = None,
    eval_loader_config: dict[str, Any] | None = None,
    callbacks: list[Callback] | None = None,
    loggers: list[LoggerDestination] | None = None,
    precision: str = "amp_bf16",
    seed: int = 42,
    dist_timeout: float | int = 300.0,
    run_name: str | None = None,
    device: str = "gpu",
    icl_seq_len: int = 1024,
    device_eval_batch_size: int = 8,
    eval_batch_size: int = 1024,
) -> None:
    """Evaluate a single model.

    Args:
        model_name: Name identifier for the model
        model_config: Model configuration dictionary
        tokenizer_config: Tokenizer configuration dictionary
        load_path: Path to checkpoint to load
        icl_tasks: List of ICL task configurations
        eval_gauntlet_config: Configuration for eval gauntlet
        eval_loader_config: Configuration for eval dataloader
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

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    # Build model
    # init_device = model_config.pop("init_device", device)
    model = build_composer_model(model_config, tokenizer, "meta")

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

    # Run evaluation if icl_tasks provided
    if icl_tasks:
        log.info(f"Starting ICL eval for {model_name}...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()

        # Build evaluators as generator and evaluate one at a time
        # Each evaluator will evaluate on the ENTIRE dataset (subset_num_batches=None)
        evaluators_gen = build_icl_evaluators(
            icl_tasks=icl_tasks,
            tokenizer=tokenizer,
            device_eval_batch_size=device_eval_batch_size,
            icl_seq_len=icl_seq_len,
        )

        for evaluator in evaluators_gen:
            log.info(f"Evaluating {evaluator.label} on full dataset...")
            trainer.eval(
                eval_dataloader=evaluator
            )  # No subset_num_batches, uses full dataset
            # Clean up to free memory
            del evaluator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time
        log.info(f"Ran {model_name} ICL eval in: {elapsed_time:.2f} seconds")

    # Run separate evaluation on single batch for perplexity/token accuracy
    if eval_loader_config:
        log.info(
            f"Starting single-batch perplexity/token accuracy eval for {model_name}..."
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()

        # Build the eval dataloader
        from composer import Evaluator
        from llmfoundry.utils.builders import build_dataloader

        eval_dataloader = build_dataloader(
            eval_loader_config,
            tokenizer,
            eval_batch_size,
        )

        # Create evaluator with standard LM metrics (perplexity, token accuracy)
        eval_evaluator = Evaluator(
            label="eval_batch",
            dataloader=eval_dataloader,
            metric_names=[],  # Uses model's default metrics (perplexity, token_accuracy, cross_entropy)
        )

        log.info(f"Evaluating single batch (batch_size={eval_batch_size})...")
        trainer.eval(
            eval_dataloader=eval_evaluator, subset_num_batches=1
        )  # Only 1 batch

        # Cleanup
        del eval_evaluator
        del eval_dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time
        log.info(f"Ran {model_name} single-batch eval in: {elapsed_time:.2f} seconds")

    # Cleanup trainer and model
    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def evaluate(cfg: DictConfig) -> None:
    """Main evaluation function with validated configuration.

    Args:
        cfg: Validated configuration for evaluation

    Returns:
        None
    """
    # Initialize distributed
    dist.initialize_dist(get_device(None), timeout=cfg.get("dist_timeout", 300.0))
    # Set seed
    seed = cfg.get("seed", 42)
    reproducibility.seed_all(seed)

    # Build loggers
    logger_configs = cfg.get("loggers", {})
    loggers = build_loggers(logger_configs)

    # Build callbacks
    callback_configs = cfg.get("callbacks", {})
    callbacks = build_callbacks(callback_configs)

    # Get ICL tasks configuration from eval_config
    icl_tasks = cfg.eval_config.get("icl_tasks", None) if "eval_config" in cfg else None
    eval_loader_config = (
        cfg.eval_config.get("eval_loader", None) if "eval_config" in cfg else None
    )
    eval_gauntlet_config = cfg.get("eval_gauntlet", None)

    models = [cfg.model_config] if "model_config" in cfg else cfg.models
    # Evaluate each model
    for model_cfg in models:
        model_name = model_cfg.get("model_name", "model")
        model_config = OmegaConf.to_container(model_cfg)
        tokenizer_config = cfg.eval_config.tokenizer
        load_path = model_cfg.get("load_path", "")

        evaluate_model(
            model_name=model_name,
            model_config=model_config,
            tokenizer_config=tokenizer_config,
            load_path=load_path,
            icl_tasks=icl_tasks,
            eval_gauntlet_config=eval_gauntlet_config,
            eval_loader_config=eval_loader_config,
            callbacks=callbacks,
            loggers=loggers,
            precision=cfg.eval_config.get("precision", "amp_bf16"),
            seed=seed,
            dist_timeout=cfg.eval_config.get("dist_timeout", 300.0),
            run_name=cfg.eval_config.get("run_name", None),
            device=cfg.eval_config.get("device", "gpu"),
            icl_seq_len=cfg.eval_config.get("icl_seq_len", 1024),
            device_eval_batch_size=cfg.eval_config.get("device_eval_batch_size", 8),
            eval_batch_size=cfg.eval_config.get("eval_batch_size", 1024),
        )

        log.info(f"Completed evaluation for {model_name}")

        # Cleanup between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


@hydra.main(version_base=None, config_path="configs", config_name="eval_example")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation script using Hydra.

    Args:
        cfg: Configuration loaded by Hydra from configs directory

    Returns:
        None

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
    os.environ["S3_ENDPOINT_URL"] = "http://128.232.115.19:9000"

    # Validate configuration using structured schemas
    cfg = build_full_concrete_config(cfg)

    # Run evaluation
    evaluate(cfg)

    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
