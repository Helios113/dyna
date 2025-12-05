"""Model evaluation script based on Composer framework.

This script evaluates Dyna models using Composer's Trainer with support for:
- Multiple model configurations
- ICL tasks evaluation
- Eval gauntlet benchmarks
- Flexible logging (WandB, file-based, etc.)
- Distributed evaluation support
"""

import gc
import logging
import os
import time
from typing import Any, Optional, Union
from dyna.config import DataConfig
from dyna.utils.utils import load_and_concat_yamls
import hydra
import pandas as pd
import torch
from composer.core import Callback
from composer.loggers.logger_destination import LoggerDestination
from composer.trainer import Trainer
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer
from dyna.utils.builders import (
    build_evaluators
)
from dyna.config import DynaConfig
from dyna.model import ComposerDynaModel
from dyna.utils.builders import (
    build_callback
)
log = logging.getLogger(__name__)


def build_tokenizer(tokenizer_name: str, tokenizer_kwargs: dict[str, Any]) -> Any:
    """Build a tokenizer from HuggingFace.

    Args:
        tokenizer_name: Name or path of the tokenizer
        tokenizer_kwargs: Additional kwargs for tokenizer

    Returns:
        The loaded tokenizer
    """
    log.info(f"Building tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    # Set pad token to eos token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_composer_model(
    model_cfg: dict[str, Any],
    tokenizer: Any,
) -> ComposerDynaModel:
    """Build a Composer Dyna model.

    Args:
        model_cfg: Model configuration dict
        tokenizer: Tokenizer to use

    Returns:
        ComposerDynaModel instance
    """
    dyna_config = DynaConfig(**model_cfg)
    model = ComposerDynaModel(
        config=dyna_config,
        tokenizer=tokenizer,
    )
    return model


def evaluate_model(
    tokenizer: dict[str, Any],
    model_name: str,
    model: dict[str, Any],
    dist_timeout: Union[float, int],
    run_name: str,
    seed: int,
    icl_tasks: Union[str, list[dict[str, Any]]],
    max_seq_len: int,
    device_eval_batch_size: Union[int, float],
    eval_gauntlet_config: Optional[Union[str, dict[str, Any]]],
    eval_loader_config: Optional[Union[dict[str, Any], list[dict[str, Any]]]],
    loggers: list[LoggerDestination],
    python_log_level: Optional[str],
    precision: str,
    eval_gauntlet_df: Optional[pd.DataFrame],
    eval_subset_num_batches: int,
    icl_subset_num_batches: Optional[int],
    callback_configs: Optional[dict[str, Any]],
    metadata: Optional[dict[str, str]],
    logged_config: dict[str, Any],
    parallelism_config: Optional[dict[str, Any]] = None,
    should_log_config: bool = True,
    load_path: Optional[str] = None,
):
    """Evaluate a single model configuration.

    Args:
        tokenizer: Tokenizer configuration dict
        model_name: Name of the model
        model: Model configuration dict
        dist_timeout: Distributed timeout
        run_name: Name of the run
        seed: Random seed
        icl_tasks: ICL tasks to evaluate
        max_seq_len: Maximum sequence length
        device_eval_batch_size: Batch size for evaluation
        eval_gauntlet_config: Eval gauntlet configuration
        eval_loader_config: Eval loader configuration
        loggers: List of logger destinations
        python_log_level: Python log level
        precision: Training precision
        eval_gauntlet_df: Eval gauntlet dataframe
        eval_subset_num_batches: Number of batches for eval subset
        icl_subset_num_batches: Number of batches for ICL subset
        callback_configs: Callback configurations
        metadata: Metadata dictionary
        logged_config: Logged configuration
        parallelism_config: Parallelism configuration
        should_log_config: Whether to log config
        load_path: Path to load checkpoint from

    Returns:
        Tuple of (trainer, logger_keys, eval_gauntlet_callback, eval_gauntlet_df)
    """
    log.info(f'Evaluating model: {model_name}')
    
    # Build tokenizer
    tokenizer_cfg = tokenizer
    tokenizer_name = tokenizer_cfg['name']
    tokenizer_kwargs = tokenizer_cfg.get('kwargs', {})
    built_tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    # Build evaluators icl
    evaluators_icl, logger_keys_icl, eval_gauntlet_callback = build_evaluators(
        None,
        icl_tasks,
        eval_gauntlet_config,
        tokenizer=built_tokenizer,
        device_eval_batch_size=device_eval_batch_size,
        icl_seq_len=max_seq_len,
        icl_subset_num_batches=icl_subset_num_batches,
    )
    
    evaluators_perp, logger_keys_perp, _ = build_evaluators(
        eval_loader_config,
        None,
        None,
        tokenizer=built_tokenizer,
        device_eval_batch_size=device_eval_batch_size,
        icl_seq_len=max_seq_len,
        icl_subset_num_batches=icl_subset_num_batches,
    )

    # Build callbacks
    callbacks: list[Callback] = [
        build_callback(name=str(name), kwargs=callback_cfg)
        for name, callback_cfg in callback_configs.items()
    ] if callback_configs else []

    if eval_gauntlet_callback is not None:
        callbacks.append(eval_gauntlet_callback)

    # Build composer model
    composer_model = build_composer_model(model, built_tokenizer)

    # # # Add eval metrics to evaluators
    if eval_loader_config is not None:
        from dyna.utils.builders import add_metrics_to_eval_loaders
        train_metrics = composer_model.get_metrics(is_train=True)
        evaluators_perp = add_metrics_to_eval_loaders(
            evaluators_perp,
            list(train_metrics.keys()),
        )

    # Initialize eval gauntlet df if needed
    if eval_gauntlet_df is None and eval_gauntlet_callback is not None:
        eval_gauntlet_df = pd.DataFrame(
            columns=['model_name'] + list(eval_gauntlet_callback.averages) +
            [t['name'] for t in eval_gauntlet_callback.categories],
        )

    # Get FSDP config
    fsdp_config = parallelism_config.get('fsdp', None) if parallelism_config else None

    log.info(f'Building trainer for {model_name}...')
    trainer = Trainer(
        run_name=run_name,
        seed=seed,
        model=composer_model,
        callbacks=callbacks,
        loggers=loggers,
        precision=precision,
        parallelism_config={"fsdp_config": fsdp_config},
        load_path=load_path,
        load_weights_only=True,
        load_strict_model_weights=True,
        progress_bar=True,
        log_to_console=True,
        dist_timeout=dist_timeout,
        python_log_level=python_log_level,
    )

    if should_log_config:
        log.info('Evaluation config:')
        # Log config to trainer loggers
        for logger in trainer.logger.destinations:
            if hasattr(logger, 'log_hyperparameters'):
                logger.log_hyperparameters(logged_config)

    log.info(f'Starting eval for {model_name}...')
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    a = time.time()
    trainer.eval(
        eval_dataloader=evaluators_icl,
        subset_num_batches=icl_subset_num_batches,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    b = time.time()

    log.info(f'Ran {model_name} eval in: {b-a} seconds')

    model_results = calculate_markdown_results(
        logger_keys_icl,
        trainer,
        {},
        model_name,
    )

    trainer.eval(
        eval_dataloader=evaluators_perp,
        subset_num_batches=eval_subset_num_batches,
    )
    model_results = calculate_markdown_results(
        logger_keys_perp,
        trainer,
        {},
        "test1",
    )
    
    # Clean up trainer and free memory
    trainer.close()
    del trainer
    # del logger_keys
    del eval_gauntlet_callback
    
    # Force garbage collection and clear CUDA cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # log.info(f'Cleaned up resources for {model_cfg_dict["model_name"]}')


def calculate_markdown_results(
    logger_keys: list[str],
    trainer: Trainer,
    benchmark_to_taxonomy: dict[str, str],
    model_name: str,
) -> pd.DataFrame:
    """Calculate results in markdown-friendly format.

    Args:
        logger_keys: List of logger keys from evaluation
        trainer: Composer trainer
        benchmark_to_taxonomy: Mapping of benchmarks to taxonomy
        model_name: Name of the model

    Returns:
        DataFrame with results
    """
    results = {}
    for key in logger_keys:
        # dl_name is either 2-tuple (benchmark_name, num_fewshot)
        # or 3-tuple (benchmark_name, num_fewshot, subcategory)
        dl_name, metric_name = key.split('/')[1:-1], key.split('/')[-1]
        # if 'Accuracy' not in metric_name:
        #     continue
        metric = trainer.state.eval_metrics.get('/'.join(dl_name), {}).get(metric_name, None)
        if metric is None:
            continue
        if dl_name[1] not in results:
            results[dl_name[1]] = {}

        if dl_name[0] not in results[dl_name[1]]:
            results[dl_name[1]][dl_name[0]] = {}

        if metric_name not in results[dl_name[1]][dl_name[0]]:
            results[dl_name[1]][dl_name[0]][metric_name] = []

        results[dl_name[1]][dl_name[0]][metric_name].append({
            'val': metric.compute(),
            'subcat': dl_name[-1] if len(dl_name) == 3 else 'no_subcat',
        })

    df = pd.DataFrame(
        columns=[
            'Category',
            'Benchmark',
            'Subtask',
            'Accuracy',
            'Number few shot',
            'Model',
        ],
    )

    for num_shot in results:
        for benchmark in results[num_shot]:
            for metric in results[num_shot][benchmark]:
                subscores = results[num_shot][benchmark][metric]
                if len(subscores) == 1:
                    row = {
                        'Category': benchmark_to_taxonomy.get(benchmark, ''),
                        'Benchmark': benchmark,
                        'Subtask': None,
                        'Accuracy': subscores[0]['val'],
                        'Number few shot': num_shot,
                        'Model': model_name,
                    }
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                else:
                    row = {
                        'Category': benchmark_to_taxonomy.get(benchmark, ''),
                        'Benchmark': benchmark,
                        'Subtask': 'Average',
                        'Accuracy': sum(s['val'] for s in subscores) / len(subscores),
                        'Number few shot': num_shot,
                        'Model': model_name,
                    }
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    for sub in subscores:
                        row = {
                            'Category': benchmark_to_taxonomy.get(benchmark, ''),
                            'Benchmark': None,
                            'Subtask': sub['subcat'],
                            'Accuracy': sub['val'],
                            'Number few shot': num_shot,
                            'Model': model_name,
                        }
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


def allow_toplevel_keys(cfg: dict[str, Any]) -> dict[str, Any]:
    """Transform the config to allow top-level keys for model configuration.

    This function allows users to use the 'train.py' syntax in 'eval.py'.
    It converts a config with top-level 'model', 'tokenizer', and (optionally) 'load_path' keys
    into the nested 'models' list format required by 'eval.py'.
    """
    if 'model' in cfg:
        if 'models' in cfg:
            raise ValueError(
                'Please specify either model or models in the config, not both',
            )
        default_name = cfg.get('model').get('name', 'model')
        model_cfg = {
            'model': cfg.pop('model'),
            'tokenizer': cfg.pop('tokenizer', None),
            'model_name': cfg.pop('model_name', default_name),
        }
        if 'tokenizer' not in model_cfg or model_cfg['tokenizer'] is None:
            raise ValueError(
                'When specifying model, "tokenizer" must be provided in the config',
            )
        if 'load_path' in cfg:
            model_cfg['load_path'] = cfg.pop('load_path')
        cfg['models'] = [model_cfg]

    return cfg


def evaluate(cfg: DictConfig) -> tuple[pd.DataFrame]:
    """Main evaluation function.

    Args:
        cfg: Hydra configuration

    Returns:
        Tuple of (empty list, results DataFrame)
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    log.info("Starting model evaluation")
    
    # Transform config to allow top-level keys
    cfg_dict = om.to_container(cfg, resolve=True)
    cfg_dict = allow_toplevel_keys(cfg_dict)
    cfg = om.create(cfg_dict)
    
    # Get evaluation config
    eval_config = cfg.get("eval_config", {})
    model_configs = cfg.get("models", [])
    
    if not model_configs:
        raise ValueError("No models specified in configuration")
    
    # Mandatory Evaluation Parameters
    icl_tasks = eval_config.get("icl_tasks") or eval_config.get("icl_tasks_str", [])
    if icl_tasks is None:
        icl_tasks = []
    else:
        icl_tasks = [om.to_container(i, resolve=True) for i in icl_tasks]
        
    
    # Optional Evaluation Parameters with default values
    eval_loader_config = eval_config.get("eval_loader") or eval_config.get("eval_loaders")
    if eval_loader_config is not None:
        eval_loader_config = om.to_container(eval_loader_config, resolve=True)
    eval_gauntlet_config = eval_config.get("eval_gauntlet") or eval_config.get("eval_gauntlet_str")
    
    default_run_name: str = os.environ.get('RUN_NAME', 'dyna_eval')
    run_name = eval_config.get("run_name", default_run_name)
    
    # Set random seed
    seed = eval_config.get("seed", 42)
    reproducibility.seed_all(seed)
    
    # Initialize distributed
    dist_timeout = eval_config.get("dist_timeout", 600)
    device = get_device(None)
    dist.initialize_dist(device, timeout=dist_timeout)
    
    # Setup python logging level
    python_log_level = eval_config.get("python_log_level")
    if python_log_level is not None:
        logging.basicConfig(
            format=f'%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s',
            force=True,
        )
        logging.getLogger('dyna').setLevel(python_log_level.upper())
    
    # Default argument values for evaluate_model
    eval_gauntlet_df = None
    models_df = None
    composite_scores = None
    
    # Build loggers
    loggers: list[LoggerDestination] = []
    
    # Get other eval config parameters
    max_seq_len = eval_config.get("max_seq_len", 1024)
    device_eval_batch_size = eval_config.get("device_eval_batch_size", 8)
    precision = eval_config.get("precision", "amp_bf16")
    eval_subset_num_batches = eval_config.get("eval_subset_num_batches", -1)
    icl_subset_num_batches = eval_config.get("icl_subset_num_batches", -1)
    callback_configs = eval_config.get("callbacks", {})
    metadata = eval_config.get("metadata")
    fsdp_config = eval_config.get("fsdp_config")
    should_log_config = eval_config.get("log_config", True)
    
    # Evaluate each model
    for model_cfg in model_configs:
        model_cfg_dict = om.to_container(model_cfg, resolve=True)
        
        evaluate_model(
            dist_timeout=dist_timeout,
            run_name=run_name,
            seed=seed,
            icl_tasks=icl_tasks,
            max_seq_len=max_seq_len,
            device_eval_batch_size=device_eval_batch_size,
            eval_gauntlet_config=eval_gauntlet_config,
            eval_loader_config=eval_loader_config,
            loggers=loggers,
            python_log_level=python_log_level,
            parallelism_config={'fsdp': fsdp_config},
            precision=precision,
            eval_gauntlet_df=eval_gauntlet_df,
            callback_configs=callback_configs,
            eval_subset_num_batches=eval_subset_num_batches,
            icl_subset_num_batches=icl_subset_num_batches,
            metadata=metadata,
            logged_config=cfg_dict,
            should_log_config=should_log_config,
            **model_cfg_dict,
        )

DEFAULT_S3_ENDPOINT = "http://128.232.115.19:9000"

def _ensure_s3_endpoint() -> None:
    if "S3_ENDPOINT_URL" not in os.environ:
        os.environ["S3_ENDPOINT_URL"] = DEFAULT_S3_ENDPOINT

@hydra.main(version_base=None, config_path="configs", config_name="evaluate_single")
def main(cfg: DictConfig):
    """Main entry point for evaluation script.

    Args:
        cfg: Hydra configuration
    """
    
    data_schema = om.structured(DataConfig)
    data_config = om.merge(data_schema, cfg.eval_config.eval_loader)
    streams_configs = load_and_concat_yamls(cfg.eval_config.eval_loader.path)

    del data_config.path  # pyright: ignore[reportAttributeAccessIssue]

    data_config.dataset.streams = streams_configs
    
    cfg.eval_config.eval_loader = data_config
    _ensure_s3_endpoint()
    eval_gauntlet_df = evaluate(cfg)
    return eval_gauntlet_df


if __name__ == "__main__":
    main()
