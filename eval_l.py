"""Run Lighteval evaluations plus streaming perplexity for Dyna checkpoints."""

import ast
import json
import logging
import math
import os
from contextlib import nullcontext
from typing import Any, cast

import datasets
import hydra
import torch
from composer.utils import maybe_create_object_store_from_uri, parse_uri
from composer.utils.checkpoint import download_checkpoint, safe_torch_load
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks import lighteval_task as lighteval_task_module
from lighteval.tasks.prompt_manager import PromptManager
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tqdm.auto import tqdm

from dyna.config import DynaConfig
from dyna.model import ComposerDynaModel
from dyna.utils import get_data_loader
from dyna.utils.utils import load_and_concat_yamls
DEFAULT_CONVERTED_DATASET_REVISION = "refs/convert/parquet"
CONVERTED_DATASET_REVISION_ENV = "DYNA_DATASET_CONVERT_REVISION"
DISABLE_CONVERTED_DATASET_FALLBACK_ENV = "DYNA_DISABLE_DATASET_CONVERT_FALLBACK"


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_dataset_max_seq_len(cfg: DictConfig) -> int | None:
    loader_cfg = cfg.eval_config.get("perplexity_loader") or cfg.eval_config.get("eval_loader")
    if loader_cfg is None:
        return None

    dataset_cfg: Any = None
    if isinstance(loader_cfg, DictConfig):
        dataset_cfg = loader_cfg.get("dataset")
    elif isinstance(loader_cfg, dict):
        dataset_cfg = loader_cfg.get("dataset")

    if dataset_cfg is None:
        return None
    if isinstance(dataset_cfg, DictConfig):
        max_len = dataset_cfg.get("max_seq_len")
    elif isinstance(dataset_cfg, dict):
        max_len = dataset_cfg.get("max_seq_len")
    else:
        max_len = None

    return _coerce_int(max_len)


def _resolve_max_seq_len(cfg: DictConfig, model_config: DynaConfig) -> int:
    dataset_len = _resolve_dataset_max_seq_len(cfg)
    if dataset_len is not None:
        return dataset_len

    eval_len = _coerce_int(cfg.eval_config.get("max_seq_len"))
    if eval_len is not None:
        return eval_len

    model_len = _coerce_int(getattr(model_config, "max_seq_len", None))
    if model_len is not None:
        return model_len

    raise ValueError(
        "Unable to determine max_seq_len; set it on the eval dataset config or eval_config.max_seq_len."
    )


def _normalize_task_names(raw_tasks: Any) -> list[str]:
    if isinstance(raw_tasks, ListConfig):
        values = list(raw_tasks)
    elif isinstance(raw_tasks, (list, tuple)):
        values = list(raw_tasks)
    elif raw_tasks:
        values = [raw_tasks]
    else:
        values = ["lambada_openai"]

    return values


def _ensure_s3_endpoint() -> None:
    if "S3_ENDPOINT_URL" not in os.environ:
        os.environ["S3_ENDPOINT_URL"] = DEFAULT_S3_ENDPOINT


def _enable_hf_converted_branch_fallback() -> None:
    """Allow datasets that relied on scripts to fall back to converted branches/configs."""

    if getattr(_enable_hf_converted_branch_fallback, "_patched", False):
        return

    if os.environ.get(DISABLE_CONVERTED_DATASET_FALLBACK_ENV):
        log.info(
            "Skipping HF dataset convert fallback because %s is set.",
            DISABLE_CONVERTED_DATASET_FALLBACK_ENV,
        )
        return

    original_load_dataset = datasets.load_dataset

    def _load_dataset_with_fallbacks(path, *args, **initial_kwargs):  # type: ignore[override]
        attempt_kwargs = initial_kwargs
        tried_converted = False
        tried_builder = False

        while True:
            try:
                return original_load_dataset(path, *args, **attempt_kwargs)
            except Exception as err:  # noqa: BLE001
                fallback_kwargs = None
                if (_needs_converted_revision(err) and not tried_converted) and isinstance(path, str):
                    tried_converted = True
                    fallback_kwargs = attempt_kwargs.copy()
                    fallback_kwargs["revision"] = os.environ.get(
                        CONVERTED_DATASET_REVISION_ENV,
                        DEFAULT_CONVERTED_DATASET_REVISION,
                    )
                    log.warning(
                        "Dataset %s requires a converted branch; retrying with revision '%s'.",
                        path,
                        fallback_kwargs["revision"],
                    )
                elif _needs_builder_config_retry(err, attempt_kwargs) and not tried_builder:
                    tried_builder = True
                    fallback_kwargs = attempt_kwargs.copy()
                    fallback_kwargs["name"] = _pick_available_builder(str(err), fallback_kwargs.get("name"))
                    log.warning(
                        "Falling back to builder config '%s' for dataset %s.",
                        fallback_kwargs["name"],
                        path,
                    )

                if fallback_kwargs is None:
                    raise
                attempt_kwargs = fallback_kwargs

    datasets.load_dataset = _load_dataset_with_fallbacks
    lighteval_task_module.load_dataset = _load_dataset_with_fallbacks
    _enable_hf_converted_branch_fallback._patched = True


def _needs_converted_revision(err: Exception) -> bool:
    return isinstance(err, RuntimeError) and "Dataset scripts are no longer supported" in str(err)


def _needs_builder_config_retry(err: Exception, kwargs: dict[str, Any]) -> bool:
    message = str(err)
    if "BuilderConfig" not in message and "Available configs" not in message:
        return False
    current_name = kwargs.get("name")
    replacement = _pick_available_builder(message, current_name)
    if replacement is None:
        return False
    if current_name is None:
        return True
    return replacement != current_name


def _pick_available_builder(error_msg: str, fallback: str | None = None) -> str | None:
    start = error_msg.find("[")
    end = error_msg.find("]", start)
    if start == -1 or end == -1:
        return fallback
    try:
        options = ast.literal_eval(error_msg[start : end + 1])
    except (ValueError, SyntaxError):
        return fallback
    if not options:
        return fallback
    choice = str(options[0])
    return choice or fallback

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


def _build_perplexity_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    loader_cfg: DictConfig | dict[str, Any] | None,
    eval_batch_size: int,
) -> DataLoader | None:
    if not loader_cfg:
        return None

    if not isinstance(loader_cfg, DictConfig):
        loader_cfg = cast(DictConfig, OmegaConf.create(loader_cfg))

    loader_dict = cast(dict[str, Any], OmegaConf.to_container(loader_cfg, resolve=True))
    dataset_cfg = cast(dict[str, Any], loader_dict.get("dataset", {}))
    streams_path = dataset_cfg.pop("streams_path", None)
    if streams_path:
        dataset_cfg["streams"] = load_and_concat_yamls(streams_path)
    split_override = dataset_cfg.pop("split_override", None)
    if split_override and isinstance(dataset_cfg.get("streams"), dict):
        for stream in dataset_cfg["streams"].values():
            stream["split"] = split_override

    loader_dict["dataset"] = dataset_cfg
    loader_copy = cast(DictConfig, OmegaConf.create(loader_dict))

    data_spec = get_data_loader(loader_copy, tokenizer, eval_batch_size)
    return data_spec.dataloader


def _precision_context(device: torch.device, precision: str):
    precision = (precision or "").lower()
    if precision in {"amp_bf16", "bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif precision in {"amp_fp16", "fp16", "half"}:
        dtype = torch.float16
    else:
        return nullcontext()

    if device.type == "cpu" and dtype == torch.float16:
        log.warning("FP16 autocast is not supported on CPU; falling back to fp32")
        return nullcontext()

    device_type = "cuda" if device.type == "cuda" else "cpu"
    return torch.autocast(device_type=device_type, dtype=dtype)


def _progress_bar(iterable, max_batches: int | None):
    total = None
    try:
        total = len(iterable)
    except (TypeError, AttributeError):
        total = None

    if max_batches is not None:
        total = min(max_batches, total) if total is not None else max_batches

    return tqdm(iterable, total=total, desc="Streaming Perplexity", dynamic_ncols=True)


def _compute_perplexity_metrics(
    base_model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizerBase,
    precision: str,
    microbatch_size: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    base_model.eval()
    device = next(base_model.parameters()).device
    microbatch_size = max(1, int(microbatch_size))

    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_batches = 0

    progress = _progress_bar(dataloader, max_batches)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress):
            if max_batches is not None and batch_idx >= max_batches:
                break

            tensor_batch = {
                key: value
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }

            full_input_ids = tensor_batch["input_ids"]
            if full_input_ids.size(1) < 2:
                continue

            step = min(microbatch_size, full_input_ids.size(0))

            for start in range(0, full_input_ids.size(0), step):
                end = min(start + step, full_input_ids.size(0))
                micro_tensors = {
                    key: value[start:end].to(device, non_blocking=True)
                    for key, value in tensor_batch.items()
                }

                input_ids = micro_tensors["input_ids"]
                if input_ids.size(1) < 2:
                    continue

                attention_mask = micro_tensors.get("attention_mask")
                if attention_mask is None:
                    if tokenizer.pad_token_id is None:
                        raise ValueError(
                            "tokenizer.pad_token_id must be set for perplexity evaluation"
                        )
                    attention_mask = (input_ids != tokenizer.pad_token_id).long()
                labels = micro_tensors.get("labels", input_ids)

                with _precision_context(device, precision):
                    outputs = base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                logits = outputs.logits.float()
                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, 1:]
                shift_mask = attention_mask[:, 1:].to(dtype=torch.bool)

                if shift_logits.numel() == 0:
                    continue

                vocab_size = shift_logits.size(-1)
                losses = torch.nn.functional.cross_entropy(
                    shift_logits.reshape(-1, vocab_size),
                    shift_labels.reshape(-1),
                    reduction="none",
                )

                mask = shift_mask.reshape(-1)
                active_tokens = mask.sum().item()
                if active_tokens == 0:
                    continue

                total_loss += (losses * mask.float()).sum().item()
                total_tokens += active_tokens

                preds = shift_logits.argmax(dim=-1)
                total_correct += ((preds == shift_labels) & shift_mask).sum().item()

                total_batches += 1

            progress.close()

    if total_tokens == 0:
        raise RuntimeError("Evaluation dataloader produced zero valid tokens")

    avg_nll = total_loss / total_tokens
    perplexity = math.exp(avg_nll)
    token_accuracy = total_correct / total_tokens

    return {
        "avg_nll": avg_nll,
        "perplexity": perplexity,
        "token_accuracy": token_accuracy,
        "num_tokens": float(total_tokens),
        "num_batches": float(total_batches),
    }


def run_perplexity_evaluation(
    model: ComposerDynaModel,
    tokenizer: PreTrainedTokenizerBase,
    dataloader: DataLoader | None,
    precision: str,
    microbatch_size: int,
    max_batches: int | None = None,
) -> dict[str, float] | None:
    if dataloader is None:
        return None

    base_model = model.model
    return _compute_perplexity_metrics(
        base_model=base_model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        precision=precision,
        microbatch_size=microbatch_size,
        max_batches=max_batches,
    )


def _resolve_launcher_type(value: str | ParallelismManager | None) -> ParallelismManager:
    """Map user-provided launcher config to a ParallelismManager enum."""
    if isinstance(value, ParallelismManager):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        alias_map = {"DEFAULT": "NONE"}
        normalized = alias_map.get(normalized, normalized)
        if normalized in ParallelismManager.__members__:
            return ParallelismManager[normalized]
    return ParallelismManager.NONE


@hydra.main(version_base=None, config_path="configs", config_name="eval_lighteval")
def main(cfg: DictConfig):
    """Main evaluation function using Lighteval.

    Args:
        cfg: Hydra configuration
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    log.info("Starting Lighteval evaluation")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    _enable_hf_converted_branch_fallback()

    # Set random seed
    seed = cfg.eval_config.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determine device
    device = cfg.eval_config.get("device", "cuda")
    if device == "gpu" and torch.cuda.is_available():
        device = "cuda"
    elif device == "gpu":
        log.warning("GPU requested but not available, falling back to CPU")
        device = "cpu"

    # Build tokenizer
    tokenizer_name = cfg.eval_config.get("tokenizer", "gpt2")
    tokenizer_kwargs = cfg.eval_config.get("tokenizer_kwargs", {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    # Build model config
    model_config = DynaConfig(**cfg.model_config)

    # Create Lighteval-compatible model
    log.info("Initializing ComposerDynaModel")
    model = ComposerDynaModel(
        config=model_config,
        tokenizer=tokenizer,
    )

    # Load checkpoint if specified
    checkpoint_path = cfg.eval_config.get("load_path")
    if checkpoint_path:
        log.info(f"Loading checkpoint from {checkpoint_path}")
        _ensure_s3_endpoint()
        load_object_store = maybe_create_object_store_from_uri(checkpoint_path)
        _, _, parsed_load_path = parse_uri(checkpoint_path)
        
        composer_states_filepath, _, _ = download_checkpoint(
            path=parsed_load_path,
            node_checkpoint_folder="",
            object_store=load_object_store,
            progress_bar=True,
        )
        
        state_dict = safe_torch_load(
            composer_states_filepath=composer_states_filepath,
            load_monolith_rank0_only=True,
        )
        
        model_state = state_dict["state"]["model"]
        model.load_state_dict(model_state, strict=False)
        log.info("Checkpoint loaded successfully")
        
        # Clean up temporary checkpoint file
        if os.path.exists(composer_states_filepath):
            os.remove(composer_states_filepath)
    else:
        log.warning("No checkpoint specified, using randomly initialized model")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Configure eval helpers for the model
    max_seq_len = _resolve_max_seq_len(cfg, model_config)
    add_special_tokens = cfg.eval_config.get("add_special_tokens", True)
    
    model.configure_eval_helpers(
        max_length=max_seq_len,
        add_special_tokens=add_special_tokens,
    )
    log.info(
        f"Configured eval helpers: max_length={max_seq_len}, "
        f"add_special_tokens={add_special_tokens}"
    )

    # Get task names and prepare for evaluation
    task_names = _normalize_task_names(cfg.eval_config.get("tasks"))
    tasks_argument = ",".join(task_names)
    log.info(f"Running evaluation on tasks: {tasks_argument}")

    # Setup evaluation tracker (for logging results)
    output_dir = cfg.eval_config.get("output_dir", "./lighteval_results")
    os.makedirs(output_dir, exist_ok=True)

    tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=cfg.eval_config.get("save_details", True),
        push_to_hub=cfg.eval_config.get("push_to_hub", False),
        public=cfg.eval_config.get("public", False),
    )
    
    # Create pipeline parameters
    custom_tasks_file = cfg.eval_config.get("custom_tasks_file")
    custom_tasks_dir = os.path.dirname(custom_tasks_file) if custom_tasks_file else None
    launcher_setting = cfg.eval_config.get("launcher_type", "none")
    pipeline_params = PipelineParameters(
        launcher_type=_resolve_launcher_type(launcher_setting),
        num_fewshot_seeds=int(cfg.eval_config.get("num_fewshot_seeds", 1)),
        max_samples=cfg.eval_config.get("max_samples"),
        custom_tasks_directory=custom_tasks_dir,
    )


    # Create and run pipeline
    log.info("Creating evaluation pipeline")
    pipeline = Pipeline(
        tasks=tasks_argument,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=tracker,
        model=model,
    )

    log.info("Running evaluation...")
    results = pipeline.evaluate()

    # Log results
    log.info("Evaluation complete!")
    log.info("Results:")
    for task_name, task_results in results.items():
        log.info(f"\n{task_name}:")
        if isinstance(task_results, dict):
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)):
                    log.info(f"  {metric_name}: {metric_value:.4f}")
                else:
                    log.info(f"  {metric_name}: {metric_value}")
        else:
            log.info(f"  {task_results}")

    # Save results
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {results_file}")

    # Optional: Upload to wandb if configured
    if cfg.eval_config.get("wandb", {}).get("enabled", False):
        try:
            import wandb

            wandb.init(
                project=cfg.eval_config.wandb.get("project", "dyna-lighteval"),
                name=cfg.eval_config.get("run_name", "lighteval_run"),
                config=OmegaConf.to_container(cfg, resolve=True),
            )

            # Log results to wandb
            wandb.log({"results": results})
            wandb.finish()

            log.info("Results logged to Weights & Biases")
        except ImportError:
            log.warning("wandb not installed, skipping wandb logging")

    return results


if __name__ == "__main__":
    main()
