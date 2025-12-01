"""Evaluation script for Dyna models using Lighteval from Hugging Face.

This script provides functionality to load and evaluate Dyna models using the
lighteval library from Hugging Face, which offers a lightweight and flexible
evaluation framework.

Usage with Hydra:
    python eval_l.py                                    # Uses default config
    python eval_l.py --config-name=eval_lighteval       # Uses specific config
    python eval_l.py device=cpu precision=fp32          # Override parameters
    
Lighteval supports many standard benchmarks including:
    - MMLU (Massive Multitask Language Understanding)
    - HellaSwag
    - ARC (AI2 Reasoning Challenge)
    - TruthfulQA
    - GSM8K
    - PIQA
    - WinoGrande
    and many more...
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import hydra
import torch
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.base_model import BaseModel
from lighteval.models.model_config import BaseModelConfig
from lighteval.models.model_output import (
    GenerativeTaskOutput,
    LoglikelihoodOutput,
    LoglikelihoodSingleTokenOutput,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.tasks.registry import Registry, taskinfo_selector
from omegaconf import DictConfig, OmegaConf
from torch.nn.functional import log_softmax
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from dyna.config import DynaConfig
from dyna.model import ComposerDynaModel

log = logging.getLogger(__name__)


class DynaLightevalModel(BaseModel):
    """Wrapper to make Dyna models compatible with Lighteval framework.
    
    This class adapts the Dyna model to work with Lighteval's evaluation pipeline
    by implementing the required interface methods.
    """

    def __init__(
        self,
        config: DictConfig,
        model_config: DynaConfig,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
        batch_size: int = 1,
    ):
        """Initialize the Lighteval-compatible Dyna model.
        
        Args:
            config: Hydra configuration
            model_config: Dyna model configuration
            tokenizer: Tokenizer for the model
            device: Device to run inference on
            batch_size: Batch size for evaluation
        """
        self.config = config
        self.model_config = model_config
        self._tokenizer = tokenizer
        self._device = device
        self._batch_size = batch_size

        # Initialize the Dyna model
        log.info("Initializing Dyna model for Lighteval...")
        self.model = ComposerDynaModel(config=model_config, tokenizer=tokenizer)
        
        # Load checkpoint if specified
        if config.eval_config.get("load_path"):
            self._load_checkpoint(config.eval_config.load_path)
        
        # Move model to device and set to eval mode
        self.model = self.model.to(device)
        self.model.eval()
        
        # Set precision
        precision = config.eval_config.get("precision", "fp32")
        if precision == "amp_bf16":
            self.model = self.model.to(torch.bfloat16)
        elif precision == "fp16":
            self.model = self.model.to(torch.float16)
        
        log.info(f"Model loaded on {device} with precision {precision}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        log.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "state" in checkpoint:
            # Composer checkpoint format
            if "model" in checkpoint["state"]:
                state_dict = checkpoint["state"]["model"]
            else:
                state_dict = checkpoint["state"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        # The ComposerDynaModel wraps a DynaLM model
        # We need to load into self.model.model (the DynaLM instance)
        # Remove 'model.' prefix if present in keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                new_key = key[6:]  # Remove "model." prefix
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        self.model.model.load_state_dict(new_state_dict, strict=False)
        log.info("Checkpoint loaded successfully")

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Return the tokenizer."""
        return self._tokenizer

    @property
    def max_length(self) -> int:
        """Return maximum sequence length."""
        return self.model_config.max_seq_len

    def greedy_until(
        self,
        requests: list[tuple[str, dict]],
        override_bs: Optional[int] = None,
    ) -> list[GenerativeTaskOutput]:
        """Generate text using greedy decoding until stop sequences.
        
        Args:
            requests: List of (context, request_args) tuples
            override_bs: Optional batch size override
            
        Returns:
            List of generated outputs
        """
        results = []
        batch_size = override_bs if override_bs is not None else self._batch_size
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]
            contexts = [req[0] for req in batch]
            request_args = [req[1] for req in batch]
            
            # Tokenize contexts
            inputs = self._tokenizer(
                contexts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self._device)
            
            # Get generation settings
            max_new_tokens = request_args[0].get("max_new_tokens", 32)
            stop_sequences = request_args[0].get("stop_sequences", [])
            
            # Generate using autoregressive decoding
            # Note: Dyna model doesn't have built-in generate, so we implement greedy decoding
            with torch.no_grad():
                input_ids = inputs.input_ids
                
                for _ in range(max_new_tokens):
                    # Forward pass
                    outputs = self.model.model(input_ids=input_ids)
                    
                    # Get next token (greedy)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
                    
                    # Append to sequence
                    input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                    
                    # Check for EOS
                    if (next_tokens == self._tokenizer.eos_token_id).all():
                        break
                    
                    # Check max length
                    if input_ids.shape[1] >= self.max_length:
                        break
                
                outputs = input_ids
            
            # Decode outputs
            for j, output in enumerate(outputs):
                # Remove input tokens
                generated_tokens = output[inputs.input_ids[j].shape[0]:]
                generated_text = self._tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                
                # Apply stop sequences
                for stop_seq in stop_sequences:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                
                results.append(
                    GenerativeTaskOutput(
                        result=generated_text,
                        logits=None,
                        generated_tokens=generated_tokens.tolist(),
                        input_tokens=inputs.input_ids[j].tolist(),
                    )
                )
        
        return results

    def loglikelihood(
        self,
        requests: list[tuple[str, str]],
        override_bs: Optional[int] = None,
    ) -> list[LoglikelihoodOutput]:
        """Compute log-likelihood for (context, continuation) pairs.
        
        Args:
            requests: List of (context, continuation) tuples
            override_bs: Optional batch size override
            
        Returns:
            List of log-likelihood outputs
        """
        results = []
        batch_size = override_bs if override_bs is not None else self._batch_size
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]
            
            # Process each request in the batch
            batch_results = []
            for context, continuation in batch:
                # Tokenize context and continuation separately
                full_text = context + continuation
                
                context_tokens = self._tokenizer.encode(
                    context, add_special_tokens=False
                )
                full_tokens = self._tokenizer.encode(
                    full_text, add_special_tokens=False
                )
                continuation_tokens = full_tokens[len(context_tokens):]
                
                # Prepare input
                input_ids = torch.tensor([full_tokens], device=self._device)
                
                # Get logits
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits
                
                # Compute log probabilities
                log_probs = log_softmax(logits, dim=-1)
                
                # Get log-likelihood for continuation tokens
                cont_start_idx = len(context_tokens) - 1
                cont_end_idx = cont_start_idx + len(continuation_tokens)
                
                # Sum log probabilities for continuation tokens
                loglikelihood = 0.0
                for idx, token_id in enumerate(continuation_tokens):
                    pos = cont_start_idx + idx
                    if pos < log_probs.shape[1]:
                        loglikelihood += log_probs[0, pos, token_id].item()
                
                # Check if the continuation is greedy (most likely)
                is_greedy = True
                for idx, token_id in enumerate(continuation_tokens):
                    pos = cont_start_idx + idx
                    if pos < logits.shape[1]:
                        predicted_token = logits[0, pos].argmax().item()
                        if predicted_token != token_id:
                            is_greedy = False
                            break
                
                batch_results.append(
                    LoglikelihoodOutput(
                        result=(loglikelihood, is_greedy),
                        input_tokens=context_tokens,
                        generated_tokens=continuation_tokens,
                        truncated_tokens_count=0,
                    )
                )
            
            results.extend(batch_results)
        
        return results

    def loglikelihood_single_token(
        self,
        requests: list[tuple[str, str]],
        override_bs: Optional[int] = None,
    ) -> list[LoglikelihoodSingleTokenOutput]:
        """Compute log-likelihood for single token continuations.
        
        Args:
            requests: List of (context, continuation) tuples where continuation is a single token
            override_bs: Optional batch size override
            
        Returns:
            List of single-token log-likelihood outputs
        """
        results = []
        batch_size = override_bs if override_bs is not None else self._batch_size
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]
            contexts = [req[0] for req in batch]
            continuations = [req[1] for req in batch]
            
            # Tokenize contexts
            inputs = self._tokenizer(
                contexts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self._device)
            
            # Get logits
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get log probabilities for each continuation
            for j, continuation in enumerate(continuations):
                # Get token for continuation
                cont_token_id = self._tokenizer.encode(
                    continuation, add_special_tokens=False
                )[0]
                
                # Get log probability at last position
                last_token_logits = logits[j, -1, :]
                log_probs = log_softmax(last_token_logits, dim=-1)
                loglikelihood = log_probs[cont_token_id].item()
                
                # Check if greedy
                predicted_token = last_token_logits.argmax().item()
                is_greedy = predicted_token == cont_token_id
                
                results.append(
                    LoglikelihoodSingleTokenOutput(
                        result=(loglikelihood, is_greedy),
                        input_tokens=inputs.input_ids[j].tolist(),
                        generated_tokens=[cont_token_id],
                        truncated_tokens_count=0,
                    )
                )
        
        return results


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


def setup_lighteval_tasks(
    task_names: list[str],
    custom_tasks: Optional[str] = None,
) -> tuple[Registry, list[LightevalTask]]:
    """Setup Lighteval tasks from task names.
    
    Args:
        task_names: List of task names (e.g., "mmlu", "hellaswag:5")
        custom_tasks: Optional path to custom tasks file
        
    Returns:
        Tuple of (registry, list of tasks)
    """
    log.info(f"Setting up Lighteval tasks: {task_names}")
    
    # Create registry
    registry = Registry()
    
    # Load custom tasks if specified
    if custom_tasks and os.path.exists(custom_tasks):
        log.info(f"Loading custom tasks from {custom_tasks}")
        # Custom task loading would go here
        # registry.register_custom_tasks(custom_tasks)
    
    # Select tasks from registry
    tasks = taskinfo_selector(task_names)
    
    log.info(f"Selected {len(tasks)} tasks for evaluation")
    return registry, tasks


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
    batch_size = cfg.eval_config.get("batch_size", 1)
    model = DynaLightevalModel(
        config=cfg,
        model_config=model_config,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )
    
    # Setup tasks
    task_names = cfg.eval_config.get("tasks", ["lambada_openai"])
    custom_tasks = cfg.eval_config.get("custom_tasks_file", None)
    registry, tasks = setup_lighteval_tasks(task_names, custom_tasks)
    
    # Setup evaluation tracker (for logging results)
    output_dir = cfg.eval_config.get("output_dir", "./lighteval_results")
    os.makedirs(output_dir, exist_ok=True)
    
    tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=cfg.eval_config.get("save_details", True),
        push_to_hub=cfg.eval_config.get("push_to_hub", False),
        public=cfg.eval_config.get("public", False),
    )
    
    # Setup parallelism manager
    parallelism_config = cfg.eval_config.get("parallelism", {})
    parallelism = ParallelismManager(
        dp_size=parallelism_config.get("dp_size", 1),
        pp_size=parallelism_config.get("pp_size", 1),
        tp_size=parallelism_config.get("tp_size", 1),
    )
    
    # Create pipeline parameters
    pipeline_params = PipelineParameters(
        launcher_type=cfg.eval_config.get("launcher_type", "default"),
        override_batch_size=cfg.eval_config.get("override_batch_size", None),
        max_samples=cfg.eval_config.get("max_samples", None),
        num_fewshot_seeds=cfg.eval_config.get("num_fewshot_seeds", 1),
    )
    
    # Create and run pipeline
    log.info("Creating evaluation pipeline")
    pipeline = Pipeline(
        tasks=task_names,
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
        for metric_name, metric_value in task_results.items():
            log.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Save results
    results_file = os.path.join(output_dir, "results.json")
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
