from .dataset_config import DataConfig, DatasetConfig
from .dyna_config import DynaConfig
from .enums import ExecutionMode, NormStructure, RescaleMethod
from .fsdp_config import FSDPConfig
from .model_config import ModelConfig
from .scheduler_config import SchedulerConfig
from .trainer_config import TrainerConfig

CROSS_ENTROPY_IGNORE_INDEX = -100
LATENT_RECURSION_METHODS = [
    ExecutionMode.geiping_std,
    ExecutionMode.geiping_moe,
    ExecutionMode.arbit,
]
GEIPING_METHODS = [
    ExecutionMode.geiping_std,
    ExecutionMode.geiping_moe,
    ExecutionMode.arbit,
]

DEFAULT_CAUSAL_LM_TRAIN_METRICS = [
    "language_cross_entropy",
    "language_perplexity",
    "token_accuracy",
]
PROT_EMB_RESCALING_METHODS = [
    RescaleMethod.cum_avg_prot_emb,
    RescaleMethod.sqrt_prot_emb,
    RescaleMethod.sqrt_scale_prot_emb,
    RescaleMethod.avg_prot_emb,
]

__all__ = [
    # Constants
    "CROSS_ENTROPY_IGNORE_INDEX",
    "LATENT_RECURSION_METHODS",
    "GEIPING_METHODS",
    "DEFAULT_CAUSAL_LM_TRAIN_METRICS",
    "PROT_EMB_RESCALING_METHODS",
    "DataConfig",
    "DatasetConfig",
    "ExecutionMode",
    "NormStructure",
    "RescaleMethod",
    "FSDPConfig",
    "TrainerConfig",
    "SchedulerConfig",
    "DynaConfig",
    "ModelConfig",
]
"""Config submodule for Dyna project.

In here all configuration dataclasses and constants are defined and imported. This does
not include language model configurations as defined by huggingface transformers. Those
are to be defined in the model submodule.
"""
