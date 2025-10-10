from dyna.config.dataset_config import DataConfig, DatasetConfig
from dyna.config.enums import ExecutionMode, NormStructure, RescaleMethod
from dyna.config.fsdp_config import FSDPConfig
from dyna.config.model_config import ModelConfig
from dyna.config.scheduler_config import SchedulerConfig
from dyna.config.trainer_config import TrainerConfig

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
    "ModelConfig",
    "FSDPConfig",
    "TrainerConfig",
    "SchedulerConfig",
]
