from dyna.config import ExecutionMode, NormStructure, RescaleMethod
from dyna.modules.dtanh import DynamicTanh
from dyna.modules.dyna_module import DynaModule
from dyna.modules.layer_module import LayerModule
from dyna.modules.layer_scaled_identity_fn import LayerScaledIdentityFn
from dyna.modules.saturation_gate import SaturationGate

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
    # Enums (re-exported for convenience)
    "NormStructure",
    "RescaleMethod",
    "ExecutionMode",
    # Classes
    "DynaModule",
    "LayerModule",
    "SaturationGate",
    "DynamicTanh",
    "LayerScaledIdentityFn",
]
