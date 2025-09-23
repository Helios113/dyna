from dyna.config.enums import NormStructure, RescaleMethod, ExecutionMode

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
