from enum import Enum


class NormStructure(str, Enum):
    peri = "peri"
    pre = "pre"
    post = "post"
    moeut = "moeut"


class RescaleMethod(str, Enum):
    none = "none"
    complete_p = "complete_p"
    complete_p_dyn = "complete_p_dyn"
    cum_avg_prot_emb = "cum_avg_prot_emb"
    cum_avg_no_prot_emb = "cum_avg_no_prot_emb"
    sqrt_prot_emb = "sqrt_prot_emb"
    sqrt_no_prot_emb = "sqrt_no_prot_emb"
    sqrt_scale_prot_emb = "sqrt_scale_prot_emb"
    avg_prot_emb = "avg_prot_emb"


class ExecutionMode(str, Enum):
    moe = "moe"
    transformer = "transformer"
    geiping_std = "geiping_std"
    geiping_moe = "geiping_moe"
    arbit = "arbit"


class NormType(str, Enum):
    low_precision_layernorm = "low_precision_layernorm"
    low_precision_rmsnorm = "low_precision_rmsnorm"
    dynamic_tanh = "dynamic_tanh"
    unit_norm = "unit_norm"
    rms_norm = "rms_norm"
    ln_norm = "ln_norm"


class LayerType(str, Enum):
    moeut = "moeut"
    simple = "simple"
    direct = "direct"


class TransformerType(str, Enum):
    dyna = "dyna"
    pass_through = "pass_through"


class SweepType(str, Enum):
    continious = "continious"
    category = "category"
