from enum import Enum


class NormStructure(str, Enum):
    peri = "peri"
    pre = "pre"
    post = "post"
    moeut = "moeut"


class RescaleMethod(str, Enum):
    none = "none"
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
    dynamic_thanh = "dynamic_thanh"
