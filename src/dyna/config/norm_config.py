from dataclasses import asdict, dataclass

from .enums import NormType

@dataclass
class NormConfig(dict):
    norm_type: NormType | str = NormType.low_precision_rmsnorm
    attn_eps: float = 0.0
    ffn_eps: float = 0.0

    def __post_init__(self):
        # Convert string to enum if needed (for state dict loading)
        if isinstance(self.norm_type, str):
            object.__setattr__(self, "norm_type", NormType(self.norm_type))

        # Make it behave like a dict for JSON serialization
        dict.__init__(self, asdict(self))

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        # Keep dict in sync
        if hasattr(self, "__dataclass_fields__"):
            self[key] = value
