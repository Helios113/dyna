from typing import ClassVar

from transformers.modeling_utils import PreTrainedModel

from dyna.config import DynaConfig
from dyna.modules import LayerModule


class DynaPretrainedModel(PreTrainedModel):
    """Base class for Dyna pretrained models."""

    config_class = DynaConfig
    base_model_prefix: str = "Dyna"
    is_parallelizable: bool = False
    main_input_name: str = "input_ids"
    load_tf_weights = None
    _no_split_modules = ClassVar[LayerModule]
