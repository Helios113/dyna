from dyna.config import ModelConfig
from dyna.modules import LayerModule

from transformers.modeling_utils import PreTrainedModel

class DynaPretrainedModel(PreTrainedModel):
    """Base class for Dyna pretrained models."""

    config_class = ModelConfig  # type: ignore[reportGeneralTypeIssues]
    base_model_prefix: str = "Dyna"
    is_parallelizable: bool = False
    main_input_name: str = "input_ids"
    load_tf_weights = None
    _no_split_modules = [LayerModule]  # type: ignore[reportGeneralTypeIssues]
