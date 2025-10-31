import hashlib
import os
from typing import cast

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from dyna.config import DynaConfig, ModelConfig
from dyna.model import DynaLM

DEFAULT_HASH = "574a5e521fd9a27bba31cc276c203da9f7e20a6abad882281378579b7d2f389b"


def generate_standard_lm():
    with initialize(version_base=None, config_path="../configs"):
        # config is relative to a module
        cfg = compose(config_name="pytest_transformer")
        OmegaConf.resolve(cfg)
        # Model Config
        model_schema = OmegaConf.structured(ModelConfig)
        model_config: dict[str, object] = cast(
            dict[str, object],
            OmegaConf.to_container(OmegaConf.merge(model_schema, cfg.model_config)),
        )

        conf = DynaConfig(**model_config)
        torch.manual_seed(42)
        model = DynaLM(conf, 0)
        hasher = hashlib.sha256()

        state_dict = model.state_dict()
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            hasher.update(tensor.cpu().numpy().tobytes())
        hash_value = hasher.hexdigest()
        assert hash_value == DEFAULT_HASH, f"Hash mismatch: {hash_value}"

    return model


if __name__ == "__main__" and "PYTEST_VERSION" not in os.environ:
    generate_standard_lm()
