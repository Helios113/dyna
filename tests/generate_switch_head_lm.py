from typing import cast

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from dyna.config import DynaConfig, ModelConfig
from dyna.model import DynaLM


def _generate_lm_with_overrides(overrides: dict | None = None):
    """Separate LM configurations for multiple and single expert SwitchHead tests."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="pytest_switch_head")
        if overrides:
            cfg.model_config = OmegaConf.merge(cfg.model_config, overrides)

        OmegaConf.resolve(cfg)
        model_schema = OmegaConf.structured(ModelConfig)
        model_config: dict[str, object] = cast(
            dict[str, object],
            OmegaConf.to_container(OmegaConf.merge(model_schema, cfg.model_config)),
        )

        conf = DynaConfig(**model_config)
        torch.manual_seed(42)
        model = DynaLM(conf, 0)

    return model


def generate_switch_head_multiple_experts_lm():
    return _generate_lm_with_overrides()


def generate_switch_head_lm_single_expert():
    overrides = {
        "n_experts_attn": 1,
        "k_attn": 1,
    }
    return _generate_lm_with_overrides(overrides=overrides)
