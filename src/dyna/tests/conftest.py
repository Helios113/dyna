import torch
import pytest
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from dyna.config import DynaConfig
from dyna.model.model import DynaLM


@pytest.fixture
def standard_config():
    """
    Load standard configuration.
    """
    config_path = "configs/Transformer.yaml"
    cfg = OmegaConf.load(config_path)
    model_config = DynaConfig(**cfg.model_config)
    return model_config

@pytest.fixture
def test_model(standard_config):
    """
    Create test model instance using standard configuration.
    """
    model = DynaLM(standard_config, eos_token_id=0)
    return model

@pytest.fixture
def device():
    """
    Get device to use for testing.
    """
    # return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")

#----------------------------------------
# Regression testing functions
#----------------------------------------

def save_test_output(output, filename):
    """
    Save test output.
    """
    import os
    os.makedirs("test_outputs", exist_ok=True)
    torch.save(output, f"test_outputs/{filename}")
    print(f"Saved test output to test_outputs/{filename}")
    
def load_test_output(filename):
    """
    Load test output.
    """
    return torch.load(f"test_outputs/{filename}")
