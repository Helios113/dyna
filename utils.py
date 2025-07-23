import random
import string
import time
import secrets
import string
from composer.core import Callback
from llmfoundry.utils.builders import build_callback
def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def make_wandb_run_name(config: dict) -> str:
    """
    Generate a unique and informative wandb run name from config.
    Includes a consistent part (run_id), a unique part (timestamp), and key info.
    """
    # Consistent part: use config['run_id'] if present, else generate random string
    run_id = config.get('run_id')
    if not run_id:
        run_id = generate_id(8)
    # Unique part: timestamp
    timestamp = time.strftime("%d%b%y").lower()
    unique = generate_id(8)
    # Important info: select a few key hyperparameters (customize as needed)
    keys = [ 'type','model', 'd_model', 'n_layers', 'n_heads', 'ff_n_experts', 'att_n_experts', 'ff_expert_size', 'dropout']
    info = []
    for k in keys:
        if k in config:
            info.append(f"{k[:4]}={config[k]}")
    info_str = "_".join(info)
    # Compose name
    name = f"{run_id}_{timestamp}_{unique}"
    if info_str:
        name += f"_{info_str}"
    return name


def get_callbacks(cfg: dict) -> list[Callback]:

    return [
        build_callback(
            name=str(name),
            kwargs=callback_cfg,
            train_config=cfg,
        )
        for name, callback_cfg in cfg.items()
    ]