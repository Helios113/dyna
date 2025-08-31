import random
import torch
from transformers import AutoTokenizer
from composer.loggers import WandBLogger
from dyna.model.model import ComposerDynaModel, DynaConfig
from composer import Trainer
import hydra
from omegaconf import DictConfig, OmegaConf
from composer.optim import DecoupledAdamW
from composer.profiler import JSONTraceHandler, cyclic_schedule
from composer.profiler.profiler import Profiler
from tools.old_model import MoEUTLM_old
import tools.profiles
from dyna.utils.utils import (
    make_wandb_run_name,
    get_callbacks,
    get_data_loader,
    build_full_concrete_config,
    get_scheduler,
)
from beartype import beartype
from torch.nn import Module


@hydra.main(
    version_base=None, config_path="configuration", config_name="MoA_moeut_160M"
)
def main(cfg: DictConfig):
    cfg = build_full_concrete_config(cfg)
    vocab_size = 50368
    run_name = make_wandb_run_name(cfg.model_config, cfg.trainer_config)
    cfg.trainer_config.save_filename = run_name + "-ba{batch}.pt"
    wandb_logger = WandBLogger(project="dyna", log_artifacts=False, name=run_name)

    # Set all random seeds for reproducibility
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    # We don't need a tokenizer becuase all of our data is pre-tokenized.
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    # Instead of passing the DictConfig directly, unpack it as kwargs
    conf = DynaConfig(**cfg.model_config)

    model = ComposerDynaModel(config=conf, tokenizer=tokenizer).cuda()

    # Reset seeds again before creating old model to ensure same initialization
    random.seed(seed)
    torch.manual_seed(seed)

    # old_model = MoEUTLM_old(vocab_size, **tools.profiles.MoEUT_Test).cuda()
    input = torch.randint(0, vocab_size, (1, 1024)).cuda()
    out_1 = model({"input_ids": input})
    # out_2 = old_model(input)
    print(out_1["logits"].shape)
    # print(out_2.outputs.shape)

    print("Outputs from new and old model should be very close:")
    # print(torch.norm(out_1["logits"].flatten() - out_2.outputs.flatten()).item())


if __name__ == "__main__":
    main()
