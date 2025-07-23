from transformers import AutoTokenizer
from model_config import ModelConfig, TrainerConfig
from composer.loggers import WandBLogger
from model import ComposerMoEUT, MoEUTConfig
from composer import Trainer
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from composer.optim import DecoupledAdamW
from data.stream_text_data import StreamingTextDataset
from utils import make_wandb_run_name, get_callbacks
@hydra.main(version_base=None, config_path="configuration", config_name="MoA")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # max_seq_len -- VERY Important
    
    
    # CONFIGS
    # Model Config
    model_schema = OmegaConf.structured(ModelConfig)
    model_config = cfg.model_config
    model_config = OmegaConf.merge(model_schema, model_config)
    assert isinstance(model_config, DictConfig), "model_config should be a DictConfig instance"
    
    # Trainer Config
    trainer_schema = OmegaConf.structured(TrainerConfig)
    trainer_config = cfg.trainer_config
    trainer_config = OmegaConf.merge(trainer_schema, trainer_config)
    assert isinstance(trainer_config, DictConfig), "trainer_config should be a DictConfig instance"
    
    
    run_name = make_wandb_run_name(model_config)

    wandb_logger = WandBLogger(project="dyna", log_artifacts=False, name=run_name)
    # We don't need a tokenizer becuase all of our data is pre-tokenized.
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token
    
    conf = MoEUTConfig(model_config)
    
    model = ComposerMoEUT(
        config=conf,
        tokenizer=tokenizer,
        additional_train_metrics=None,
        loss_fn=None,
    )
    
    
    # Make dataloader
    
    os.environ["S3_ENDPOINT_URL"] = "http://128.232.115.19:9000"
    # Remote path where full dataset is persistently stored
    remote = 's3://loop-llm/smoll_corpus/fineweb-edu-dedup'

    # Local working dir where dataset is cached during operation
    local = '/nfs-share/pa511/code_bases/abbie/data_cache'
    dataset = StreamingTextDataset(tokenizer=tokenizer, max_seq_len=1024, local=local, remote=remote, shuffle=False, batch_size=2, predownload=1)
    train_dataloader = DataLoader(dataset, batch_size=2)

    # Make optimizer
    optimizer = DecoupledAdamW(model.parameters())
    
    scheduler = None
    eval_dataloader = None
    
    loggers = [wandb_logger]
    callbacks = get_callbacks(cfg.callbacks)
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        callbacks=callbacks,
        optimizers=optimizer,
        schedulers=scheduler,
        loggers=loggers,
        **trainer_config,
    )
    
    trainer.fit()

if __name__ == "__main__":
    main()
