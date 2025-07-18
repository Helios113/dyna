from transformers import AutoTokenizer
from model_config import ModelConfig, TrainerConfig
from model import ComposerMoEUT
from composer import Trainer
from streaming import StreamingDataset

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configuration", config_name="MoA")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
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

    # We don't need a tokenizer becuase all of our data is pre-tokenized.
    # tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
    
    model = ComposerMoEUT(
        config=model_config,
        tokenizer=None,
        additional_train_metrics=None,
        loss_fn=None,
    )
    optimizer = None
    scheduler = None
    train_dataloader = None
    eval_dataloader = None
    
    # trainer = Trainer(
    #     model=model,
    #     train_dataloader=train_dataloader,
    #     eval_dataloader=eval_dataloader,
    #     optimizers=optimizer,
    #     schedulers=scheduler,
    #     **trainer_config,
    # )
    
    # trainer.fit()

if __name__ == "__main__":
    main()
