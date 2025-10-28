import torch
from composer.core import State
from composer.loggers import WandBLogger
from composer.utils import checkpoint, maybe_create_object_store_from_uri, parse_uri

from dyna.model.base import DynaPretrainedModel


def condition_model(mode: DynaPretrainedModel, keys: list, load_path):
    state_dict = mode.state_dict()
    state = State()
    if load_path is not None and keys is not None:
        load_object_store = None
        if load_path is not None:
            log.info(f"Loading checkpoint from {load_path}")
            if load_object_store is None:
                load_object_store = maybe_create_object_store_from_uri(load_path)
                log.debug(f"Created object store from load path: {load_object_store}")
            if isinstance(load_object_store, WandBLogger):
                import wandb

                if wandb.run is None:
                    load_object_store.init(self.state, self.logger)
            _, _, parsed_load_path = parse_uri(load_path)
            log.debug(f"Parsed load path: {parsed_load_path}")

            # first we want to use download_checkpoint from checkpoint.py in composer utils

            # then we want to use load_checkpoint from checkpoint.py in composer utils to load into a random dict removing all refrences to proper state

            # finally, we want to get the required keys and load them into the model, potentially clean the local checkpoint.

            # remote_state = checkpoint.load_checkpoint(
            #     state=state,
            #     logger=self.logger,
            #     path=parsed_load_path,
            #     object_store=load_object_store,
            #     load_weights_only=load_weights_only,
            #     strict_model_weights=load_strict_model_weights,
            #     progress_bar=load_progress_bar,
            #     ignore_keys=load_ignore_keys,
            #     exclude_algorithms=load_exclude_algorithms,
            #     algorithm_passes=self.engine.algorithm_passes,
            # )

        for key in keys:
            if key in checkpoint_state_dict:
                if key in state_dict:
                    state_dict[key] = checkpoint_state_dict[key]
                    log.info(
                        f"Loaded key '{key}' from checkpoint into model state dict."
                    )
                else:
                    log.warning(f"Key '{key}' not found in model state dict.")
            else:
                log.warning(f"Key '{key}' not found in checkpoint state dict.")

        mode.load_state_dict(state_dict)
