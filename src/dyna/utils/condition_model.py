import os

from composer.utils import maybe_create_object_store_from_uri, parse_uri
from composer.utils.checkpoint import (
    download_checkpoint,
    safe_torch_load,
)

from dyna.model import ComposerDynaModel


def condition_model(model: ComposerDynaModel, keys: list, load_path):
    # state_dict = mode.state_dict()
    if load_path is not None and keys is not None:
        load_object_store = None
        if load_path is not None:
            if load_object_store is None:
                # Does not support WandBLogger check original
                # checkpointing code to extend if needed
                load_object_store = maybe_create_object_store_from_uri(load_path)

            _, _, parsed_load_path = parse_uri(load_path)

            # Download checkpoint to local file and name
            composer_states_filepath, extracted_checkpoint_folder, extracted_rank_n = (
                download_checkpoint(
                    path=parsed_load_path,
                    node_checkpoint_folder="",
                    object_store=load_object_store,
                    progress_bar=False,
                )
            )

            # Load state dict from local file
            state_dict = safe_torch_load(
                composer_states_filepath=composer_states_filepath,
                load_monolith_rank0_only=True,
            )
            # Load state dict into new model
            payload_dict = state_dict["state"]["model"]
            weights_to_load = {}
            for key in keys:
                if key in payload_dict:
                    weights_to_load[key] = payload_dict[key]

            # Update your model's state_dict with the filtered weights
            model_dict = model.state_dict()
            model_dict.update(weights_to_load)

            # Load the updated state_dict back into your model
            model.load_state_dict(model_dict)

            # set loaded weights from keys to be frozen
            for key in weights_to_load:
                param = model.state_dict()[key]
                param.requires_grad = False

            if os.path.exists(composer_states_filepath):
                os.remove(composer_states_filepath)
                print(f"File '{composer_states_filepath}' has been deleted.")
            else:
                print(f"The file '{composer_states_filepath}' does not exist.")
        # mode.load_state_dict(state_dict)
