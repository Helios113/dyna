"""Local builders replacing llm-foundry helpers."""
from __future__ import annotations

from enum import Enum
import os
from typing import Any, TYPE_CHECKING, cast
import copy
import json
import logging
import transformers
from composer import DataSpec
from composer.core import Algorithm, Callback, Evaluator
from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity
from composer.optim.scheduler import ComposerScheduler
from torchmetrics import Metric
if TYPE_CHECKING:
    from dyna.callbacks.eval_gauntlet import EvalGauntlet
from dyna.utils.utils import to_dict_container, to_list_container, construct_from_registry
from transformers import PreTrainedTokenizerBase
import omegaconf as om
from datasets import Dataset as HFDataset
from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase
from omegaconf import DictConfig, OmegaConf
from composer.utils import dist, get_file
from dyna import registry
from dyna.registry import metrics
log = logging.getLogger(__name__)

try:  # Composer >=0.20 removed TokenAccuracy
    from composer.metrics.nlp import TokenAccuracy
except ImportError:  # pragma: no cover - Composer version mismatch
    from composer.metrics.nlp import MaskedAccuracy as TokenAccuracy

from dyna.metrics.eval_metrics import (
    InContextLearningLMAccuracy,
    InContextLearningLMExpectedCalibrationError,
    InContextLearningMultipleChoiceAccuracy,
    InContextLearningGenerationExactMatchAccuracy,
    InContextLearningMCExpectedCalibrationError,
)
from dyna.registry import callbacks, norms, schedulers


def _normalize_name(name: str | Enum) -> str:
    if isinstance(name, Enum):
        return str(name.value)
    return str(name)



def build_norm(name: str | Enum, **kwargs: Any):
    """Instantiate a registered normalization module."""
    try:
        factory = norms.get(_normalize_name(name))
    except KeyError as exc:  # pragma: no cover - developer error
        raise ValueError(f"Unknown norm type '{name}'") from exc
    return factory(**kwargs)


metrics.register('token_accuracy', func=TokenAccuracy)
metrics.register('lm_accuracy', func=InContextLearningLMAccuracy)
metrics.register(
    'lm_expected_calibration_error',
    func=InContextLearningLMExpectedCalibrationError,
)
metrics.register(
    'mc_expected_calibration_error',
    func=InContextLearningMCExpectedCalibrationError,
)
metrics.register('mc_accuracy', func=InContextLearningMultipleChoiceAccuracy)
metrics.register(
    'qa_accuracy',
    func=InContextLearningGenerationExactMatchAccuracy,
)
metrics.register('language_cross_entropy', func=LanguageCrossEntropy)
metrics.register('language_perplexity', func=LanguagePerplexity)

DEFAULT_CAUSAL_LM_TRAIN_METRICS = [
    'language_cross_entropy',
    'language_perplexity',
    'token_accuracy',
]

DEFAULT_CAUSAL_LM_EVAL_METRICS = [
    'language_cross_entropy',
    'language_perplexity',
    'token_accuracy',
    'lm_accuracy',
    'lm_expected_calibration_error',
    'mc_expected_calibration_error',
    'mc_accuracy',
    'qa_accuracy',
]

DEFAULT_ENC_DEC_METRICS = [
    'language_cross_entropy',
    'masked_accuracy',
]


def build_metric(name: str, kwargs: dict[str, Any] | None = None) -> Metric:
    """Builds a metric from the registry."""
    return construct_from_registry(
        name=name,
        registry=registry.metrics,
        partial_function=True,
        pre_validation_function=Metric,
        post_validation_function=None,
        kwargs=kwargs,
    )


def build_callback(
    name: str,
    kwargs: dict[str, Any] | None = None,
) -> Any:
    """Instantiate a callback by name from the callback registry."""
    try:
        factory = callbacks.get(name)
    except KeyError as exc:  # pragma: no cover - developer error
        raise ValueError(f"Unknown callback '{name}'") from exc
    callback_kwargs = kwargs or {}
    return factory(**callback_kwargs)


def build_scheduler(
    name: str,
    scheduler_config: dict[str, Any] | None = None,
):
    """Instantiate a scheduler from the scheduler registry."""
    try:
        factory = schedulers.get(name)
    except KeyError as exc:  # pragma: no cover - developer error
        raise ValueError(f"Unknown scheduler '{name}'") from exc
    config = scheduler_config or {}
    return factory(**config)



def build_evaluators(
    eval_loader_config: dict[str, Any] | list[dict[str, Any]] | None,
    icl_tasks_config: str | list[dict[str, Any]] | None,
    eval_gauntlet_config: str | dict[str, Any] | None,
    *,
    tokenizer: PreTrainedTokenizerBase | None,
    device_eval_batch_size: int | float,
    icl_seq_len: int,
    icl_subset_num_batches: int | None,
) -> tuple[list[Evaluator], list[str], EvalGauntlet | None]:

    evaluators = []
    if eval_loader_config is not None:
        evaluators = build_eval_loaders(
            eval_loader_config,
            tokenizer,
            device_eval_batch_size,
        )

    logger_keys = []
    eval_gauntlet_callback = None
    if icl_tasks_config is not None:
        if tokenizer is None:
            raise ValueError('Tokenizer is required for icl tasks')
        if not isinstance(device_eval_batch_size, int):
            raise ValueError(
                'device_eval_batch_size should be an int for icl tasks.',
            )

        icl_evaluators, logger_keys, eval_gauntlet_callback = build_icl_data_and_gauntlet(
            icl_tasks_config,
            eval_gauntlet_config,
            tokenizer,
            device_eval_batch_size,
            icl_seq_len,
            icl_subset_num_batches,
        )
        evaluators.extend(icl_evaluators)

    return evaluators, logger_keys, eval_gauntlet_callback


def build_eval_loaders(
    eval_loader_config: dict[str, Any] | list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase | None,
    device_eval_batch_size: int | float,
) -> list[Evaluator]:
    evaluators: list[Evaluator] = []
    if isinstance(eval_loader_config, list):
        eval_configs = eval_loader_config
        is_multi_eval = True
    elif isinstance(eval_loader_config, dict):
        eval_configs = [eval_loader_config]
        is_multi_eval = False
    else:
        raise ValueError(
            f'Got invalid type for eval_loader_config: {type(eval_loader_config)}, {eval_loader_config=}',
        )

    for eval_config in eval_configs:
        label = eval_config.pop('label') if is_multi_eval else None
        eval_dataloader = build_dataloader(
            eval_config,
            tokenizer,
            device_eval_batch_size,
        )
        eval_loader: Evaluator = Evaluator(
            label=f'eval/{label}' if is_multi_eval else 'eval',
            dataloader=eval_dataloader,
            # Load the eval data to fail fast. metrics will get added
            # later in add_metrics_to_eval_loaders, after the model is loaded
            metric_names=[],
            device_eval_microbatch_size=device_eval_batch_size,
        )
        evaluators.append(eval_loader)
    return evaluators


def add_metrics_to_eval_loaders(
    evaluators: list[Evaluator],
    metric_names: list[str],
) -> list[Evaluator]:
    eval_loaders, other_evaluators = [], []
    for evaluator in evaluators:
        if evaluator.metric_names == []:
            evaluator.metric_names = metric_names
            eval_loaders.append(evaluator)
        else:
            other_evaluators.append(evaluator)

    # Put the base eval_loaders first
    return eval_loaders + other_evaluators

def build_icl_data_and_gauntlet(
    icl_tasks_config: str | list[dict[str, Any]],
    eval_gauntlet_config: str | dict[str, Any] | None,
    tokenizer: PreTrainedTokenizerBase,
    device_eval_batch_size: int,
    icl_seq_len: int,
    icl_subset_num_batches: int | None = None,
) -> tuple[list[Evaluator], list[str], EvalGauntlet | None]:
    from dyna.callbacks.eval_gauntlet import EvalGauntlet
    
    icl_evaluators, logger_keys = build_icl_evaluators(
        icl_tasks_config,
        tokenizer,
        icl_seq_len,
        device_eval_batch_size,
        icl_subset_num_batches=icl_subset_num_batches,
    )
    eval_gauntlet_cb = None
    if eval_gauntlet_config is not None:
        if isinstance(eval_gauntlet_config, str):
            with open(eval_gauntlet_config, 'r') as icl_f:
                eval_gauntlet_cfg = om.load(icl_f)
                assert isinstance(eval_gauntlet_cfg, DictConfig)
            eval_gauntlet = to_dict_container(
                eval_gauntlet_cfg['eval_gauntlet'],
            )
        elif isinstance(eval_gauntlet_config, dict):  # pyright: ignore
            eval_gauntlet = eval_gauntlet_config
        else:
            raise ValueError(
                f'Got invalid type for eval_gauntlet_config: {type(eval_gauntlet_config)}',
            )
        eval_gauntlet['logger_keys'] = logger_keys
        eval_gauntlet['benchmark_sizes'] = {
            e.label: e.dataloader.num_samples for e in icl_evaluators
        }
        eval_gauntlet_cb = EvalGauntlet(**eval_gauntlet)
    return icl_evaluators, logger_keys, eval_gauntlet_cb

def build_dataloader(
    cfg: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase | None,
    device_batch_size: int | float,
) -> DataSpec:
    """Builds a dataloader from a config.

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the loader.
        tokenizer (PreTrainedTokenizerBase | None): The tokenizer that the model will use.
        device_batch_size (int): The size of the batches (number of examples)
            that the dataloader will produce.
    """
    name = cfg.pop('name')
    kwargs: dict[str, Any] = {
        **cfg,
        'tokenizer': tokenizer,
        'device_batch_size': device_batch_size,
    }

    return construct_from_registry(
        name=name,
        registry=registry.dataloaders,
        partial_function=False,
        pre_validation_function=None,
        post_validation_function=None,
        kwargs=kwargs,
    )

def build_icl_evaluators(
    icl_tasks: str | list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    default_max_seq_len: int,
    default_batch_size: int,
    destination_dir: str | None = None,
    icl_subset_num_batches: int | None = None,
) -> tuple[list[Evaluator], list[str]]:
    if destination_dir is None:
        destination_dir = os.getcwd()

    evaluators = []
    logger_keys = []

    icl_tasks_list = None
    if isinstance(icl_tasks, str):
        log.info(f'Extracting ICL task config from path: {icl_tasks}')
        with open(icl_tasks, 'r') as icl_f:
            icl_task_cfg = om.load(icl_f)
        icl_tasks_list = to_list_container(icl_task_cfg.icl_tasks)
    else:
        icl_tasks_list = icl_tasks

    def _validate_cfg(icl_cfg: dict[str, Any]):
        assert 'label' in icl_cfg
        assert 'dataset_uri' in icl_cfg and icl_cfg['dataset_uri'] is not None
        assert 'icl_task_type' in icl_cfg
        assert 'num_fewshot' in icl_cfg

        if 'metric_names' not in icl_cfg:
            if icl_cfg['icl_task_type'] == 'language_modeling':
                icl_cfg['metric_names'] = ['InContextLearningLMAccuracy']
            elif icl_cfg['icl_task_type'] == 'multiple_choice':
                print("Setting metric names for multiple_choice", flush=True)
                icl_cfg['metric_names'] = [
                    'InContextLearningMultipleChoiceAccuracy',
                ]
            elif icl_cfg['icl_task_type'] == 'schema':
                icl_cfg['metric_names'] = [
                    'InContextLearningMultipleChoiceAccuracy',
                ]
            elif icl_cfg['icl_task_type'] == 'generation_task_with_answers':
                icl_cfg['metric_names'] = [
                    'InContextLearningGenerationExactMatchAccuracy',
                ]
            else:
                icl_task_type = icl_cfg['icl_task_type']
                raise ValueError(
                    f'No metric_names defined, unable to build default metrics for icl_task_type={icl_task_type}.',
                )

        if 'max_seq_len' not in icl_cfg:
            icl_cfg['max_seq_len'] = default_max_seq_len
        if 'batch_size' not in icl_cfg:
            icl_cfg['batch_size'] = default_batch_size

        if 'num_beams' in icl_cfg:
            raise ValueError(
                'num_beams is no longer supported as a top level icl_task parameter.'  + \
                'Please use generation_kwargs.num_beams instead.')

    for icl_cfg in icl_tasks_list:
        assert isinstance(
            icl_cfg,
            dict,
        ), f'Expected dict, got {type(icl_cfg)}, {icl_cfg=}'
        _validate_cfg(icl_cfg)
        for num_fewshot in list(icl_cfg['num_fewshot']):
            if tokenizer.pad_token_id is None:
                # Current workaround to support GPT2 tokenizer with `pad_token_id = None`
                pad_tok_id = tokenizer.eos_token_id
            else:
                pad_tok_id = tokenizer.pad_token_id

            icl_cfg_label = icl_cfg['label']
            label = f'{icl_cfg_label}/{num_fewshot}-shot'
            metric_names = list(icl_cfg['metric_names'])
            # TODO: fix Composer bug when copying local paths and destination exists
            destination_path = f'{destination_dir}/{icl_cfg_label}-{num_fewshot}.jsonl'
            if dist.get_local_rank() == 0 and os.path.exists(destination_path):
                os.remove(destination_path)
            dist.barrier()

            hf_parsing_map = icl_cfg.get('hf_parsing_map', {})
            hf_loading_vars = icl_cfg.get('hf_loading_vars', {})
            early_stopping_criteria = icl_cfg.get(
                'early_stopping_criteria',
                [],
            )
            # TODO: fix manual removal of non-constructor fields
            icl_constructor_kwargs = copy.deepcopy(icl_cfg)
            icl_constructor_kwargs.pop('label', None)
            icl_constructor_kwargs.pop('metric_names', None)
            icl_constructor_kwargs.pop('icl_task_type', None)
            icl_constructor_kwargs.pop('batch_size', None)
            icl_constructor_kwargs.pop('has_categories', None)

            # Add custom constructor arguments
            icl_constructor_kwargs['pad_tok_id'] = pad_tok_id
            icl_constructor_kwargs['num_fewshot'] = num_fewshot

            # Support backwards compatibility for the naming of "prelimiter" as "question_prelimiter"
            if 'question_prelimiter' in icl_constructor_kwargs:
                if 'prelimiter' in icl_constructor_kwargs:
                    raise ValueError(
                        'Both "question_prelimiter" and "prelimiter" are specified in the ICL task config. '
                        +
                        'Please only specify one of them, as they map to the same argument.',
                    )
                else:
                    icl_constructor_kwargs['prelimiter'
                                          ] = icl_constructor_kwargs.pop(
                                              'question_prelimiter',
                                          )

            assert early_stopping_criteria is None or isinstance(
                early_stopping_criteria,
                list,
            )

            dataloaders = get_icl_task_dataloader(
                icl_task_type=icl_cfg['icl_task_type'],
                dataset_uri=icl_cfg['dataset_uri'],
                tokenizer=tokenizer,
                batch_size=icl_cfg['batch_size'],
                hf_loading_vars=hf_loading_vars,
                hf_parsing_map=hf_parsing_map,
                has_categories=icl_cfg.get('has_categories', False),
                destination_path=destination_path,
                kwargs=icl_constructor_kwargs,
            )
            if 'has_categories' in icl_cfg and icl_cfg[
                'has_categories'] and isinstance(dataloaders, dict):
                for category in dataloaders.keys():
                    logger_keys.extend([
                        f'metrics/{label}/{category}/{m}' for m in metric_names
                    ])
                    evaluators.append(
                        Evaluator(
                            label=f'{label}/{category}',
                            dataloader=dataloaders[category],
                            metric_names=metric_names,
                        ),
                    )
            else:
                logger_keys.extend([
                    f'metrics/{label}/{m}' for m in metric_names
                ])
                evaluators.append(
                    Evaluator(
                        label=label,
                        dataloader=dataloaders,
                        metric_names=metric_names,
                        subset_num_batches=icl_subset_num_batches,
                    ),
                )
    print("Logger keys:", logger_keys, flush=True)
    print("Evaluators:", evaluators, flush=True)
    return evaluators, logger_keys

def get_icl_task_dataloader(
    icl_task_type: str,
    dataset_uri: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    has_categories: bool = False,
    hf_loading_vars: dict | None = None,
    hf_parsing_map: dict | None = None,
    destination_path: str = '',
    kwargs: dict[str, Any] | None = None,
) -> DataSpec | dict[str, DataSpec]:
    r"""Constructs a dataloader (or dataloaders if has_categories is True)

    capable of evaluating LLMs on in-context learning language modeling tasks,
    for example LAMBADA. An example usage is below:

        .. testsetup::

            import transformers
            from composer.models import HuggingFaceModel
            from composer.trainer import Trainer
            dataset_uri = "/tmp/dataset_uri.jsonl"
            dataset = RandomTextClassificationDataset(size=16, use_keys=True)
            train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
            hf_model, tokenizer = HuggingFaceModel.hf_from_composer_checkpoint('composer-hf-checkpoint.pt')
            # At this point, hf_model is randomly initialized
            composer_model = HuggingFaceModel(hf_model, hf_tokenizer)

    Example:
        .. testcode::


            dl = get_icl_task_dataloader(
                'language_modeling',
                dataset_uri,
                tokenizer,
                batch_size=2,
                max_seq_len=2048,
                pad_tok_id=tokenizer.pad_token_id,
                num_fewshot=10,
                prompt_string='translate english to french',
                example_delimiter='\\n',
                continuation_delimiter=''
                )
            eval_evaluator = Evaluator(
                    label="lambada",
                    dataloader=dl,
                    metric_names=['InContextLearningLMAccuracy']
                )
            trainer = Trainer(
                    model=model,
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_evaluator,
                    optimizers=optimizer,
                    max_duration="1ep",
                )

    Args:
        icl_task_type (str): Name of icl_task type. One of ['multiple_choice', 'schema', 'language_modeling', 'generation_task_with_answers', 'code_evaluation']
        dataset_uri (str): A local path, a remote path beginning with ``s3://`` or another backend, or a HuggingFace dataset uri prepended with ``hf://``.
            Alternate backends must be supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
            A local dataset must consist of rows of JSON data points with task dependant fields.
            The default keys expected are "context" and "answer".
        tokenizer (transformers.PreTrainedTokenizerBase): The tokenizer used to map between strings and token ids.
        batch_size (int): Size of a batch used for eval
        has_categories: (bool): If ``True``, we will search the dataset file for a category key, and partition the dataset into a separate dataloader for each category occurring in the data.
        hf_loading_vars (Dict, default = None): A dictionary containing keyword arguments to be passed into `load_dataset` if dataset is being pulled from HF.
        hf_parsing_map (Dict, default = None): A dictionary containing a mapping from HF columns to ICL dataset keys. The dictionary should be formatted {icl_key:[hf_key1, hf_key1]}.
            Column contents will be concatenated with ' ' separating them. If not included, will load the columns already present in the HF dataset.
        destination_path: Where the dataloader will be saved.
        kwargs (Dict[str, Any], default=None): Dictionary containing a mapping from ICL dataset constructor's parameter names and their desired values.

    Returns:
        DataLoader: A dataloader used for performing in-context learning evaluation on the dataset provided.
    """
    if hf_loading_vars is None:
        hf_loading_vars = {}
    if hf_parsing_map is None:
        hf_parsing_map = {}
    if has_categories:
        result_dls = {}
        output_files = partition_dataset_by_category(
            dataset_uri,
            destination_path,
            hf_loading_vars,
            hf_parsing_map,
        )
        categories = sorted(output_files.keys())
        for category in categories:
            partition_uri = output_files[category]
            result_dls[category] = build_icl_dataloader(
                icl_task_type=icl_task_type,
                dataset_uri=partition_uri,
                tokenizer=tokenizer,
                batch_size=batch_size,
                destination_path=partition_uri + '_tmp',
                hf_loading_vars=hf_loading_vars,
                hf_parsing_map=hf_parsing_map,
                kwargs=kwargs,
            )
        return result_dls
    else:
        return build_icl_dataloader(
            icl_task_type=icl_task_type,
            dataset_uri=dataset_uri,
            tokenizer=tokenizer,
            batch_size=batch_size,
            hf_loading_vars=hf_loading_vars,
            hf_parsing_map=hf_parsing_map,
            destination_path=destination_path,
            kwargs=kwargs,
        )
def partition_dataset_by_category(
    dataset_uri: str,
    destination_path: str,
    hf_loading_vars: dict,
    hf_parsing_map: dict,
) -> dict[str, str]:
    """If has_categories is enabled, we partition the dataset into a separate.

    dataset for each category value in the data and write each partition to a
    local file.

    Args:
        dataset_uri (str): Location of dataset.
        destination_path (str): Base destination path, we will write a separate partition off this URI for each category.
        hf_loading_vars (Dict): A dictionary containing keyword arguments to be passed into `load_dataset` if dataset is being pulled from HF.
        hf_parsing_map (Dict): A dictionary containing a mapping from HF columns to ICL dataset keys. The dictionary should be formatted {icl_key:[hf_key1, hf_key1]}.
            Column contents will be concatenated with ' ' separating them. If not included, will load the columns already present in the HF dataset.


    Raises:
        MissingConditionalImportError: If datasets not installed raise exception.
        Exception: If 'category' key missing from dataset, raise exception.

    Returns:
        Dict[str, str]: Mapping of category names to partitioned dataset local files names.
    """
    if dataset_uri.startswith('hf://'):
        dataset_uri = dataset_uri.replace('hf://', '')
        dataset = load_dataset(dataset_uri, **hf_loading_vars)
        assert isinstance(dataset,
                          HFDataset) or isinstance(dataset, IterableDataset)
        if hf_parsing_map:
            dataset_parsing_func = lambda example: {
                k: ' '.join([str(example[col]) for col in v])
                for k, v in hf_parsing_map.items()
            }
            assert hasattr(dataset, 'column_names')
            dataset = dataset.map(
                dataset_parsing_func,
                remove_columns=dataset.column_names,
            )
    else:
        with dist.local_rank_zero_download_and_wait(destination_path):
            if dist.get_local_rank() == 0:
                get_file(dataset_uri, destination_path, overwrite=True)
        dataset = load_dataset(
            'json',
            data_files=destination_path,
            split='train',
            streaming=False,
        )
    assert isinstance(dataset,
                      HFDataset) or isinstance(dataset, IterableDataset)
    assert hasattr(dataset, 'features')
    assert dataset.features is not None
    if 'category' not in dataset.features.keys():
        raise Exception(
            f"""Attempted to partition dataset by `category` \
            but it doesn't have a `category` key. \
            Got keys: {str(list(dataset.features.keys()))}""",
        )
    categories = sorted(
        set(
            dataset['category'],
        ),
    )  # pyright: ignore[reportIndexIssue, reportGeneralTypeIssues]
    output_files = {}
    for cat in categories:
        path = destination_path.split('/')
        cat_dest = '/'.join(path[:-1]) + f'/{cat}_{path[-1]}'
        tmp_path_to_broadcast = str(os.path.abspath(cat_dest))
        gathered_paths = dist.all_gather_object(tmp_path_to_broadcast)
        if dist.get_local_rank() == 0:
            subset = [
                l for l in dataset if
                l['category'] == cat  # pyright: ignore[reportGeneralTypeIssues]
            ]  # pyright: ignore[reportArgumentType, reportCallIssue]
            with open(gathered_paths[0], 'w', encoding='utf8') as f:
                for l in subset:
                    f.write(json.dumps(l, ensure_ascii=False) + '\n')
        output_files[cat] = cat_dest
    return output_files

def build_icl_dataloader(
    icl_task_type: str,
    dataset_uri: str,
    tokenizer: transformers.PreTrainedTokenizerBase,
    batch_size: int,
    hf_loading_vars: dict,
    hf_parsing_map: dict,
    destination_path: str = '',
    kwargs: dict[str, Any] | None = None,
) -> DataSpec:
    """Factory method that builds the specific dataset for the specified.

    icl_task_type. See documentation for `get_icl_task_dataloader` for argument
    documentation.

    When writing a dataset for a new task, here you will need to:
        1. add the dataset to the factory and choose an appropriate string
        2. set the batch size for that task (see InContextLearningMultipleChoiceTaskDataset for why
            this might be different)
        3. set the `split_batch` function if necessary
    """
    # Add named parameters to kwargs
    if kwargs is None:
        kwargs = {}
    kwargs.update({
        'dataset_uri': dataset_uri,
        'tokenizer': tokenizer,
        'hf_loading_vars': hf_loading_vars,
        'hf_parsing_map': hf_parsing_map,
        'destination_path': destination_path,
    })
    dataset = construct_from_registry(
        name=icl_task_type,
        registry=registry.icl_datasets,
        partial_function=False,
        pre_validation_function=None,
        post_validation_function=None,
        kwargs=kwargs,
    )
    sampler = dist.get_sampler(dataset, drop_last=False, shuffle=False)

    return DataSpec(
        DataLoader(
            dataset,
            batch_size=dataset.get_effective_batch_size(batch_size),
            sampler=sampler,
            collate_fn=dataset.collate_fn,
        ),
        get_num_samples_in_batch=dataset.get_num_samples_in_batch,
        split_batch=dataset.split_batch,
    )

def get_callbacks(cfg: DictConfig | dict[str, Any] | None) -> list[Callback]:
    # Lazy import to avoid circular dependency
    
    if cfg is None:
        return []

    callbacks: list[Callback] = []
    for name, callback_cfg in cfg.items():
        if isinstance(callback_cfg, DictConfig):
            callback_kwargs = cast(
                dict[str, Any], OmegaConf.to_container(callback_cfg, resolve=True)
            )
        else:
            callback_kwargs = cast(dict[str, Any], callback_cfg) if callback_cfg else {}

        callbacks.append(
            build_callback(
                name=str(name),
                kwargs=callback_kwargs,
            )
        )

    return callbacks

def get_scheduler(cfg: DictConfig) -> ComposerScheduler:
    
    cfg_dict = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    scheduler_name = cfg_dict.pop("name")
    return build_scheduler(name=scheduler_name, scheduler_config=cfg_dict)

__all__ = [
    "build_norm",
    "build_metric",
    "build_callback",
    "build_scheduler",
]
