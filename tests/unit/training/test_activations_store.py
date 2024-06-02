from collections.abc import Iterable
from math import ceil

import numpy as np
import pytest
import torch
from datasets import Dataset, IterableDataset
from transformer_lens import HookedTransformer

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.training.activations_store import ActivationsStore
from tests.unit.helpers import build_sae_cfg, load_model_cached


def tokenize_with_bos(model: HookedTransformer, text: str) -> list[int]:
    assert model.tokenizer is not None
    assert model.tokenizer.bos_token_id is not None
    return [model.tokenizer.bos_token_id] + model.tokenizer.encode(text)


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
            "prepend_bos": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_name": "blocks.1.attn.hook_z",
            "hook_layer": 1,
            "d_in": 64,
            "prepend_bos": True,
        },
        {
            "model_name": "gelu-2l",
            "dataset_path": "NeelNanda/c4-tokenized-2b",
            "tokenized": True,
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 512,
            "prepend_bos": True,
        },
        {
            "model_name": "gpt2",
            "dataset_path": "apollo-research/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
            "tokenized": True,
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 768,
            "prepend_bos": True,
        },
        {
            "model_name": "gpt2",
            "dataset_path": "Skylion007/openwebtext",
            "tokenized": False,
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 768,
            "prepend_bos": True,
        },
    ],
    ids=[
        "tiny-stories-1M-resid-pre",
        "tiny-stories-1M-attn-out",
        "gelu-2l-tokenized",
        "gpt2-tokenized",
        "gpt2",
    ],
)
def cfg(request: pytest.FixtureRequest) -> LanguageModelSAERunnerConfig:
    # This function will be called with each parameter set
    params = request.param
    return build_sae_cfg(**params)


@pytest.fixture
def model(cfg: LanguageModelSAERunnerConfig):
    return load_model_cached(cfg.model_name)


# tests involving loading real models / real datasets are very slow
# so do lots of stuff in this one test to make each load of model / data count
# poetry run py.test tests/unit/training/test_activations_store.py -k 'test_activations_store__shapes_look_correct_with_real_models_and_datasets' --profile-svg -s
def test_activations_store__shapes_look_correct_with_real_models_and_datasets(
    cfg: LanguageModelSAERunnerConfig, model: HookedTransformer
):
    # --- first, test initialisation ---

    # config if you want to benchmark this:
    #
    # cfg.context_size = 1024
    # cfg.n_batches_in_buffer = 64
    # cfg.store_batch_size_prompts = 16

    store = ActivationsStore.from_config(model, cfg)

    assert store.model == model

    assert isinstance(store.dataset, IterableDataset)
    assert isinstance(store.iterable_dataset, Iterable)

    # the rest is in the dataloader.
    expected_size = (
        cfg.store_batch_size_prompts * cfg.context_size * cfg.n_batches_in_buffer // 2
    )
    assert store.storage_buffer.shape == (expected_size, 1, cfg.d_in)

    # --- Next, get batch tokens and assert they look correct ---

    batch = store.get_batch_tokens()

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (
        store.store_batch_size_prompts,
        store.context_size,
    )
    assert batch.device == store.device

    # --- Next, get activations and assert they look correct ---

    activations = store.get_activations(batch)

    assert isinstance(activations, torch.Tensor)
    assert activations.shape == (
        store.store_batch_size_prompts,
        store.context_size,
        1,
        store.d_in,
    )
    assert activations.device == store.device

    # --- Next, get buffer and assert it looks correct ---

    n_batches_in_buffer = 3
    buffer = store.get_buffer(n_batches_in_buffer)

    assert isinstance(buffer, torch.Tensor)
    buffer_size_expected = (
        store.store_batch_size_prompts * store.context_size * n_batches_in_buffer
    )

    assert buffer.shape == (buffer_size_expected, 1, store.d_in)
    assert buffer.device == store.device

    # # check the buffer norm
    # if cfg.normalize_activations:
    #     assert torch.allclose(
    #         buffer.norm(dim=-1), torch.ones_like(buffer.norm(dim=-1)), atol=1e-6
    #     )


def test_activations_store__get_activations_head_hook(ts_model: HookedTransformer):
    cfg = build_sae_cfg(
        hook_name="blocks.0.attn.hook_q",
        hook_head_index=2,
        hook_layer=1,
        d_in=4,
    )
    activation_store_head_hook = ActivationsStore.from_config(ts_model, cfg)
    batch = activation_store_head_hook.get_batch_tokens()
    activations = activation_store_head_hook.get_activations(batch)

    assert isinstance(activations, torch.Tensor)
    assert activations.shape == (
        activation_store_head_hook.store_batch_size_prompts,
        activation_store_head_hook.context_size,
        1,
        activation_store_head_hook.d_in,
    )
    assert activations.device == activation_store_head_hook.device


# 12 is divisible by the length of "hello world", 11 and 13 are not
@pytest.mark.parametrize("context_size", [11, 12, 13])
def test_activations_store__get_batch_tokens__fills_the_context_separated_by_bos(
    ts_model: HookedTransformer, context_size: int
):
    assert ts_model.tokenizer is not None
    dataset = Dataset.from_list(
        [
            {"text": "hello world"},
        ]
        * 100
    )
    cfg = build_sae_cfg(
        store_batch_size_prompts=2,
        context_size=context_size,
    )

    activation_store = ActivationsStore.from_config(ts_model, cfg, dataset=dataset)
    encoded_text = tokenize_with_bos(ts_model, "hello world")
    tokens = activation_store.get_batch_tokens()
    assert tokens.shape == (2, context_size)  # batch_size x context_size
    all_expected_tokens = (encoded_text * ceil(2 * context_size / len(encoded_text)))[
        : 2 * context_size
    ]
    expected_tokens1 = all_expected_tokens[:context_size]
    expected_tokens2 = all_expected_tokens[context_size:]
    if expected_tokens2[0] != ts_model.tokenizer.bos_token_id:
        expected_tokens2 = [ts_model.tokenizer.bos_token_id] + expected_tokens2[:-1]
    assert tokens[0].tolist() == expected_tokens1
    assert tokens[1].tolist() == expected_tokens2


def test_activations_store__get_next_dataset_tokens__tokenizes_each_example_in_order(
    ts_model: HookedTransformer,
):
    cfg = build_sae_cfg()
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
    )
    activation_store = ActivationsStore.from_config(ts_model, cfg, dataset=dataset)

    assert activation_store._get_next_dataset_tokens().tolist() == tokenize_with_bos(
        ts_model, "hello world1"
    )
    assert activation_store._get_next_dataset_tokens().tolist() == tokenize_with_bos(
        ts_model, "hello world2"
    )
    assert activation_store._get_next_dataset_tokens().tolist() == tokenize_with_bos(
        ts_model, "hello world3"
    )


def test_activations_store__get_next_dataset_tokens__can_handle_long_examples(
    ts_model: HookedTransformer,
):
    cfg = build_sae_cfg()
    dataset = Dataset.from_list(
        [
            {"text": " France" * 3000},
        ]
    )
    activation_store = ActivationsStore.from_config(ts_model, cfg, dataset=dataset)

    assert len(activation_store._get_next_dataset_tokens().tolist()) == 3001


def test_activations_store_goes_to_cpu(ts_model: HookedTransformer):
    cfg = build_sae_cfg(act_store_device="cpu")
    activation_store = ActivationsStore.from_config(ts_model, cfg)
    activations = activation_store.next_batch()
    assert activations.device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU to test on.")
def test_activations_store_with_model_on_gpu(ts_model: HookedTransformer):
    cfg = build_sae_cfg(act_store_device="cpu", device="cuda:0")
    activation_store = ActivationsStore.from_config(ts_model.to("cuda:0"), cfg)  # type: ignore
    activations = activation_store.next_batch()
    assert activations.device == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU to test on.")
def test_activations_store_moves_with_model(ts_model: HookedTransformer):
    # "with_model" resets to default so the second post_init in build_sae_cfg works
    cfg = build_sae_cfg(act_store_device="with_model", device="cuda:0")
    activation_store = ActivationsStore.from_config(ts_model.to("cuda:0"), cfg)  # type: ignore
    activations = activation_store.next_batch()
    assert activations.device == torch.device("cuda:0")


def test_activations_store_estimate_norm_scaling_factor(
    cfg: LanguageModelSAERunnerConfig, model: HookedTransformer
):
    # --- first, test initialisation ---

    # config if you want to benchmark this:
    #
    # cfg.context_size = 1024
    # cfg.n_batches_in_buffer = 64
    # cfg.store_batch_size_prompts = 16

    store = ActivationsStore.from_config(model, cfg)

    factor = store.estimate_norm_scaling_factor(n_batches_for_norm_estimate=10)
    assert isinstance(factor, float)

    scaled_norm = store._storage_buffer.norm(dim=-1).mean() * factor  # type: ignore
    assert scaled_norm == pytest.approx(np.sqrt(store.d_in), abs=5)
