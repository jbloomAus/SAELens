from collections.abc import Iterable
from math import ceil
from typing import Any

import pytest
import torch
from datasets import Dataset, IterableDataset
from transformer_lens import HookedTransformer

from sae_training.activations_store import ActivationsStore
from tests.unit.helpers import build_sae_cfg


def tokenize_with_bos(model: HookedTransformer, text: str) -> list[int]:
    assert model.tokenizer is not None
    assert model.tokenizer.bos_token_id is not None
    return [model.tokenizer.bos_token_id] + model.tokenizer.encode(text)


@pytest.fixture
def cfg():
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    return build_sae_cfg()


@pytest.fixture
def activation_store(cfg: Any, model: HookedTransformer):
    return ActivationsStore(cfg, model)


def test_activations_store__init__(cfg: Any, model: HookedTransformer):
    store = ActivationsStore(cfg, model)

    assert store.cfg == cfg
    assert store.model == model

    assert isinstance(store.dataset, IterableDataset)
    assert isinstance(store.iterable_dataset, Iterable)

    # I expect the dataloader to be initialised
    assert hasattr(store, "dataloader")

    # I expect the buffer to be initialised
    assert hasattr(store, "storage_buffer")

    # the rest is in the dataloader.
    expected_size = (
        cfg.store_batch_size * cfg.context_size * cfg.n_batches_in_buffer // 2
    )
    assert store.storage_buffer.shape == (expected_size, 1, cfg.d_in)


def test_activations_store__get_batch_tokens(activation_store: ActivationsStore):
    batch = activation_store.get_batch_tokens()

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (
        activation_store.cfg.store_batch_size,
        activation_store.cfg.context_size,
    )
    assert batch.device == activation_store.cfg.device


def test_activations_store__get_activations(activation_store: ActivationsStore):
    batch = activation_store.get_batch_tokens()
    activations = activation_store.get_activations(batch)

    cfg = activation_store.cfg
    assert isinstance(activations, torch.Tensor)
    assert activations.shape == (cfg.store_batch_size, cfg.context_size, 1, cfg.d_in)
    assert activations.device == cfg.device


def test_activations_store__get_activations_head_hook(model: HookedTransformer):
    cfg = build_sae_cfg(
        hook_point="blocks.0.attn.hook_q",
        hook_point_head_index=2,
        hook_point_layer=1,
        d_in=4,
    )
    activation_store_head_hook = ActivationsStore(cfg, model)
    batch = activation_store_head_hook.get_batch_tokens()
    activations = activation_store_head_hook.get_activations(batch)

    cfg = activation_store_head_hook.cfg
    assert isinstance(activations, torch.Tensor)
    assert activations.shape == (cfg.store_batch_size, cfg.context_size, 1, cfg.d_in)
    assert activations.device == cfg.device


def test_activations_store__get_buffer(activation_store: ActivationsStore):
    n_batches_in_buffer = 3
    buffer = activation_store.get_buffer(n_batches_in_buffer)

    cfg = activation_store.cfg
    assert isinstance(buffer, torch.Tensor)
    buffer_size_expected = cfg.store_batch_size * cfg.context_size * n_batches_in_buffer

    assert buffer.shape == (buffer_size_expected, 1, cfg.d_in)
    assert buffer.device == cfg.device


# 12 is divisible by the length of "hello world", 11 and 13 are not
@pytest.mark.parametrize("context_size", [11, 12, 13])
def test_activations_store__get_batch_tokens__fills_the_context_separated_by_bos(
    model: HookedTransformer, context_size: int
):
    assert model.tokenizer is not None
    dataset = Dataset.from_list(
        [
            {"text": "hello world"},
        ]
        * 100
    )
    cfg = build_sae_cfg(
        store_batch_size=2,
        context_size=context_size,
    )

    activation_store = ActivationsStore(
        cfg, model, dataset=dataset, create_dataloader=False
    )
    encoded_text = tokenize_with_bos(model, "hello world")
    tokens = activation_store.get_batch_tokens()
    assert tokens.shape == (2, context_size)  # batch_size x context_size
    all_expected_tokens = (encoded_text * ceil(2 * context_size / len(encoded_text)))[
        : 2 * context_size
    ]
    expected_tokens1 = all_expected_tokens[:context_size]
    expected_tokens2 = all_expected_tokens[context_size:]
    if expected_tokens2[0] != model.tokenizer.bos_token_id:
        expected_tokens2 = [model.tokenizer.bos_token_id] + expected_tokens2[:-1]
    assert tokens[0].tolist() == expected_tokens1
    assert tokens[1].tolist() == expected_tokens2


def test_activations_store__get_next_dataset_tokens__tokenizes_each_example_in_order(
    cfg: Any, model: HookedTransformer
):
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
    )
    activation_store = ActivationsStore(
        cfg, model, dataset=dataset, create_dataloader=False
    )

    assert activation_store._get_next_dataset_tokens().tolist() == tokenize_with_bos(
        model, "hello world1"
    )
    assert activation_store._get_next_dataset_tokens().tolist() == tokenize_with_bos(
        model, "hello world2"
    )
    assert activation_store._get_next_dataset_tokens().tolist() == tokenize_with_bos(
        model, "hello world3"
    )
