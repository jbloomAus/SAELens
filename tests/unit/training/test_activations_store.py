import os
from collections.abc import Iterable
from math import ceil
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset, IterableDataset
from transformer_lens import HookedTransformer

from sae_lens.config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
    PretokenizeRunnerConfig,
)
from sae_lens.load_model import load_model
from sae_lens.pretokenize_runner import pretokenize_dataset
from sae_lens.training.activations_store import FILE_EXTENSION, ActivationsStore
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
            "normalize_activations": "expected_average_only_in",
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_name": "blocks.1.attn.hook_z",
            "hook_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "gelu-2l",
            "dataset_path": "NeelNanda/c4-tokenized-2b",
            "tokenized": True,
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 512,
            "context_size": 1024,
        },
        {
            "model_name": "gpt2",
            "dataset_path": "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
            "tokenized": True,
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 768,
            "context_size": 1024,
        },
        {
            "model_name": "gpt2",
            "dataset_path": "Skylion007/openwebtext",
            "tokenized": False,
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 768,
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

    if cfg.normalize_activations == "expected_average_only_in":
        store.estimated_norm_scaling_factor = 10.399

    assert store.model == model
    assert isinstance(store.dataset, IterableDataset)
    assert isinstance(store.iterable_sequences, Iterable)

    # the rest is in the dataloader.
    expected_size = (
        cfg.store_batch_size_prompts * cfg.context_size * cfg.n_batches_in_buffer
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

    buffer = store.get_buffer()

    assert isinstance(buffer, np.memmap)
    buffer_size_expected = (
        store.store_batch_size_prompts * store.context_size * cfg.n_batches_in_buffer
    )

    assert buffer.shape == (buffer_size_expected, 1, store.d_in)

    # check the buffer norm
    if cfg.normalize_activations == "expected_average_only_in":
        example_activations = store.next_batch()
        assert torch.allclose(
            example_activations.norm(dim=-1),
            np.sqrt(store.d_in) * torch.ones_like(example_activations.norm(dim=-1)),
            atol=2,
        )


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

    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
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


def test_activations_store__iterate_raw_dataset_tokens__tokenizes_each_example_in_order(
    ts_model: HookedTransformer,
):
    tokenizer = ts_model.tokenizer
    assert tokenizer is not None
    cfg = build_sae_cfg()
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
    )
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    iterator = activation_store._iterate_raw_dataset_tokens()

    assert next(iterator).tolist() == tokenizer.encode("hello world1")
    assert next(iterator).tolist() == tokenizer.encode("hello world2")
    assert next(iterator).tolist() == tokenizer.encode("hello world3")


def test_activations_store__iterate_raw_dataset_tokens__can_handle_long_examples(
    ts_model: HookedTransformer,
):
    cfg = build_sae_cfg()
    dataset = Dataset.from_list(
        [
            {"text": " France" * 3000},
        ]
    )
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    iterator = activation_store._iterate_raw_dataset_tokens()

    assert len(next(iterator).tolist()) == 3000


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


def test_activations_store_estimate_norm_scaling_factor(ts_model: HookedTransformer):
    # --- first, test initialisation ---
    cfg = build_sae_cfg()
    model = ts_model
    store = ActivationsStore.from_config(model, cfg)
    factor = store.estimate_norm_scaling_factor(n_batches_for_norm_estimate=10)
    assert isinstance(factor, float)

    scaled_norm = torch.tensor(store.next_batch()).norm(dim=-1).mean() * factor  # type: ignore
    assert scaled_norm == pytest.approx(np.sqrt(store.d_in), abs=5)


def test_activations_store___iterate_tokenized_sequences__yields_concat_and_batched_sequences(
    ts_model: HookedTransformer,
):
    tokenizer = ts_model.tokenizer
    assert tokenizer is not None
    cfg = build_sae_cfg(prepend_bos=True, context_size=5)
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
    )
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    iterator = activation_store._iterate_tokenized_sequences()

    expected = [
        tokenizer.bos_token_id,
        *tokenizer.encode("hello world1"),
        tokenizer.bos_token_id,
        *tokenizer.encode("hello world2"),
        tokenizer.bos_token_id,
        *tokenizer.encode("hello world3"),
    ]
    assert next(iterator).tolist() == expected[:5]


def test_activations_store___iterate_tokenized_sequences__yields_sequences_of_context_size(
    ts_model: HookedTransformer,
):
    tokenizer = ts_model.tokenizer
    assert tokenizer is not None
    cfg = build_sae_cfg(prepend_bos=True, context_size=5)
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 20
    )
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    for toks in activation_store._iterate_tokenized_sequences():
        assert toks.shape == (5,)


def test_activations_store__errors_if_pretokenized_context_size_doesnt_match_cfg(
    ts_model: HookedTransformer,
):
    tokenizer = ts_model.tokenizer
    assert tokenizer is not None
    cfg = build_sae_cfg(prepend_bos=True, context_size=5)
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 20
    )
    pretokenize_cfg = PretokenizeRunnerConfig(context_size=10)
    tokenized_dataset = pretokenize_dataset(dataset, tokenizer, cfg=pretokenize_cfg)
    with pytest.raises(ValueError):
        ActivationsStore.from_config(ts_model, cfg, override_dataset=tokenized_dataset)


def test_activations_store___iterate_tokenized_sequences__yields_identical_results_with_and_without_pretokenizing(
    ts_model: HookedTransformer,
):
    tokenizer = ts_model.tokenizer
    assert tokenizer is not None
    cfg = build_sae_cfg(prepend_bos=True, context_size=5)
    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 20
    )
    pretokenize_cfg = PretokenizeRunnerConfig(
        context_size=5,
        num_proc=1,
        shuffle=False,
        begin_batch_token="bos",
        sequence_separator_token="bos",
    )
    tokenized_dataset = pretokenize_dataset(dataset, tokenizer, cfg=pretokenize_cfg)
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    tokenized_activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=tokenized_dataset
    )
    seqs = [seq.tolist() for seq in activation_store._iterate_tokenized_sequences()]
    pretok_seqs = [
        seq.tolist()
        for seq in tokenized_activation_store._iterate_tokenized_sequences()
    ]
    assert seqs == pretok_seqs


def test_activation_store__errors_if_neither_dataset_nor_dataset_path(
    ts_model: HookedTransformer,
):
    cfg = build_sae_cfg(dataset_path="")

    example_ds = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 20
    )

    ActivationsStore.from_config(ts_model, cfg, override_dataset=example_ds)

    with pytest.raises(ValueError):
        ActivationsStore.from_config(ts_model, cfg, override_dataset=None)


def test_activation_store_save_load_cls_methods():
    dtype = np.float32
    d_in = 1024

    memmap_filename = "tmp.dat"
    memmap_buffer = np.memmap(
        memmap_filename,
        dtype=dtype,
        mode="w+",
        shape=(10, 1, d_in),
    )

    ActivationsStore._save_buffer(memmap_buffer, memmap_filename)

    loaded_buffer = ActivationsStore._load_buffer(memmap_filename, num_layers=1, dtype=dtype, d_in=d_in)  # type: ignore

    assert loaded_buffer.shape == memmap_buffer.shape
    assert np.allclose(memmap_buffer, loaded_buffer)

    os.remove(memmap_filename)


def test_activation_store_save_load_buffer(
    tmp_path: Path,
):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    example_ds = Dataset.from_list(
        [
            {"text": "hello world1"},
            {"text": "hello world2"},
            {"text": "hello world3"},
        ]
        * 5000
    )

    # total_training_steps = 20_000
    context_size = 256
    n_batches_in_buffer = 4
    store_batch_size = 1
    n_buffers = 3

    tokens_in_buffer = n_batches_in_buffer * store_batch_size * context_size
    total_training_tokens = n_buffers * tokens_in_buffer

    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path),
        # new_cached_activations_path=cached_activations_fixture_path,
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        d_in=512,
        dataset_path="NeelNanda/c4-tokenized-2b",
        context_size=context_size,  # Speed things up.
        is_dataset_tokenized=True,
        prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=4096,
        # Loss Function
        ## Reconstruction Coefficient.
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=store_batch_size,
        normalize_activations="none",
        #
        shuffle_every_n_buffers=2,
        n_shuffles_with_last_section=1,
        n_shuffles_in_entire_dir=1,
        n_shuffles_final=1,
        # Misc
        device=device,
        seed=42,
        dtype="float16",
    )

    model = load_model(
        model_class_name=cfg.model_class_name,
        model_name=cfg.model_name,
        device=cfg.device,
        model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
    )

    store = ActivationsStore.from_config(model, cfg, override_dataset=example_ds)
    buffer_filename = f"{tmp_path}/{0}.{FILE_EXTENSION}"

    buffer = store.get_buffer()
    store.save_buffer(buffer, buffer_filename)
    loaded_buffer = store.load_buffer(buffer_filename)

    assert loaded_buffer.shape == buffer.shape
    assert np.allclose(buffer, loaded_buffer)

    os.remove(buffer_filename)
