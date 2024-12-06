import os
import tempfile
from collections.abc import Iterable
from math import ceil
from typing import Any, Optional

import numpy as np
import pytest
import torch
from datasets import Dataset, IterableDataset
from safetensors.torch import load_file
from transformer_lens import HookedTransformer

from sae_lens.config import LanguageModelSAERunnerConfig, PretokenizeRunnerConfig
from sae_lens.load_model import load_model
from sae_lens.pretokenize_runner import pretokenize_dataset
from sae_lens.training.activations_store import (
    ActivationsStore,
    validate_pretokenized_dataset_tokenizer,
)
from tests.unit.helpers import build_sae_cfg, load_model_cached


def tokenize_with_bos(model: HookedTransformer, text: str) -> list[int]:
    assert model.tokenizer is not None
    assert model.tokenizer.bos_token_id is not None
    return [model.tokenizer.bos_token_id] + model.tokenizer.encode(text)  # type: ignore


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
            "normalize_activations": "expected_average_only_in",
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "hook_name": "blocks.1.attn.hook_z",
            "hook_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "gelu-2l",
            "dataset_path": "NeelNanda/c4-tokenized-2b",
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 512,
            "context_size": 1024,
        },
        {
            "model_name": "gpt2",
            "dataset_path": "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 768,
            "context_size": 1024,
        },
        {
            "model_name": "gpt2",
            "dataset_path": "Skylion007/openwebtext",
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

    # check the buffer norm
    if cfg.normalize_activations == "expected_average_only_in":
        assert torch.allclose(
            buffer.norm(dim=-1),
            np.sqrt(store.d_in) * torch.ones_like(buffer.norm(dim=-1)),
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


def test_activations_store__get_activations__gives_same_results_with_hf_model_and_tlens_model():
    hf_model = load_model(
        model_class_name="AutoModelForCausalLM",
        model_name="gpt2",
        device="cpu",
    )
    tlens_model = HookedTransformer.from_pretrained_no_processing("gpt2", device="cpu")
    dataset = Dataset.from_list(
        [
            {"text": "hello world"},
        ]
        * 100
    )

    cfg = build_sae_cfg(hook_name="blocks.4.hook_resid_post", hook_layer=4, d_in=768)
    store_tlens = ActivationsStore.from_config(
        tlens_model, cfg, override_dataset=dataset
    )
    batch_tlens = store_tlens.get_batch_tokens()
    activations_tlens = store_tlens.get_activations(batch_tlens)

    cfg = build_sae_cfg(hook_name="transformer.h.4", hook_layer=4, d_in=768)
    store_hf = ActivationsStore.from_config(hf_model, cfg, override_dataset=dataset)
    batch_hf = store_hf.get_batch_tokens()
    activations_hf = store_hf.get_activations(batch_hf)

    assert torch.allclose(activations_hf, activations_tlens, atol=1e-3)


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


def test_activations_store___iterate_tokenized_sequences__works_with_huggingface_models():
    hf_model = load_model(
        model_class_name="AutoModelForCausalLM",
        model_name="gpt2",
        device="cpu",
    )
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
        hf_model, cfg, override_dataset=dataset
    )
    for toks in activation_store._iterate_tokenized_sequences():
        assert toks.shape == (5,)


# We expect the code to work for context_size being less than or equal to the
# length of the dataset
@pytest.mark.parametrize(
    "context_size, expected_error",
    [(5, RuntimeWarning), (10, None), (15, ValueError)],
)
def test_activations_store__errors_on_context_size_mismatch(
    ts_model: HookedTransformer, context_size: int, expected_error: Optional[ValueError]
):
    tokenizer = ts_model.tokenizer
    assert tokenizer is not None
    cfg = build_sae_cfg(prepend_bos=True, context_size=context_size)
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

    # This context_size should raise an error or a warning if it mismatches the dataset size
    if expected_error is ValueError:
        with pytest.raises(expected_error):
            ActivationsStore.from_config(
                ts_model, cfg, override_dataset=tokenized_dataset
            )
    elif expected_error is RuntimeWarning:
        # If the context_size is smaller than the dataset size we should output a RuntimeWarning
        with pytest.warns(expected_error):
            ActivationsStore.from_config(
                ts_model, cfg, override_dataset=tokenized_dataset
            )
    else:
        # If the context_size is equal to the dataset size the function should pass
        ActivationsStore.from_config(ts_model, cfg, override_dataset=tokenized_dataset)


def test_activations_store__errors_on_negative_context_size():
    with pytest.raises(ValueError):
        # We should raise an error when the context_size is negative
        build_sae_cfg(prepend_bos=True, context_size=-1)


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


def test_validate_pretokenized_dataset_tokenizer_errors_if_the_tokenizer_doesnt_match_the_model():
    ds_path = "chanind/openwebtext-gpt2"
    model_tokenizer = HookedTransformer.from_pretrained("opt-125m").tokenizer
    assert model_tokenizer is not None
    with pytest.raises(ValueError):
        validate_pretokenized_dataset_tokenizer(ds_path, model_tokenizer)


def test_validate_pretokenized_dataset_tokenizer_runs_successfully_if_tokenizers_match(
    ts_model: HookedTransformer,
):
    ds_path = "chanind/openwebtext-gpt2"
    model_tokenizer = ts_model.tokenizer
    assert model_tokenizer is not None
    validate_pretokenized_dataset_tokenizer(ds_path, model_tokenizer)


def test_validate_pretokenized_dataset_tokenizer_does_nothing_if_the_dataset_is_not_created_by_sae_lens(
    ts_model: HookedTransformer,
):
    ds_path = "apollo-research/monology-pile-uncopyrighted-tokenizer-gpt2"
    model_tokenizer = ts_model.tokenizer
    assert model_tokenizer is not None
    validate_pretokenized_dataset_tokenizer(ds_path, model_tokenizer)


def test_validate_pretokenized_dataset_tokenizer_does_nothing_if_the_dataset_path_doesnt_exist(
    ts_model: HookedTransformer,
):
    ds_path = "blah/nonsense-1234"
    model_tokenizer = ts_model.tokenizer
    assert model_tokenizer is not None
    validate_pretokenized_dataset_tokenizer(ds_path, model_tokenizer)


def test_activations_store_respects_position_offsets(ts_model: HookedTransformer):
    cfg = build_sae_cfg(
        context_size=10,
        seqpos_slice=(2, 8),  # Only consider positions 2 to 7 (inclusive)
    )
    dataset = Dataset.from_list(
        [
            {"text": "This is a test sentence for slicing."},
        ]
        * 100
    )

    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )

    batch = activation_store.get_batch_tokens(1)
    activations = activation_store.get_activations(batch)

    assert batch.shape == (1, 10)  # Full context size
    assert activations.shape == (1, 6, 1, cfg.d_in)  # Only 6 positions (2 to 7)


@pytest.mark.parametrize(
    "params",
    [
        {
            "sae_kwargs": {
                "normalize_activations": "none",
            },
            "should_save": False,
        },
        {
            "sae_kwargs": {
                "normalize_activations": "expected_average_only_in",
            },
            "should_save": True,
        },
    ],
)
def test_activations_store_save_with_norm_scaling_factor(
    ts_model: HookedTransformer, params: dict[str, Any]
):
    cfg = build_sae_cfg(**params["sae_kwargs"])
    activation_store = ActivationsStore.from_config(ts_model, cfg)
    activation_store.set_norm_scaling_factor_if_needed()
    if params["sae_kwargs"]["normalize_activations"] == "expected_average_only_in":
        assert activation_store.estimated_norm_scaling_factor is not None
    with tempfile.NamedTemporaryFile() as temp_file:
        activation_store.save(temp_file.name)
        assert os.path.exists(temp_file.name)
        state_dict = load_file(temp_file.name)
        assert isinstance(state_dict, dict)
        if params["should_save"]:
            assert "estimated_norm_scaling_factor" in state_dict
            estimated_norm_scaling_factor = state_dict["estimated_norm_scaling_factor"]
            assert estimated_norm_scaling_factor.shape == ()
            assert (
                estimated_norm_scaling_factor.item()
                == activation_store.estimated_norm_scaling_factor
            )
        else:
            assert "estimated_norm_scaling_factor" not in state_dict
