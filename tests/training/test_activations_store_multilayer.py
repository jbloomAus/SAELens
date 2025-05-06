"""Tests for ActivationsStore with multiple layer support."""

import pytest
import torch
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import ActivationsStore
from tests.helpers import build_sae_cfg, build_multilayer_sae_cfg, load_model_cached


def test_activations_store_init_with_multiple_layers(ts_model: HookedTransformer):
    """Test initialization with a list of layers instead of a single layer."""
    # Initialize with multiple layers
    cfg = build_multilayer_sae_cfg(
        hook_name_template="blocks.{layer}.hook_resid_pre",
        hook_layers=[0, 1, 2]
    )

    activation_store = ActivationsStore.from_config(ts_model, cfg)

    assert activation_store.hook_names == [
        "blocks.0.hook_resid_pre",
        "blocks.1.hook_resid_pre",
        "blocks.2.hook_resid_pre",
    ]

    cfg_single = build_multilayer_sae_cfg(
        hook_name_template="blocks.{layer}.hook_resid_pre",
        hook_layers=[1]
    )

    single_layer_store = ActivationsStore.from_config(ts_model, cfg_single)
    assert single_layer_store.hook_names == [
        "blocks.1.hook_resid_pre",
    ]


def test_activations_store_get_activations_multiple_layers(ts_model: HookedTransformer):
    """Test that get_activations collects activations from all specified layers."""
    # Setup with multiple layers
    cfg = build_multilayer_sae_cfg(
        hook_name_template="blocks.{layer}.hook_resid_pre",
        hook_layers=[0, 1, 2],
        context_size=5
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 10)
    activation_store = ActivationsStore.from_config(ts_model, cfg, override_dataset=dataset)

    # Get a batch of tokens and activations
    batch_tokens = activation_store.get_batch_tokens()
    activations = activation_store.get_activations(batch_tokens)

    # Check shape: [batch_size, context_size, num_layers, d_in]
    assert activations.shape == (
        cfg.store_batch_size_prompts,
        cfg.context_size,
        len(cfg.hook_names),
        cfg.d_in
    )

    # Verify that layers are in the correct order
    # Run with cache directly to compare against
    _, cache = ts_model.run_with_cache(
        batch_tokens,
        names_filter=[f"blocks.{i}.hook_resid_pre" for i in [0, 1, 2]]
    )

    for i, layer in enumerate([0, 1, 2]):
        hook_name = f"blocks.{layer}.hook_resid_pre"
        # Compare the activations for this layer with what we got from run_with_cache
        assert torch.allclose(
            activations[:, :, i, :],
            cache[hook_name],
            atol=1e-5
        )


def test_activations_store_get_buffer_multiple_layers(ts_model: HookedTransformer):
    """Test buffer handling with multiple layers."""
    # Setup with multiple layers
    cfg = build_multilayer_sae_cfg(
        hook_name_template="blocks.{layer}.hook_resid_pre",
        hook_layers=[0, 1, 2],
        context_size=5
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 20)
    activation_store = ActivationsStore.from_config(ts_model, cfg, override_dataset=dataset)

    # Get buffer with 2 batches
    buffer_activations, buffer_tokens = activation_store.get_buffer(n_batches_in_buffer=2)

    # Check shape: [(batch_size * context_size * n_batches), num_layers, d_in]
    expected_size = cfg.store_batch_size_prompts * cfg.context_size * 2
    assert buffer_activations.shape == (expected_size, len(cfg.hook_names), cfg.d_in)
    assert buffer_tokens.shape == (expected_size,)


def test_activations_store_next_batch_multiple_layers(ts_model: HookedTransformer):
    """Test that next_batch returns correct batch shape with multiple layers."""
    # Setup with multiple layers
    cfg = build_multilayer_sae_cfg(
        hook_name_template="blocks.{layer}.hook_resid_pre",
        hook_layers=[0, 1, 2],
        context_size=5,
        train_batch_size_tokens=10
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 20)
    activation_store = ActivationsStore.from_config(ts_model, cfg, override_dataset=dataset)

    batch = activation_store.next_batch()
    assert batch.shape == (10, len(cfg.hook_names), activation_store.d_in)


def test_activations_store_normalization_multiple_layers(ts_model: HookedTransformer):
    """Test normalization when using multiple layers."""
    cfg = build_multilayer_sae_cfg(
        hook_name_template="blocks.{layer}.hook_resid_pre",
        hook_layers=[0, 1, 2],
        normalize_activations="expected_average_only_in",
        context_size=5
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 20)
    activation_store = ActivationsStore.from_config(ts_model, cfg, override_dataset=dataset)
    activation_store.set_norm_scaling_factor_if_needed()

    batch = activation_store.next_batch()

    avg_norm = batch.norm(dim=-1).mean(dim=1)
    expected_norm = torch.full_like(avg_norm, cfg.d_in ** 0.5)
    torch.testing.assert_close(avg_norm, expected_norm, atol=1.0, rtol=0.1)


def test_backward_compatibility_single_layer(ts_model: HookedTransformer):
    """Test that single layer behavior is unchanged with the multi-layer support."""
    cfg_single = build_sae_cfg(
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        context_size=5
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 10)
    single_store = ActivationsStore.from_config(ts_model, cfg_single, override_dataset=dataset)

    cfg_multi = build_multilayer_sae_cfg(
        hook_name_template="blocks.{layer}.hook_resid_pre",
        hook_layers=[0],
        context_size=5
    )
    multi_store = ActivationsStore.from_config(ts_model, cfg_multi, override_dataset=dataset)

    # Get tokens and activations from both
    batch_tokens_single = single_store.get_batch_tokens()
    activations_single = single_store.get_activations(batch_tokens_single)

    batch_tokens_multi = multi_store.get_batch_tokens()
    activations_multi = multi_store.get_activations(batch_tokens_multi)

    torch.testing.assert_close(batch_tokens_single, batch_tokens_multi)
    torch.testing.assert_close(activations_single, activations_multi)
