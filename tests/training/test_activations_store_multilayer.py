"""Tests for ActivationsStore with multiple layer support."""

import pytest
import torch
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import ActivationsStore
from tests.helpers import build_sae_cfg, load_model_cached


def test_activations_store_init_with_multiple_layers(ts_model: HookedTransformer):
    """Test initialization with a list of layers instead of a single layer."""
    # Initialize with multiple layers
    cfg = build_sae_cfg(
        hook_name="blocks.{layer}.hook_resid_pre",
        hook_layers=[0, 1, 2]
    )

    activation_store = ActivationsStore.from_config(ts_model, cfg)

    # Check that the hook layers are correctly stored
    assert activation_store.hook_layers == [0, 1, 2]

    # Verify backward compatibility - a single hook_layer should be converted to a list
    cfg_single = build_sae_cfg(
        hook_name="blocks.{layer}.hook_resid_pre",
        hook_layer=1
    )

    single_layer_store = ActivationsStore.from_config(ts_model, cfg_single)
    assert single_layer_store.hook_layers == [1]


def test_activations_store_get_activations_multiple_layers(ts_model: HookedTransformer):
    """Test that get_activations collects activations from all specified layers."""
    # Setup with multiple layers
    cfg = build_sae_cfg(
        hook_name="blocks.{layer}.hook_resid_pre",
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
        activation_store.store_batch_size_prompts,
        activation_store.context_size,
        len(activation_store.hook_layers),
        activation_store.d_in
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
    cfg = build_sae_cfg(
        hook_name="blocks.{layer}.hook_resid_pre",
        hook_layers=[0, 1, 2],
        context_size=5
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 20)
    activation_store = ActivationsStore.from_config(ts_model, cfg, override_dataset=dataset)

    # Get buffer with 2 batches
    buffer_activations, buffer_tokens = activation_store.get_buffer(n_batches_in_buffer=2)

    # Check shape: [(batch_size * context_size * n_batches), num_layers, d_in]
    expected_size = activation_store.store_batch_size_prompts * activation_store.context_size * 2
    assert buffer_activations.shape == (expected_size, len(activation_store.hook_layers), activation_store.d_in)
    assert buffer_tokens.shape == (expected_size,)


def test_activations_store_next_batch_multiple_layers(ts_model: HookedTransformer):
    """Test that next_batch returns correct batch shape with multiple layers."""
    # Setup with multiple layers
    cfg = build_sae_cfg(
        hook_name="blocks.{layer}.hook_resid_pre",
        hook_layers=[0, 1, 2],
        context_size=5,
        train_batch_size_tokens=10
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 20)
    activation_store = ActivationsStore.from_config(ts_model, cfg, override_dataset=dataset)

    batch = activation_store.next_batch()
    assert batch.shape == (10, len(cfg.hook_layers), activation_store.d_in)

@pytest.mark.skip("TODO(mkbehr): does activation need to be handled differently?")
def test_activations_store_normalization_multiple_layers(ts_model: HookedTransformer):
    """Test normalization when using multiple layers."""
    # Setup with normalization and multiple layers
    cfg = build_sae_cfg(
        hook_name="blocks.{layer}.hook_resid_pre",
        hook_layers=[0, 1, 2],
        normalize_activations="expected_average_only_in",
        context_size=5
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 20)
    activation_store = ActivationsStore.from_config(ts_model, cfg, override_dataset=dataset)
    activation_store.set_norm_scaling_factor_if_needed()

    # Get a batch with normalized activations
    batch = activation_store.next_batch()

    # Check that the activations have been properly normalized
    # The norm should be approximately sqrt(d_in) for each layer
    for layer_idx in range(len(activation_store.hook_layers)):
        layer_activations = batch[:, layer_idx, :]
        # Check if average norm is approximately as expected (allowing for some variance)
        avg_norm = layer_activations.norm(dim=-1).mean()
        expected_norm = (activation_store.d_in ** 0.5)
        assert avg_norm.item() == pytest.approx(expected_norm, abs=2.0)


def test_backward_compatibility_single_layer(ts_model: HookedTransformer):
    """Test that single layer behavior is unchanged with the multi-layer support."""
    # Create a store with single layer (old behavior)
    cfg_single = build_sae_cfg(
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        context_size=5
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 10)
    single_store = ActivationsStore.from_config(ts_model, cfg_single, override_dataset=dataset)

    # Create a store with single layer (new behavior)
    cfg_multi = build_sae_cfg(
        hook_name="blocks.{layer}.hook_resid_pre",
        hook_layers=[0],
        context_size=5
    )
    multi_store = ActivationsStore.from_config(ts_model, cfg_multi, override_dataset=dataset)

    # Get tokens and activations from both
    batch_tokens_single = single_store.get_batch_tokens()
    activations_single = single_store.get_activations(batch_tokens_single)

    batch_tokens_multi = multi_store.get_batch_tokens()
    activations_multi = multi_store.get_activations(batch_tokens_multi)

    # Check that activations have the same shape and values
    assert activations_single.shape == activations_multi.shape
    # Run with deterministic seed to ensure tokens are the same
    if torch.allclose(batch_tokens_single, batch_tokens_multi):
        assert torch.allclose(activations_single, activations_multi, atol=1e-5)


def test_mixed_hook_formats(ts_model: HookedTransformer):
    """Test that both formatted and non-formatted hook names work with multiple layers."""
    # Test with formatted hook name (with {layer})
    cfg_formatted = build_sae_cfg(
        hook_name="blocks.{layer}.hook_resid_pre",
        hook_layers=[0, 1],
        context_size=5
    )

    # Test with non-formatted hook name
    cfg_non_formatted = build_sae_cfg(
        hook_name="blocks.0.hook_resid_pre",  # Specific to layer 0
        hook_layers=[0],  # Only layer 0 works with this hook
        context_size=5
    )

    dataset = Dataset.from_list([{"text": "hello world"}] * 10)

    # Both should initialize without errors
    store_formatted = ActivationsStore.from_config(
        ts_model, cfg_formatted, override_dataset=dataset
    )
    store_non_formatted = ActivationsStore.from_config(
        ts_model, cfg_non_formatted, override_dataset=dataset
    )

    # Both should be able to get activations
    activations_formatted = store_formatted.get_activations(
        store_formatted.get_batch_tokens()
    )
    activations_non_formatted = store_non_formatted.get_activations(
        store_non_formatted.get_batch_tokens()
    )

    # Check shapes
    assert activations_formatted.shape == (
        store_formatted.store_batch_size_prompts,
        store_formatted.context_size,
        len(store_formatted.hook_layers),
        store_formatted.d_in
    )

    assert activations_non_formatted.shape == (
        store_non_formatted.store_batch_size_prompts,
        store_non_formatted.context_size,
        len(store_non_formatted.hook_layers),
        store_non_formatted.d_in
    )
