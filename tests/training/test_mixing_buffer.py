import pytest
import torch

from sae_lens.training.mixing_buffer import mixing_buffer
from tests.helpers import assert_not_close


def test_mixing_buffer_yields_batches_of_correct_size_despite_loader_size_fluctuations():
    # Create a simple activations loader that yields 2 batches
    batch_size = 4
    buffer_size = 16
    d_in = 8

    # total number of activations is 16 + 3 + 16 - 1 = 34
    # so we should get 34 // 4 = 8 batches
    activations = [
        torch.randn(buffer_size + 3, d_in),
        torch.randn(buffer_size - 1, d_in),
    ]

    # Get batches from mixing buffer
    batches = list(
        mixing_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            activations_loader=iter(activations),
        )
    )

    assert len(batches) == 8
    for batch in batches:
        assert batch.shape == (batch_size, d_in)


def test_mixing_buffer_mixes_activations():
    buffer_size = 100
    batch_size = 50
    activations = [torch.arange(30), torch.arange(30, 60), torch.arange(60, 120)]

    buffer = mixing_buffer(
        buffer_size=buffer_size,
        batch_size=batch_size,
        activations_loader=iter(activations),
    )

    batch = next(buffer)
    assert batch.shape == (50,)
    assert_not_close(batch, torch.arange(50))
    assert len(torch.unique(batch)) == len(batch)  # All elements are unique


def test_mixing_buffer_empty_loader():
    buffer = mixing_buffer(buffer_size=16, batch_size=4, activations_loader=iter([]))

    # Should not yield any batches
    assert not list(buffer)


def test_mixing_buffer_error_on_small_buffer():
    # Test when buffer size is smaller than batch size
    batch_size = 8
    buffer_size = 4  # Too small

    activations = [torch.randn(batch_size, 4)]

    with pytest.raises(ValueError):
        buffer = mixing_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            activations_loader=iter(activations),
        )
        next(buffer)


def test_mixing_buffer_maintains_dtype():
    # Test that dtype is preserved
    batch_size = 4
    buffer_size = 16
    dtype = torch.float64

    activations = [
        torch.randn(batch_size, 8, dtype=dtype),
        torch.randn(batch_size, 8, dtype=dtype),
    ]

    batches = list(
        mixing_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            activations_loader=iter(activations),
        )
    )

    for batch in batches:
        assert batch.dtype == dtype
