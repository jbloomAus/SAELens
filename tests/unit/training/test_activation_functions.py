import pytest
import torch

from sae_lens.sae import get_activation_fn


def test_get_activation_fn_tanh_relu():
    tanh_relu = get_activation_fn("tanh-relu")
    assert tanh_relu(torch.tensor([-1.0, 0.0])).tolist() == [0.0, 0.0]
    assert tanh_relu(torch.tensor(1e10)).item() == pytest.approx(1.0)


def test_get_activation_fn_relu():
    relu = get_activation_fn("relu")
    assert relu(torch.tensor([-1.0, 0.0])).tolist() == [0.0, 0.0]
    assert relu(torch.tensor(999.9)).item() == pytest.approx(999.9)


def test_get_activation_fn_error_for_unknown_values():
    with pytest.raises(ValueError):
        get_activation_fn("unknown")


def test_get_activation_fn_topk_32():
    topk = get_activation_fn("topk", k=32)
    example_activations = torch.randn(10, 4, 512)
    post_activations = topk(example_activations)
    assert post_activations.shape == example_activations.shape
    assert post_activations.nonzero().shape[0] == 32 * 10 * 4

    expected_activations = example_activations.flatten().topk(32).values
    assert torch.allclose(
        post_activations.flatten().sort(dim=0, descending=True).values[:32],
        expected_activations,
    )


def test_get_activation_fn_topk_16():
    topk = get_activation_fn("topk", k=16)
    example_activations = torch.randn(10, 4, 512)
    post_activations = topk(example_activations)
    assert post_activations.shape == example_activations.shape
    assert post_activations.nonzero().shape[0] == 16 * 10 * 4

    expected_activations = example_activations.flatten().topk(16).values
    assert torch.allclose(
        post_activations.flatten().sort(dim=0, descending=True).values[:16],
        expected_activations,
    )
