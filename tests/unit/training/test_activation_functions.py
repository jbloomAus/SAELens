import pytest
import torch

from sae_lens.training.activation_functions import get_activation_fn


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
