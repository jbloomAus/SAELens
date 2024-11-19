from pathlib import Path

import pytest
import torch

from sae_lens.sae import SAE
from sae_lens.training.training_sae import (
    TrainingSAE,
    TrainingSAEConfig,
    _calculate_topk_aux_acts,
)
from tests.unit.helpers import build_sae_cfg


@pytest.mark.parametrize("scale_sparsity_penalty_by_decoder_norm", [True, False])
def test_TrainingSAE_training_forward_pass_can_scale_sparsity_penalty_by_decoder_norm(
    scale_sparsity_penalty_by_decoder_norm: bool,
):
    cfg = build_sae_cfg(
        d_in=3,
        d_sae=5,
        scale_sparsity_penalty_by_decoder_norm=scale_sparsity_penalty_by_decoder_norm,
        normalize_sae_decoder=False,
    )
    training_sae = TrainingSAE(TrainingSAEConfig.from_sae_runner_config(cfg))
    x = torch.randn(32, 3)
    train_step_output = training_sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=2.0,
    )
    feature_acts = train_step_output.feature_acts
    decoder_norm = training_sae.W_dec.norm(dim=1)
    # double-check decoder norm is not all ones, or this test is pointless
    assert not torch.allclose(decoder_norm, torch.ones_like(decoder_norm), atol=1e-2)
    scaled_feature_acts = feature_acts * decoder_norm

    if scale_sparsity_penalty_by_decoder_norm:
        assert (
            pytest.approx(train_step_output.losses["l1_loss"].detach().item())  # type: ignore
            == 2.0 * scaled_feature_acts.norm(p=1, dim=1).mean().detach().item()
        )
    else:
        assert (
            pytest.approx(train_step_output.losses["l1_loss"].detach().item())  # type: ignore
            == 2.0 * feature_acts.norm(p=1, dim=1).mean().detach().item()
        )


def test_calculate_topk_aux_acts():
    # Create test inputs
    k_aux = 3
    hidden_pre = torch.tensor(
        [
            [1.0, 2.0, -3.0, 4.0, -5.0, 6.0],
            [-1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1],
        ]
    )

    # Create dead neuron mask where neurons 1,3,5 are dead
    dead_neuron_mask = torch.tensor([False, True, False, True, False, True])

    # Calculate expected result
    # For each row, should select top k_aux=3 values from dead neurons (indices 1,3,5)
    # and zero out all other values
    expected = torch.zeros_like(hidden_pre)
    expected[0, [1, 3, 5]] = torch.tensor([2.0, 4.0, 6.0])
    expected[1, [1, 3, 5]] = torch.tensor([-2.0, -4.0, -6.0])
    expected[2, [1, 3, 5]] = torch.tensor([0.2, 0.4, 0.6])
    expected[3, [1, 3, 5]] = torch.tensor([-0.5, -0.3, -0.1])

    result = _calculate_topk_aux_acts(k_aux, hidden_pre, dead_neuron_mask)

    assert torch.allclose(result, expected)


def test_calculate_topk_aux_acts_k_less_than_dead():
    # Create test inputs with k_aux less than number of dead neurons
    k_aux = 1  # Only select top 1 dead neuron
    hidden_pre = torch.tensor(
        [
            [1.0, 2.0, -3.0, 4.0],  # 2 items in batch
            [-1.0, -2.0, 3.0, -4.0],
        ]
    )

    # Create dead neuron mask where neurons 1,3 are dead (2 dead neurons)
    dead_neuron_mask = torch.tensor([False, True, False, True])

    # Calculate expected result
    # For each row, should select only top k_aux=1 value from dead neurons (indices 1,3)
    # and zero out all other values
    expected = torch.zeros_like(hidden_pre)
    expected[0, 3] = 4.0  # Only highest value among dead neurons for first item
    expected[1, 1] = -2.0  # Only highest value among dead neurons for second item

    result = _calculate_topk_aux_acts(k_aux, hidden_pre, dead_neuron_mask)

    assert torch.allclose(result, expected)


def test_TrainingSAE_calculate_topk_aux_loss():
    # Create a small test SAE with d_sae=4, d_in=3
    cfg = build_sae_cfg(
        d_in=3,
        d_sae=4,
        architecture="topk",
        normalize_sae_decoder=False,
    )

    sae = TrainingSAE(TrainingSAEConfig.from_sae_runner_config(cfg))

    # Set up test inputs
    hidden_pre = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [1.0, 0.0, -3.0, -4.0]]  # batch size 2
    )
    sae.W_dec.data = torch.tensor(2 * torch.ones((4, 3)))
    sae.b_dec.data = torch.tensor(torch.zeros(3))

    sae_out = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    sae_in = torch.tensor([[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]])
    # Mark neurons 1 and 3 as dead
    dead_neuron_mask = torch.tensor([False, True, False, True])

    # Calculate loss
    loss = sae.calculate_topk_aux_loss(
        sae_in=sae_in,
        hidden_pre=hidden_pre,
        sae_out=sae_out,
        dead_neuron_mask=dead_neuron_mask,
    )

    # The loss should:
    # 1. Select top k_aux=2 (half of d_sae) dead neurons
    # 2. Decode their activations (should be 2x the sum of the activations of the dead neurons)
    # thus, (-12, -12, -12), (-8, -8, -8)
    # and the residual is (1, -1, 0), (1, -1, 0)
    # Thus, squared errors are (169, 121, 144), (81, 49, 64)
    # and the sums are (434, 194)
    # and the mean of these is 314

    assert loss == 314


def test_TrainingSAE_forward_includes_topk_loss_with_topk_architecture():
    cfg = build_sae_cfg(
        d_in=3,
        d_sae=4,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
        normalize_sae_decoder=False,
    )
    sae = TrainingSAE(TrainingSAEConfig.from_sae_runner_config(cfg))
    x = torch.randn(32, 3)
    train_step_output = sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=2.0,
        dead_neuron_mask=None,
    )
    assert "auxiliary_reconstruction_loss" in train_step_output.losses
    assert train_step_output.losses["auxiliary_reconstruction_loss"] == 0.0


def test_TrainingSAE_forward_includes_topk_loss_is_nonzero_if_dead_neurons_present():
    cfg = build_sae_cfg(
        d_in=3,
        d_sae=4,
        architecture="topk",
        activation_fn_kwargs={"k": 2},
        normalize_sae_decoder=False,
    )
    sae = TrainingSAE(TrainingSAEConfig.from_sae_runner_config(cfg))
    x = torch.randn(32, 3)
    train_step_output = sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=2.0,
        dead_neuron_mask=torch.tensor([False, True, False, True]),
    )
    assert "auxiliary_reconstruction_loss" in train_step_output.losses
    assert train_step_output.losses["auxiliary_reconstruction_loss"] > 0.0


@pytest.mark.parametrize("architecture", ["standard", "gated", "jumprelu", "topk"])
def test_TrainingSAE_encode_returns_same_value_as_encode_with_hidden_pre(
    architecture: str,
):
    cfg = build_sae_cfg(architecture=architecture)
    sae = TrainingSAE(TrainingSAEConfig.from_sae_runner_config(cfg))
    x = torch.randn(32, cfg.d_in)
    encode_out = sae.encode(x)
    encode_with_hidden_pre_out = sae.encode_with_hidden_pre_fn(x)[0]
    assert torch.allclose(encode_out, encode_with_hidden_pre_out)


def test_TrainingSAE_initializes_only_with_log_threshold_if_jumprelu():
    cfg = build_sae_cfg(architecture="jumprelu", jumprelu_init_threshold=0.01)
    sae = TrainingSAE(TrainingSAEConfig.from_sae_runner_config(cfg))
    param_names = dict(sae.named_parameters()).keys()
    assert "log_threshold" in param_names
    assert "threshold" not in param_names
    assert torch.allclose(
        sae.threshold,
        torch.ones_like(sae.log_threshold.data) * cfg.jumprelu_init_threshold,
    )


def test_TrainingSAE_jumprelu_save_and_load(tmp_path: Path):
    cfg = build_sae_cfg(architecture="jumprelu")
    training_sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    training_sae.save_model(str(tmp_path))

    loaded_training_sae = TrainingSAE.load_from_pretrained(str(tmp_path))
    loaded_sae = SAE.load_from_pretrained(str(tmp_path))

    assert training_sae.cfg.to_dict() == loaded_training_sae.cfg.to_dict()
    for param_name, param in training_sae.named_parameters():
        assert torch.allclose(param, loaded_training_sae.state_dict()[param_name])

    test_input = torch.randn(32, cfg.d_in)
    training_sae_out = training_sae.encode_with_hidden_pre_fn(test_input)[0]
    loaded_training_sae_out = loaded_training_sae.encode_with_hidden_pre_fn(test_input)[
        0
    ]
    loaded_sae_out = loaded_sae.encode(test_input)
    assert torch.allclose(training_sae_out, loaded_training_sae_out)
    assert torch.allclose(training_sae_out, loaded_sae_out)
