from copy import deepcopy
from pathlib import Path

import pytest
import torch
from sparsify import SparseCoder, SparseCoderConfig

from sae_lens.sae import SAE
from sae_lens.training.training_sae import (
    TrainingSAE,
    TrainingSAEConfig,
    _calculate_topk_aux_acts,
)
from tests.helpers import build_sae_cfg


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


def test_TrainingSAE_topk_aux_loss_matches_unnormalized_sparsify_implementation():
    d_in = 128
    d_sae = 192
    cfg = build_sae_cfg(
        d_in=d_in,
        d_sae=d_sae,
        architecture="topk",
        activation_fn_kwargs={"k": 26},
        normalize_sae_decoder=False,
    )

    sae = TrainingSAE(TrainingSAEConfig.from_sae_runner_config(cfg))
    sparse_coder_sae = SparseCoder(
        d_in=d_in, cfg=SparseCoderConfig(num_latents=d_sae, k=26)
    )

    with torch.no_grad():
        # increase b_enc so all features are likely above 0
        # sparsify includes a relu() in their pre_acts, but
        # this is not something we need to try to replicate.
        sae.b_enc.data = sae.b_enc + 100.0
        # make sure all params are the same
        sparse_coder_sae.encoder.weight.data = sae.W_enc.T
        sparse_coder_sae.encoder.bias.data = sae.b_enc
        sparse_coder_sae.b_dec.data = sae.b_dec
        sparse_coder_sae.W_dec.data = sae.W_dec  # type: ignore

    dead_neuron_mask = torch.randn(d_sae) > 0.1
    input_acts = torch.randn(200, d_in)
    input_var = (input_acts - input_acts.mean(0)).pow(2).sum()

    sae_out = sae.training_forward_pass(
        sae_in=input_acts,
        current_l1_coefficient=0.0,
        dead_neuron_mask=dead_neuron_mask,
    )
    comparison_sae_out = sparse_coder_sae.forward(
        input_acts, dead_mask=dead_neuron_mask
    )
    comparison_aux_loss = comparison_sae_out.auxk_loss.detach().item()

    normalization = input_var / input_acts.shape[0]
    raw_aux_loss = sae_out.losses["auxiliary_reconstruction_loss"].item()  # type: ignore
    norm_aux_loss = raw_aux_loss / normalization
    assert norm_aux_loss == pytest.approx(comparison_aux_loss, abs=1e-4)


def test_TrainingSAE_calculate_topk_aux_loss():
    # Create a small test SAE with d_sae=3, d_in=4
    cfg = build_sae_cfg(
        d_in=4,
        d_sae=3,
        architecture="topk",
        normalize_sae_decoder=False,
    )
    sae = TrainingSAE(TrainingSAEConfig.from_sae_runner_config(cfg))

    # Set up test inputs
    hidden_pre = torch.tensor(
        [[1.0, -2.0, 3.0], [1.0, 0.0, -3.0]]  # batch size 2
    )
    sae.W_dec.data = torch.tensor(2 * torch.ones((3, 4)))
    sae.b_dec.data = torch.tensor(torch.zeros(4))

    sae_out = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0]])
    sae_in = torch.tensor([[2.0, 1.0, 3.0, 4.0], [5.0, 4.0, 6.0, 7.0]])
    # Mark neurons 1 and 2 as dead
    dead_neuron_mask = torch.tensor([False, True, True])

    # Calculate loss
    loss = sae.calculate_topk_aux_loss(
        sae_in=sae_in,
        hidden_pre=hidden_pre,
        sae_out=sae_out,
        dead_neuron_mask=dead_neuron_mask,
    )

    # The loss should:
    # 1. Select top k_aux=2 (half of d_in=4) dead neurons
    # 2. Decode their activations (should be 2x the activations of the dead neurons)
    # For batch 1: dead neurons are [-2.0, 3.0] -> activations [-4.0, 6.0] -> sum 2.0 for each output dim
    # For batch 2: dead neurons are [0.0, -3.0] -> activations [0.0, -6.0] -> sum -6.0 for each output dim
    # Residuals are: [1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]
    # errors are: [1.0, 3.0, 2.0, 2.0], [-7., -5., -6., -6.]
    # Squared errors are: [1.0, 9.0, 4.0, 4.0], [49.0, 25.0, 36.0, 36.0]
    # Sum over features: 18.0, 146.0
    # Mean over batch: 82.0

    assert loss == 82


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


@torch.no_grad()
def test_TrainingSAE_fold_w_dec_norm_jumprelu():
    cfg = build_sae_cfg(architecture="jumprelu")
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    # make sure all parameters are not 0s
    for param in sae.parameters():
        param.data = torch.rand_like(param)

    assert sae.W_dec.norm(dim=-1).mean().item() != pytest.approx(1.0, abs=1e-6)
    sae2 = deepcopy(sae)
    sae2.fold_W_dec_norm()

    # fold_W_dec_norm should normalize W_dec to have unit norm.
    assert sae2.W_dec.norm(dim=-1).mean().item() == pytest.approx(1.0, abs=1e-6)

    W_dec_norms = sae.W_dec.norm(dim=-1).unsqueeze(1)
    assert torch.allclose(sae2.b_enc, sae.b_enc * W_dec_norms.squeeze())
    assert torch.allclose(sae2.threshold, sae.threshold * W_dec_norms.squeeze())

    # we expect activations of features to differ by W_dec norm weights.
    activations = torch.randn(10, 4, cfg.d_in, device=cfg.device)
    feature_activations_1 = sae.encode(activations)
    feature_activations_2 = sae2.encode(activations)

    assert torch.allclose(
        feature_activations_1.nonzero(),
        feature_activations_2.nonzero(),
    )

    expected_feature_activations_2 = feature_activations_1 * sae.W_dec.norm(dim=-1)
    torch.testing.assert_close(feature_activations_2, expected_feature_activations_2)

    sae_out_1 = sae.decode(feature_activations_1)
    sae_out_2 = sae2.decode(feature_activations_2)

    # but actual outputs should be the same
    torch.testing.assert_close(sae_out_1, sae_out_2)
