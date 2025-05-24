import os
from pathlib import Path

import pytest
import torch
from torch import nn

from sae_lens.saes.jumprelu_sae import JumpReLU, JumpReLUSAE, JumpReLUTrainingSAE
from sae_lens.saes.sae import SAE, TrainStepInput
from tests.helpers import (
    assert_close,
    assert_not_close,
    build_jumprelu_sae_cfg,
    build_jumprelu_sae_training_cfg,
)


def test_JumpReLUTrainingSAE_encoding():
    sae = JumpReLUTrainingSAE(build_jumprelu_sae_training_cfg())

    batch_size = 32
    d_in = sae.cfg.d_in
    d_sae = sae.cfg.d_sae

    x = torch.randn(batch_size, d_in)
    feature_acts, hidden_pre = sae.encode_with_hidden_pre(x)

    assert feature_acts.shape == (batch_size, d_sae)
    assert hidden_pre.shape == (batch_size, d_sae)

    # Check the JumpReLU thresholding
    sae_in = sae.process_sae_in(x)
    expected_hidden_pre = sae_in @ sae.W_enc + sae.b_enc
    threshold = torch.exp(sae.log_threshold)
    expected_feature_acts = JumpReLU.apply(
        expected_hidden_pre, threshold, sae.bandwidth
    )

    assert_close(feature_acts, expected_feature_acts, atol=1e-6)  # type: ignore


def test_JumpReLUTrainingSAE_training_forward_pass():
    sae = JumpReLUTrainingSAE(build_jumprelu_sae_training_cfg())

    batch_size = 32
    d_in = sae.cfg.d_in

    x = torch.randn(batch_size, d_in)
    train_step_output = sae.training_forward_pass(
        step_input=TrainStepInput(
            sae_in=x,
            coefficients={"l0": sae.cfg.l0_coefficient},
            dead_neuron_mask=None,
        ),
    )

    assert train_step_output.sae_out.shape == (batch_size, d_in)
    assert train_step_output.feature_acts.shape == (batch_size, sae.cfg.d_sae)
    assert (
        pytest.approx(train_step_output.loss.detach(), rel=1e-3)
        == (
            train_step_output.losses["mse_loss"] + train_step_output.losses["l0_loss"]
        ).item()  # type: ignore
    )

    expected_mse_loss = (
        (torch.pow((train_step_output.sae_out - x.float()), 2))
        .sum(dim=-1)
        .mean()
        .detach()
        .float()
    )

    assert (
        pytest.approx(train_step_output.losses["mse_loss"].item()) == expected_mse_loss  # type: ignore
    )


def test_JumpReLUSAE_initialization():
    cfg = build_jumprelu_sae_cfg(device="cpu")
    sae = JumpReLUSAE.from_dict(cfg.to_dict())
    assert isinstance(sae.W_enc, nn.Parameter)
    assert isinstance(sae.W_dec, nn.Parameter)
    assert isinstance(sae.b_enc, nn.Parameter)
    assert isinstance(sae.b_dec, nn.Parameter)
    assert isinstance(sae.threshold, nn.Parameter)

    assert sae.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sae.b_enc.shape == (cfg.d_sae,)
    assert sae.b_dec.shape == (cfg.d_in,)
    assert sae.threshold.shape == (cfg.d_sae,)

    # encoder/decoder should be initialized, everything else should be 0s
    assert_not_close(sae.W_enc, torch.zeros_like(sae.W_enc))
    assert_not_close(sae.W_dec, torch.zeros_like(sae.W_dec))
    assert_close(sae.b_dec, torch.zeros_like(sae.b_dec))
    assert_close(sae.b_enc, torch.zeros_like(sae.b_enc))
    assert_close(sae.threshold, torch.zeros_like(sae.threshold))


@pytest.mark.parametrize("use_error_term", [True, False])
def test_JumpReLUSAE_forward(use_error_term: bool):
    cfg = build_jumprelu_sae_cfg(d_in=2, d_sae=3)
    sae = JumpReLUSAE.from_dict(cfg.to_dict())
    sae.use_error_term = use_error_term
    sae.threshold.data = torch.tensor([1.0, 0.5, 0.25])
    sae.W_enc.data = torch.ones_like(sae.W_enc.data)
    sae.W_dec.data = torch.ones_like(sae.W_dec.data)
    sae.b_enc.data = torch.zeros_like(sae.b_enc.data)
    sae.b_dec.data = torch.zeros_like(sae.b_dec.data)

    sae_in = 0.3 * torch.ones(1, 2)
    expected_recons = torch.tensor([[1.2, 1.2]])
    # if we use error term, we should always get the same output as what we put in
    expected_output = sae_in if use_error_term else expected_recons
    out, cache = sae.run_with_cache(sae_in)
    assert_close(out, expected_output)
    assert_close(cache["hook_sae_input"], sae_in)
    assert_close(cache["hook_sae_output"], out)
    assert_close(cache["hook_sae_recons"], expected_recons)
    if use_error_term:
        assert_close(cache["hook_sae_error"], expected_output - expected_recons)

    assert_close(cache["hook_sae_acts_pre"], torch.tensor([[0.6, 0.6, 0.6]]))
    # the threshold of 1.0 should block the first latent from firing
    assert_close(cache["hook_sae_acts_post"], torch.tensor([[0.0, 0.6, 0.6]]))


def test_JumpReLUTrainingSAE_initialization():
    cfg = build_jumprelu_sae_training_cfg()
    sae = JumpReLUTrainingSAE.from_dict(cfg.to_dict())

    assert sae.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert isinstance(sae.log_threshold, torch.nn.Parameter)
    assert sae.log_threshold.shape == (cfg.d_sae,)
    assert sae.b_enc.shape == (cfg.d_sae,)
    assert sae.b_dec.shape == (cfg.d_in,)
    assert isinstance(sae.activation_fn, torch.nn.ReLU)
    assert sae.device == torch.device("cpu")
    assert sae.dtype == torch.float32

    # biases
    assert_close(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert_close(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)

    # check if the decoder weight norm is 0.1 by default
    assert_close(
        sae.W_dec.norm(dim=1),
        0.1 * torch.ones_like(sae.W_dec.norm(dim=1)),
        atol=1e-6,
    )

    #  Default currently should be tranpose initialization
    assert_close(sae.W_enc, sae.W_dec.T, atol=1e-6)


def test_JumpReLUTrainingSAE_save_and_load_inference_sae(tmp_path: Path) -> None:
    # Create a training SAE with specific parameter values
    cfg = build_jumprelu_sae_training_cfg(device="cpu")
    training_sae = JumpReLUTrainingSAE(cfg)

    # Set some known values for testing
    training_sae.W_enc.data = torch.randn_like(training_sae.W_enc.data)
    training_sae.W_dec.data = torch.randn_like(training_sae.W_dec.data)
    training_sae.b_enc.data = torch.randn_like(training_sae.b_enc.data)
    training_sae.b_dec.data = torch.randn_like(training_sae.b_dec.data)
    training_sae.log_threshold.data = torch.randn_like(training_sae.log_threshold.data)

    # Save original state for comparison
    original_W_enc = training_sae.W_enc.data.clone()
    original_W_dec = training_sae.W_dec.data.clone()
    original_b_enc = training_sae.b_enc.data.clone()
    original_b_dec = training_sae.b_dec.data.clone()
    original_threshold = training_sae.threshold.data.clone()  # exp(log_threshold)

    # Save as inference model
    model_path = str(tmp_path)
    training_sae.save_inference_model(model_path)

    assert os.path.exists(model_path)

    # Load as inference SAE
    inference_sae = SAE.load_from_disk(model_path, device="cpu")

    # Should be loaded as JumpReLUSAE
    assert isinstance(inference_sae, JumpReLUSAE)

    # Check that all parameters match
    assert torch.allclose(inference_sae.W_enc, original_W_enc)
    assert torch.allclose(inference_sae.W_dec, original_W_dec)
    assert torch.allclose(inference_sae.b_enc, original_b_enc)
    assert torch.allclose(inference_sae.b_dec, original_b_dec)

    # Most importantly, check that log_threshold was converted to threshold
    assert torch.allclose(inference_sae.threshold, original_threshold)

    # Verify forward pass gives same results
    sae_in = torch.randn(10, cfg.d_in, device="cpu")

    # Get output from training SAE
    training_feature_acts, _ = training_sae.encode_with_hidden_pre(sae_in)
    training_sae_out = training_sae.decode(training_feature_acts)

    # Get output from inference SAE
    inference_feature_acts = inference_sae.encode(sae_in)
    inference_sae_out = inference_sae.decode(inference_feature_acts)

    # Should produce identical outputs
    assert torch.allclose(training_feature_acts, inference_feature_acts)
    assert torch.allclose(training_sae_out, inference_sae_out)

    # Test the full forward pass
    training_full_out = training_sae(sae_in)
    inference_full_out = inference_sae(sae_in)
    assert torch.allclose(training_full_out, inference_full_out)
