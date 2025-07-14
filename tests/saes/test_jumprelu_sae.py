import pytest
import torch
from torch import nn

from sae_lens.saes.jumprelu_sae import JumpReLU, JumpReLUSAE, JumpReLUTrainingSAE
from sae_lens.saes.sae import TrainStepInput
from tests.helpers import build_jumprelu_sae_cfg, build_jumprelu_sae_training_cfg


def test_jumprelu_sae_encoding():
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

    assert torch.allclose(feature_acts, expected_feature_acts, atol=1e-6)  # type: ignore


def test_jumprelu_sae_training_forward_pass():
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


def test_sae_jumprelu_initialization():
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
    assert not torch.allclose(sae.W_enc, torch.zeros_like(sae.W_enc))
    assert not torch.allclose(sae.W_dec, torch.zeros_like(sae.W_dec))
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec))
    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc))
    assert torch.allclose(sae.threshold, torch.zeros_like(sae.threshold))


@pytest.mark.parametrize("use_error_term", [True, False])
def test_sae_jumprelu_forward(use_error_term: bool):
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
    assert torch.allclose(out, expected_output)
    assert torch.allclose(cache["hook_sae_input"], sae_in)
    assert torch.allclose(cache["hook_sae_output"], out)
    assert torch.allclose(cache["hook_sae_recons"], expected_recons)
    if use_error_term:
        assert torch.allclose(
            cache["hook_sae_error"], expected_output - expected_recons
        )

    assert torch.allclose(cache["hook_sae_acts_pre"], torch.tensor([[0.6, 0.6, 0.6]]))
    # the threshold of 1.0 should block the first latent from firing
    assert torch.allclose(cache["hook_sae_acts_post"], torch.tensor([[0.0, 0.6, 0.6]]))


def test_SparseAutoencoder_initialization_jumprelu():
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
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)

    # check if the decoder weight norm is 0.1 by default
    assert torch.allclose(
        sae.W_dec.norm(dim=1), 0.1 * torch.ones_like(sae.W_dec.norm(dim=1)), atol=1e-6
    )

    #  Default currently should be tranpose initialization
    assert torch.allclose(sae.W_enc, sae.W_dec.T, atol=1e-6)
