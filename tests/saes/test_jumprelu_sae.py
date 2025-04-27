import pytest
import torch

from sae_lens.saes.jumprelu_sae import JumpReLU, JumpReLUTrainingSAE
from sae_lens.saes.sae import TrainStepInput
from tests.helpers import build_jumprelu_sae_training_cfg


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
