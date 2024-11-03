import pytest
import torch

from sae_lens.training.training_sae import JumpReLU, TrainingSAE
from tests.unit.helpers import build_sae_cfg


def test_jumprelu_sae_encoding():
    cfg = build_sae_cfg(architecture="jumprelu")
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    batch_size = 32
    d_in = sae.cfg.d_in
    d_sae = sae.cfg.d_sae

    x = torch.randn(batch_size, d_in)
    feature_acts, hidden_pre = sae.encode_with_hidden_pre_jumprelu(x)

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
    cfg = build_sae_cfg(architecture="jumprelu")
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    batch_size = 32
    d_in = sae.cfg.d_in

    x = torch.randn(batch_size, d_in)
    train_step_output = sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=sae.cfg.l1_coefficient,
    )

    assert train_step_output.sae_out.shape == (batch_size, d_in)
    assert train_step_output.feature_acts.shape == (batch_size, sae.cfg.d_sae)
    assert pytest.approx(train_step_output.loss.detach(), rel=1e-3) == (
        train_step_output.mse_loss + train_step_output.l1_loss
    )

    expected_mse_loss = (
        (torch.pow((train_step_output.sae_out - x.float()), 2))
        .sum(dim=-1)
        .mean()
        .detach()
        .float()
    )

    assert pytest.approx(train_step_output.mse_loss) == expected_mse_loss
