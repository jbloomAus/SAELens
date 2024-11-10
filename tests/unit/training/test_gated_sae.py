import pytest
import torch

from sae_lens.training.training_sae import TrainingSAE
from tests.unit.helpers import build_sae_cfg


def test_gated_sae_initialization():
    cfg = build_sae_cfg()
    setattr(cfg, "architecture", "gated")
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    assert sae.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    # assert sae.b_enc.shape == (cfg.d_sae,)
    assert sae.b_mag.shape == (cfg.d_sae,)
    assert sae.b_gate.shape == (cfg.d_sae,)
    assert sae.r_mag.shape == (cfg.d_sae,)
    assert sae.b_dec.shape == (cfg.d_in,)
    assert isinstance(sae.activation_fn, torch.nn.ReLU)
    assert sae.device == torch.device("cpu")
    assert sae.dtype == torch.float32

    # biases
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert torch.allclose(sae.b_mag, torch.zeros_like(sae.b_mag), atol=1e-6)
    assert torch.allclose(sae.b_gate, torch.zeros_like(sae.b_gate), atol=1e-6)

    # check if the decoder weight norm is 1 by default
    assert torch.allclose(
        sae.W_dec.norm(dim=1), torch.ones_like(sae.W_dec.norm(dim=1)), atol=1e-6
    )


def test_gated_sae_encoding():
    cfg = build_sae_cfg()
    setattr(cfg, "architecture", "gated")
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    batch_size = 32
    d_in = sae.cfg.d_in
    d_sae = sae.cfg.d_sae

    x = torch.randn(batch_size, d_in)
    feature_acts, hidden_pre = sae.encode_with_hidden_pre_gated(x)

    assert feature_acts.shape == (batch_size, d_sae)
    assert hidden_pre.shape == (batch_size, d_sae)

    # Check the gating mechanism
    gating_pre_activation = x @ sae.W_enc + sae.b_gate
    active_features = (gating_pre_activation > 0).float()
    magnitude_pre_activation = x @ (sae.W_enc * sae.r_mag.exp()) + sae.b_mag
    feature_magnitudes = sae.activation_fn(magnitude_pre_activation)

    expected_feature_acts = active_features * feature_magnitudes
    assert torch.allclose(feature_acts, expected_feature_acts, atol=1e-6)


def test_gated_sae_loss():
    cfg = build_sae_cfg()
    setattr(cfg, "architecture", "gated")
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

    sae_in_centered = x - sae.b_dec
    via_gate_feature_magnitudes = torch.relu(sae_in_centered @ sae.W_enc + sae.b_gate)
    preactivation_l1_loss = (
        sae.cfg.l1_coefficient * torch.sum(via_gate_feature_magnitudes, dim=-1).mean()
    )

    via_gate_reconstruction = (
        via_gate_feature_magnitudes @ sae.W_dec.detach() + sae.b_dec.detach()
    )
    aux_reconstruction_loss = torch.sum(
        (via_gate_reconstruction - x) ** 2, dim=-1
    ).mean()

    expected_loss = (
        train_step_output.losses["mse_loss"]
        + preactivation_l1_loss
        + aux_reconstruction_loss
    )
    assert (
        pytest.approx(train_step_output.loss.item(), rel=1e-3) == expected_loss.item()
    )


def test_gated_sae_forward_pass():
    cfg = build_sae_cfg()
    setattr(cfg, "architecture", "gated")
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    batch_size = 32
    d_in = sae.cfg.d_in

    x = torch.randn(batch_size, d_in)
    sae_out = sae(x)

    assert sae_out.shape == (batch_size, d_in)


def test_gated_sae_training_forward_pass():
    cfg = build_sae_cfg()
    setattr(cfg, "architecture", "gated")
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

    # Detach the loss tensor and convert to numpy for comparison
    detached_loss = train_step_output.loss.detach().cpu().numpy()
    expected_loss = (
        (
            train_step_output.losses["mse_loss"]
            + train_step_output.losses["l1_loss"]
            + train_step_output.losses["auxiliary_reconstruction_loss"]
        )
        .detach()  # type: ignore
        .cpu()
        .numpy()
    )

    assert pytest.approx(detached_loss, rel=1e-3) == expected_loss
