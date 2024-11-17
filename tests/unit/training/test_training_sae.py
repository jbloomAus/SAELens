from pathlib import Path

import pytest
import torch

from sae_lens.sae import SAE
from sae_lens.training.training_sae import TrainingSAE, TrainingSAEConfig
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
        current_sparsity_coefficient=2.0,
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


@pytest.mark.parametrize("architecture", ["standard", "gated", "jumprelu"])
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
