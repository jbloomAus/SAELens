import pytest
import torch

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
        current_l1_coefficient=2.0,
    )
    feature_acts = train_step_output.feature_acts
    decoder_norm = training_sae.W_dec.norm(dim=1)
    # double-check decoder norm is not all ones, or this test is pointless
    assert not torch.allclose(decoder_norm, torch.ones_like(decoder_norm), atol=1e-2)
    scaled_feature_acts = feature_acts * decoder_norm

    if scale_sparsity_penalty_by_decoder_norm:
        assert (
            pytest.approx(train_step_output.l1_loss)
            == 2.0 * scaled_feature_acts.norm(p=1, dim=1).mean().detach().item()
        )
    else:
        assert (
            pytest.approx(train_step_output.l1_loss)
            == 2.0 * feature_acts.norm(p=1, dim=1).mean().detach().item()
        )
