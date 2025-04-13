import pytest
import torch

from sae_lens.crosscoder_sae import CrosscoderSAE
from sae_lens.training.training_crosscoder_sae import (
    TrainingCrosscoderSAE,
    TrainingCrosscoderSAEConfig,
)
from tests.helpers import build_sae_cfg

def test_TrainingCrosscoderSAE_training_forward_pass_can_scale_sparsity_penalty_by_decoder_norm():
    cfg = build_sae_cfg(
        d_in=3,
        d_sae=5,
        hook_layers=[1,2,3,4],
        normalize_sae_decoder=False,
        scale_sparsity_penalty_by_decoder_norm=True,
    )
    training_sae = TrainingCrosscoderSAE(
        TrainingCrosscoderSAEConfig.from_sae_runner_config(cfg),
        use_error_term=True,
    )
    x = torch.randn(32, 4, 3)
    train_step_output = training_sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=2.0,
    )
    feature_acts = train_step_output.feature_acts
    decoder_norms = training_sae.W_dec.norm(dim=-1)
    decoder_norm = decoder_norms.sum(dim=-1)
    # double-check decoder norm is not all ones, or this test is pointless
    assert not torch.allclose(decoder_norm, torch.ones_like(decoder_norm), atol=1e-2)
    scaled_feature_acts = feature_acts * decoder_norm

    assert (
        pytest.approx(train_step_output.losses["l1_loss"].detach().item())  # type: ignore
        == 2.0 * scaled_feature_acts.norm(p=1, dim=1).mean().detach().item()
    )
