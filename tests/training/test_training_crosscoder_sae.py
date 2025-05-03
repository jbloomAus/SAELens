import pytest
import torch

from sae_lens.crosscoder_sae import CrosscoderSAE
from sae_lens.training.training_crosscoder_sae import (
    TrainingCrosscoderSAE,
    TrainingCrosscoderSAEConfig,
)
from tests.helpers import build_sae_cfg

def build_crosscoder_sae_cfg(**kwargs):
    return build_sae_cfg(
        **(kwargs | {
            "hook_layers": [1,2,3,4],
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
            }))

def test_TrainingCrosscoderSAE_training_forward_pass_can_scale_sparsity_penalty_by_decoder_norm():
    cfg = build_crosscoder_sae_cfg(
        d_in=3,
        d_sae=5,
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

@pytest.mark.parametrize("architecture", ["standard", "gated", "jumprelu", "topk"])
def test_TrainingCrosscoderSAE_encode_returns_same_value_as_encode_with_hidden_pre(
    architecture: str,
):
    if architecture != "standard":
        pytest.xfail("TODO(mkbehr): support other architectures")
    cfg = build_crosscoder_sae_cfg(architecture=architecture)
    sae = TrainingCrosscoderSAE(
        TrainingCrosscoderSAEConfig.from_sae_runner_config(cfg),
        use_error_term=True,
    )
    x = torch.randn(32, len(cfg.hook_layers), cfg.d_in)
    encode_out = sae.encode(x)
    encode_with_hidden_pre_out = sae.encode_with_hidden_pre_fn(x)[0]
    assert torch.allclose(encode_out, encode_with_hidden_pre_out)

def test_TrainingCrosscoderSAE_heuristic_init():
    cfg = build_crosscoder_sae_cfg(
        d_in=3,
        d_sae=5,
        decoder_heuristic_init=True,
        decoder_heuristic_init_norm=0.2,
    )
    sae = TrainingCrosscoderSAE(
        TrainingCrosscoderSAEConfig.from_sae_runner_config(cfg),
        use_error_term=True)
    print(sae.W_dec.norm(dim=0))
    print(sae.W_dec.norm(dim=1))
    print(sae.W_dec.norm(dim=2))
    torch.testing.assert_close(sae.W_dec.norm(dim=[1,2]),
                               torch.full((5,), 0.2))
