import os
from copy import deepcopy
from pathlib import Path

import einops
import pytest
import torch
from torch import nn
from transformer_lens.hook_points import HookPoint

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.crosscoder_sae import CrosscoderSAE
from sae_lens.sae import _disable_hooks
from tests.helpers import ALL_ARCHITECTURES, build_sae_cfg


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "hook_name": "blocks.{}.hook_resid_pre",
            "hook_layers": [1,2,3],
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "hook_name": "blocks.{}.hook_resid_pre",
            "hook_layers": [1,2,3],
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "hook_name": "blocks.{}.hook_resid_pre",
            "hook_layers": [1,2,3],
            "d_in": 64,
        },
        # TODO(mkbehr): hook_z support
        # {
        #     "model_name": "tiny-stories-1M",
        #     "dataset_path": "roneneldan/TinyStories",
        #     "hook_name": "blocks.{}.attn.hook_z",
        #     "hook_layers": [1,2,3],
        #     "d_in": 64,
        # },
    ],
    ids=[
        "tiny-stories-1M-resid-pre",
        "tiny-stories-1M-resid-pre-L1-W-dec-Norm",
        "tiny-stories-1M-resid-pre-pretokenized",
        # "tiny-stories-1M-attn-out",
    ],
)
def cfg(request: pytest.FixtureRequest):
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    params = request.param
    return build_sae_cfg(**params)


def test_crosscoder_sae_init(cfg: LanguageModelSAERunnerConfig):
    sae = CrosscoderSAE.from_dict(cfg.get_base_sae_cfg_dict())

    assert isinstance(sae, CrosscoderSAE)

    n_layers = len(cfg.hook_layers)
    assert sae.W_enc.shape == (n_layers, cfg.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.d_sae, n_layers, cfg.d_in)
    assert sae.b_enc.shape == (cfg.d_sae,)
    assert sae.b_dec.shape == (n_layers, cfg.d_in)


def test_crosscoder_sae_fold_w_dec_norm(cfg: LanguageModelSAERunnerConfig):
    sae = CrosscoderSAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae.turn_off_forward_pass_hook_z_reshaping()  # hook z reshaping not needed here.
    assert sae.W_dec.norm(dim=[-2,-1]).mean().item() != pytest.approx(1.0, abs=1e-6)
    sae2 = deepcopy(sae)
    sae2.fold_W_dec_norm()

    W_dec_norms = sae.W_dec.norm(dim=[-2,-1], keepdim=True)
    assert torch.allclose(sae2.W_dec.data, sae.W_dec.data / W_dec_norms)
    assert torch.allclose(sae2.W_enc.data,
                          sae.W_enc.data * einops.rearrange(
                              W_dec_norms, "d_sae 1 1 -> 1 1 d_sae"))
    assert torch.allclose(sae2.b_enc.data, sae.b_enc.data * W_dec_norms.squeeze())

    # fold_W_dec_norm should normalize W_dec to have unit norm.
    assert sae2.W_dec.norm(dim=[-2,-1]).mean().item() == pytest.approx(1.0, abs=1e-6)

    # we expect activations of features to differ by W_dec norm weights.
    activations = torch.randn(10, 4, len(cfg.hook_layers), cfg.d_in,
                              device=cfg.device)
    feature_activations_1 = sae.encode(activations)
    feature_activations_2 = sae2.encode(activations)

    assert torch.allclose(
        feature_activations_1.nonzero(),
        feature_activations_2.nonzero(),
    )

    expected_feature_activations_2 = feature_activations_1 * sae.W_dec.norm(dim=[-2,-1])
    torch.testing.assert_close(feature_activations_2, expected_feature_activations_2)

    sae_out_1 = sae.decode(feature_activations_1)
    sae_out_2 = sae2.decode(feature_activations_2)

    # but actual outputs should be the same
    torch.testing.assert_close(sae_out_1, sae_out_2)
