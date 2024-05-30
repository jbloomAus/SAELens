import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae import SAE
from tests.unit.helpers import build_sae_cfg


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "tokenized": False,
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_name": "blocks.1.attn.hook_z",
            "hook_layer": 1,
            "d_in": 64,
        },
    ],
    ids=[
        "tiny-stories-1M-resid-pre",
        "tiny-stories-1M-resid-pre-L1-W-dec-Norm",
        "tiny-stories-1M-resid-pre-pretokenized",
        "tiny-stories-1M-attn-out",
    ],
)
def cfg(request: pytest.FixtureRequest):
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    params = request.param
    return build_sae_cfg(**params)


def test_sae_init(cfg: LanguageModelSAERunnerConfig):
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())

    assert isinstance(sae, SAE)

    assert sae.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sae.b_enc.shape == (cfg.d_sae,)
    assert sae.b_dec.shape == (cfg.d_in,)


def test_sae_fold_w_dec_norm(cfg: LanguageModelSAERunnerConfig):
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae.turn_off_forward_pass_hook_z_reshaping()  # hook z reshaping not needed here.
    assert sae.W_dec.norm(dim=-1).mean().item() != pytest.approx(1.0, abs=1e-6)
    sae2 = deepcopy(sae)
    sae2.fold_W_dec_norm()

    W_dec_norms = sae.W_dec.norm(dim=-1).unsqueeze(1)
    assert torch.allclose(sae2.W_dec.data, sae.W_dec.data / W_dec_norms)
    assert torch.allclose(sae2.W_enc.data, sae.W_enc.data * W_dec_norms.T)
    assert torch.allclose(sae2.b_enc.data, sae.b_enc.data * W_dec_norms.squeeze())

    # fold_W_dec_norm should normalize W_dec to have unit norm.
    assert sae2.W_dec.norm(dim=-1).mean().item() == pytest.approx(1.0, abs=1e-6)

    # we expect activations of features to differ by W_dec norm weights.
    activations = torch.randn(10, 4, cfg.d_in, device=cfg.device)
    feature_activations_1 = sae.encode(activations)
    feature_activations_2 = sae2.encode(activations)

    assert torch.allclose(
        feature_activations_1.nonzero(),
        feature_activations_2.nonzero(),
    )

    expected_feature_activations_2 = feature_activations_1 * sae.W_dec.norm(dim=-1)
    torch.testing.assert_close(feature_activations_2, expected_feature_activations_2)

    sae_out_1 = sae.decode(feature_activations_1)
    sae_out_2 = sae2.decode(feature_activations_2)

    # but actual outputs should be the same
    torch.testing.assert_close(sae_out_1, sae_out_2)


def test_sae_fold_norm_scaling_factor(cfg: LanguageModelSAERunnerConfig):

    norm_scaling_factor = 3.0

    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae.turn_off_forward_pass_hook_z_reshaping()  # hook z reshaping not needed here.

    sae2 = deepcopy(sae)
    sae2.fold_activation_norm_scaling_factor(norm_scaling_factor)

    assert torch.allclose(sae2.W_enc.data, sae.W_enc.data * norm_scaling_factor)

    # we expect activations of features to differ by W_dec norm weights.
    # assume activations are already scaled
    activations = torch.randn(10, 4, cfg.d_in, device=cfg.device)
    # we divide to get the unscale activations
    unscaled_activations = activations / norm_scaling_factor

    feature_activations_1 = sae.encode(activations)
    # with the scaling folded in, the unscaled activations should produce the same
    # result.
    feature_activations_2 = sae2.encode(unscaled_activations)

    assert torch.allclose(
        feature_activations_1.nonzero(),
        feature_activations_2.nonzero(),
    )

    torch.testing.assert_close(feature_activations_2, feature_activations_1)

    sae_out_1 = sae.decode(feature_activations_1)
    sae_out_2 = sae2.decode(feature_activations_2)

    # but actual outputs should be the same
    torch.testing.assert_close(sae_out_1, sae_out_2)


def test_sae_save_and_load_from_pretrained(tmp_path: Path) -> None:
    cfg = build_sae_cfg(device="cpu")
    model_path = str(tmp_path)
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae_state_dict = sae.state_dict()
    sae.save_model(model_path)

    assert os.path.exists(model_path)

    sae_loaded = SAE.load_from_pretrained(model_path, device="cpu")

    sae_loaded_state_dict = sae_loaded.state_dict()

    # check state_dict matches the original
    for key in sae.state_dict().keys():
        assert torch.allclose(
            sae_state_dict[key],
            sae_loaded_state_dict[key],
        )

    sae_in = torch.randn(10, cfg.d_in, device=cfg.device)
    sae_out_1 = sae(sae_in)
    sae_out_2 = sae_loaded(sae_in)
    assert torch.allclose(sae_out_1, sae_out_2)


# TODO: Handle scaling factor in saeBase
# def test_sae_save_and_load_from_pretrained_lacks_scaling_factor(
#     tmp_path: Path,
# ) -> None:
#     cfg = build_sae_cfg(device="cpu")
#     model_path = str(tmp_path)
#     sparse_autoencoder = saeBase(**cfg.get_sae_base_parameters())
#     sparse_autoencoder_state_dict = sparse_autoencoder.state_dict()

#     sparse_autoencoder.save_model(model_path)

#     assert os.path.exists(model_path)

#     sparse_autoencoder_loaded = saeBase.load_from_pretrained(model_path)
#     sparse_autoencoder_loaded.cfg.verbose = True
#     sparse_autoencoder_loaded.cfg.checkpoint_path = cfg.checkpoint_path
#     sparse_autoencoder_loaded = sparse_autoencoder_loaded.to("cpu")
#     sparse_autoencoder_loaded_state_dict = sparse_autoencoder_loaded.state_dict()
#     # check cfg matches the original
#     assert sparse_autoencoder_loaded.cfg == cfg

#     # check state_dict matches the original
#     for key in sparse_autoencoder.state_dict().keys():
#         if key == "scaling_factor":
#             assert isinstance(cfg.d_sae, int)
#             assert torch.allclose(
#                 torch.ones(cfg.d_sae, dtype=cfg.dtype, device=cfg.device),
#                 sparse_autoencoder_loaded_state_dict[key],
#             )
#         else:
#             assert torch.allclose(
#                 sparse_autoencoder_state_dict[key],
#                 sparse_autoencoder_loaded_state_dict[key],
#             )


def test_sae_get_name_returns_correct_name_from_cfg_vals() -> None:
    cfg = build_sae_cfg(model_name="test_model", hook_name="test_hook_name", d_sae=128)
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    assert sae.get_name() == "sae_test_model_test_hook_name_128"
