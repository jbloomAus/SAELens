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
from tests.helpers import ALL_ARCHITECTURES, build_multilayer_sae_cfg


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "hook_name_template": "blocks.{layer}.hook_resid_pre",
            "hook_layers": [0,1,2],
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "hook_name_template": "blocks.{layer}.hook_resid_pre",
            "hook_layers": [0,1,2],
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "hook_name_template": "blocks.{layer}.hook_resid_pre",
            "hook_layers": [0,1,2],
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
        # TODO(mkbehr): hook_z support
        # {
        #     "model_name": "tiny-stories-1M",
        #     "dataset_path": "roneneldan/TinyStories",
        #     "hook_name": "blocks.{layer}.attn.hook_z",
        #     "hook_layers": [0,1,2],
        #     "d_in": 64,
        #     "normalize_sae_decoder": False,
        #     "scale_sparsity_penalty_by_decoder_norm": True,
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
    return build_multilayer_sae_cfg(**params)


def test_crosscoder_sae_init(cfg: LanguageModelSAERunnerConfig):
    sae = CrosscoderSAE.from_dict(cfg.get_base_sae_cfg_dict())

    assert isinstance(sae, CrosscoderSAE)

    n_layers = len(cfg.hook_names)
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
    activations = torch.randn(10, 4, len(cfg.hook_names), cfg.d_in,
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

@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
@torch.no_grad()
def test_sae_fold_w_dec_norm_all_architectures(architecture: str):
    if architecture != "standard":
        pytest.xfail("TODO(mkbehr): support other architectures")
    cfg = build_multilayer_sae_cfg(architecture=architecture, hook_layers=[0,1,2])
    sae = CrosscoderSAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae.turn_off_forward_pass_hook_z_reshaping()  # hook z reshaping not needed here.

    # make sure all parameters are not 0s
    for param in sae.parameters():
        param.data = torch.rand_like(param)

    assert sae.W_dec.norm(dim=[-2,-1]).mean().item() != pytest.approx(1.0, abs=1e-6)
    sae2 = deepcopy(sae)
    sae2.fold_W_dec_norm()

    # fold_W_dec_norm should normalize W_dec to have unit norm.
    assert sae2.W_dec.norm(dim=[-2,-1]).mean().item() == pytest.approx(1.0, abs=1e-6)

    # we expect activations of features to differ by W_dec norm weights.
    activations = torch.randn(10, 4, len(cfg.hook_names), cfg.d_in, device=cfg.device)
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

@torch.no_grad()
def test_sae_fold_norm_scaling_factor(cfg: LanguageModelSAERunnerConfig):
    norm_scaling_factor = torch.Tensor([2.0, 3.0, 4.0])

    sae = CrosscoderSAE.from_dict(cfg.get_base_sae_cfg_dict())
    # make sure b_dec and b_enc are not 0s
    sae.b_dec.data = torch.randn(len(cfg.hook_names), cfg.d_in, device=cfg.device)
    sae.b_enc.data = torch.randn(cfg.d_sae, device=cfg.device)  # type: ignore
    sae.turn_off_forward_pass_hook_z_reshaping()  # hook z reshaping not needed here.

    sae2 = deepcopy(sae)
    sae2.fold_activation_norm_scaling_factor(norm_scaling_factor)

    assert sae2.cfg.normalize_activations == "none"

    assert torch.allclose(sae2.W_enc.data, sae.W_enc.data * norm_scaling_factor.reshape((-1,1,1)))

    # we expect activations of features to differ by W_dec norm weights.
    # assume activations are already scaled
    activations = torch.randn(10, 4, len(cfg.hook_names), cfg.d_in, device=cfg.device)
    # we divide to get the unscale activations
    unscaled_activations = activations / norm_scaling_factor.unsqueeze(-1)

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
    sae_out_2 = norm_scaling_factor.unsqueeze(-1) * sae2.decode(feature_activations_2)

    # but actual outputs should be the same
    torch.testing.assert_close(sae_out_1, sae_out_2)


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
@torch.no_grad()
def test_sae_fold_norm_scaling_factor_all_architectures(architecture: str):
    if architecture != "standard":
        pytest.xfail("TODO(mkbehr): support other architectures")
    cfg = build_multilayer_sae_cfg(architecture=architecture, hook_layers=[0,1,2])
    norm_scaling_factor = torch.Tensor([2.0, 3.0, 4.0])

    sae = CrosscoderSAE.from_dict(cfg.get_base_sae_cfg_dict())
    # make sure all parameters are not 0s
    for param in sae.parameters():
        param.data = torch.rand_like(param)

    sae2 = deepcopy(sae)
    sae2.fold_activation_norm_scaling_factor(norm_scaling_factor)

    assert sae2.cfg.normalize_activations == "none"

    assert torch.allclose(sae2.W_enc.data, sae.W_enc.data * norm_scaling_factor.reshape((-1,1,1)))

    # we expect activations of features to differ by W_dec norm weights.
    # assume activations are already scaled
    activations = torch.randn(10, 4, len(cfg.hook_names), cfg.d_in, device=cfg.device)
    # we divide to get the unscale activations
    unscaled_activations = activations / norm_scaling_factor.unsqueeze(-1)

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
    sae_out_2 = norm_scaling_factor.unsqueeze(-1) * sae2.decode(feature_activations_2)

    # but actual outputs should be the same
    torch.testing.assert_close(sae_out_1, sae_out_2)

def test_sae_save_and_load_from_pretrained(tmp_path: Path) -> None:
    cfg = build_multilayer_sae_cfg(hook_layers=[0,1,2])
    model_path = str(tmp_path)
    sae = CrosscoderSAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae_state_dict = sae.state_dict()
    sae.save_model(model_path)

    assert os.path.exists(model_path)

    sae_loaded = CrosscoderSAE.load_from_pretrained(model_path, device="cpu")

    sae_loaded_state_dict = sae_loaded.state_dict()

    # check state_dict matches the original
    for key in sae.state_dict():
        assert torch.allclose(
            sae_state_dict[key],
            sae_loaded_state_dict[key],
        )

    sae_in = torch.randn(10, len(cfg.hook_names), cfg.d_in, device=cfg.device)
    sae_out_1 = sae(sae_in)
    sae_out_2 = sae_loaded(sae_in)
    assert torch.allclose(sae_out_1, sae_out_2)

@pytest.mark.xfail(reason="TODO(mkbehr): support other architectures")
def test_sae_save_and_load_from_pretrained_gated(tmp_path: Path) -> None:
    cfg = build_multilayer_sae_cfg(architecture="gated", hook_layers=[0,1,2])
    model_path = str(tmp_path)
    sae = CrosscoderSAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae_state_dict = sae.state_dict()
    sae.save_model(model_path)

    assert os.path.exists(model_path)

    sae_loaded = CrosscoderSAE.load_from_pretrained(model_path, device="cpu")

    sae_loaded_state_dict = sae_loaded.state_dict()

    # check state_dict matches the original
    for key in sae.state_dict():
        assert torch.allclose(
            sae_state_dict[key],
            sae_loaded_state_dict[key],
        )

    sae_in = torch.randn(10, len(cfg.hook_names), cfg.d_in, device=cfg.device)
    sae_out_1 = sae(sae_in)
    sae_out_2 = sae_loaded(sae_in)
    assert torch.allclose(sae_out_1, sae_out_2)

def test_sae_save_and_load_from_pretrained_topk(tmp_path: Path) -> None:
    cfg = build_multilayer_sae_cfg(activation_fn_kwargs={"k": 30}, hook_layers=[0,1,2])
    model_path = str(tmp_path)
    sae = CrosscoderSAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae_state_dict = sae.state_dict()
    sae.save_model(model_path)

    assert os.path.exists(model_path)

    sae_loaded = CrosscoderSAE.load_from_pretrained(model_path, device="cpu")

    sae_loaded_state_dict = sae_loaded.state_dict()

    # check state_dict matches the original
    for key in sae.state_dict():
        assert torch.allclose(
            sae_state_dict[key],
            sae_loaded_state_dict[key],
        )

    sae_in = torch.randn(10, len(cfg.hook_names), cfg.d_in, device=cfg.device)
    sae_out_1 = sae(sae_in)
    sae_out_2 = sae_loaded(sae_in)
    assert torch.allclose(sae_out_1, sae_out_2)

def test_sae_get_name_returns_correct_name_from_cfg_vals() -> None:
    cfg = build_multilayer_sae_cfg(model_name="test_model", hook_name_template="blocks.{layer}.test_hook_name", d_sae=128, hook_layers=[0,1,2])
    sae = CrosscoderSAE.from_dict(cfg.get_base_sae_cfg_dict())
    assert sae.get_name() == "sae_test_model_blocks.layers_0_through_2.test_hook_name_128"
