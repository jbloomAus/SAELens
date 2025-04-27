# Placeholder for standard SAE tests

import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from transformer_lens.hook_points import HookPoint

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.saes.gated_sae import GatedTrainingSAE
from sae_lens.saes.jumprelu_sae import JumpReLUSAE, JumpReLUTrainingSAE
from sae_lens.saes.sae import SAE, _disable_hooks
from sae_lens.saes.standard_sae import StandardSAE, StandardTrainingSAE
from tests.helpers import (
    ALL_ARCHITECTURES,
    build_runner_cfg,
)


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
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
    return build_runner_cfg(**params)


def test_sae_init(cfg: LanguageModelSAERunnerConfig):
    sae = StandardSAE(cfg.sae)  # type: ignore

    assert isinstance(sae, SAE)

    assert sae.W_enc.shape == (cfg.sae.d_in, cfg.sae.d_sae)
    assert sae.W_dec.shape == (cfg.sae.d_sae, cfg.sae.d_in)
    assert sae.b_enc.shape == (cfg.sae.d_sae,)
    assert sae.b_dec.shape == (cfg.sae.d_in,)


def test_sae_fold_w_dec_norm(cfg: LanguageModelSAERunnerConfig):
    sae = StandardSAE(cfg.sae)  # type: ignore
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
    activations = torch.randn(10, 4, cfg.sae.d_in, device=cfg.device)
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


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
@torch.no_grad()
def test_sae_fold_w_dec_norm_all_architectures(architecture: str):
    cfg = build_runner_cfg(architecture=architecture)
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae.turn_off_forward_pass_hook_z_reshaping()  # hook z reshaping not needed here.

    # make sure all parameters are not 0s
    for param in sae.parameters():
        param.data = torch.rand_like(param)

    assert sae.W_dec.norm(dim=-1).mean().item() != pytest.approx(1.0, abs=1e-6)
    sae2 = deepcopy(sae)
    sae2.fold_W_dec_norm()

    # fold_W_dec_norm should normalize W_dec to have unit norm.
    assert sae2.W_dec.norm(dim=-1).mean().item() == pytest.approx(1.0, abs=1e-6)

    # we expect activations of features to differ by W_dec norm weights.
    activations = torch.randn(10, 4, cfg.sae.d_in, device=cfg.device)
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


@torch.no_grad()
def test_sae_fold_norm_scaling_factor(cfg: LanguageModelSAERunnerConfig):
    norm_scaling_factor = 3.0

    sae = StandardSAE(cfg.sae)  # type: ignore
    # make sure b_dec and b_enc are not 0s
    sae.b_dec.data = torch.randn(cfg.sae.d_in, device=cfg.device)
    sae.b_enc.data = torch.randn(cfg.sae.d_sae, device=cfg.device)  # type: ignore
    sae.turn_off_forward_pass_hook_z_reshaping()  # hook z reshaping not needed here.

    sae2 = deepcopy(sae)
    sae2.fold_activation_norm_scaling_factor(norm_scaling_factor)

    assert sae2.cfg.normalize_activations == "none"

    assert torch.allclose(sae2.W_enc.data, sae.W_enc.data * norm_scaling_factor)

    # we expect activations of features to differ by W_dec norm weights.
    # assume activations are already scaled
    activations = torch.randn(10, 4, cfg.sae.d_in, device=cfg.device)
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
    sae_out_2 = norm_scaling_factor * sae2.decode(feature_activations_2)

    # but actual outputs should be the same
    torch.testing.assert_close(sae_out_1, sae_out_2)


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
@torch.no_grad()
def test_sae_fold_norm_scaling_factor_all_architectures(architecture: str):
    cfg = build_runner_cfg(architecture=architecture)
    norm_scaling_factor = 3.0

    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    # make sure all parameters are not 0s
    for param in sae.parameters():
        param.data = torch.rand_like(param)

    sae2 = deepcopy(sae)
    sae2.fold_activation_norm_scaling_factor(norm_scaling_factor)

    assert sae2.cfg.normalize_activations == "none"

    assert torch.allclose(sae2.W_enc.data, sae.W_enc.data * norm_scaling_factor)

    # we expect activations of features to differ by W_dec norm weights.
    # assume activations are already scaled
    activations = torch.randn(10, 4, cfg.sae.d_in, device=cfg.device)
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
    sae_out_2 = norm_scaling_factor * sae2.decode(feature_activations_2)

    # but actual outputs should be the same
    torch.testing.assert_close(sae_out_1, sae_out_2)


def test_sae_save_and_load_from_pretrained(tmp_path: Path) -> None:
    cfg = build_runner_cfg()
    model_path = str(tmp_path)
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae_state_dict = sae.state_dict()
    sae.save_model(model_path)

    assert os.path.exists(model_path)

    sae_loaded = SAE.load_from_pretrained(model_path, device="cpu")

    sae_loaded_state_dict = sae_loaded.state_dict()

    # check state_dict matches the original
    for key in sae.state_dict():
        assert torch.allclose(
            sae_state_dict[key],
            sae_loaded_state_dict[key],
        )

    sae_in = torch.randn(10, cfg.sae.d_in, device=cfg.device)
    sae_out_1 = sae(sae_in)
    sae_out_2 = sae_loaded(sae_in)
    assert torch.allclose(sae_out_1, sae_out_2)


def test_sae_seqpos(tmp_path: Path) -> None:
    cfg = build_runner_cfg(seqpos_slice=(1, 3))
    model_path = str(tmp_path)
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())

    assert sae.cfg.seqpos_slice == (1, 3)

    sae.save_model(model_path)

    sae_loaded = SAE.load_from_pretrained(model_path, device="cpu")

    assert sae_loaded.cfg.seqpos_slice == (1, 3)


# TODO: Handle scaling factor in saeBase
# def test_sae_save_and_load_from_pretrained_lacks_scaling_factor(
#     tmp_path: Path,
# ) -> None:
#     cfg = build_sae_cfg()
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
#             assert isinstance(cfg.sae.d_sae, int)
#             assert torch.allclose(
#                 torch.ones(cfg.sae.d_sae, dtype=cfg.dtype, device=cfg.device),
#                 sparse_autoencoder_loaded_state_dict[key],
#             )
#         else:
#             assert torch.allclose(
#                 sparse_autoencoder_state_dict[key],
#                 sparse_autoencoder_loaded_state_dict[key],
#             )


def test_sae_get_name_returns_correct_name_from_cfg_vals() -> None:
    cfg = build_runner_cfg(
        model_name="test_model", hook_name="test_hook_name", d_sae=128
    )
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    assert sae.get_name() == "sae_test_model_test_hook_name_128"


def test_sae_move_between_devices() -> None:
    cfg = build_runner_cfg()
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())

    sae.to("meta")
    assert sae.device == torch.device("meta")
    assert sae.cfg.device == "meta"
    assert sae.W_enc.device == torch.device("meta")


def test_sae_change_dtype() -> None:
    cfg = build_runner_cfg(dtype="float64")
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())

    sae.to(dtype=torch.float16)
    assert sae.dtype == torch.float16
    assert sae.cfg.dtype == "torch.float16"


def test_sae_jumprelu_initialization():
    cfg = build_runner_cfg(architecture="jumprelu", device="cpu")
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    assert isinstance(sae.W_enc, nn.Parameter)
    assert isinstance(sae.W_dec, nn.Parameter)
    assert isinstance(sae.b_enc, nn.Parameter)
    assert isinstance(sae.b_dec, nn.Parameter)
    assert isinstance(sae.threshold, nn.Parameter)

    assert sae.W_enc.shape == (cfg.sae.d_in, cfg.sae.d_sae)
    assert sae.W_dec.shape == (cfg.sae.d_sae, cfg.sae.d_in)
    assert sae.b_enc.shape == (cfg.sae.d_sae,)
    assert sae.b_dec.shape == (cfg.sae.d_in,)
    assert sae.threshold.shape == (cfg.sae.d_sae,)

    # encoder/decoder should be initialized, everything else should be 0s
    assert not torch.allclose(sae.W_enc, torch.zeros_like(sae.W_enc))
    assert not torch.allclose(sae.W_dec, torch.zeros_like(sae.W_dec))
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec))
    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc))
    assert torch.allclose(sae.threshold, torch.zeros_like(sae.threshold))


@pytest.mark.parametrize("use_error_term", [True, False])
def test_sae_jumprelu_forward(use_error_term: bool):
    cfg = build_runner_cfg(architecture="jumprelu", d_in=2, d_sae=3)
    sae = JumpReLUSAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae.use_error_term = use_error_term
    sae.threshold.data = torch.tensor([1.0, 0.5, 0.25])
    sae.W_enc.data = torch.ones_like(sae.W_enc.data)
    sae.W_dec.data = torch.ones_like(sae.W_dec.data)
    sae.b_enc.data = torch.zeros_like(sae.b_enc.data)
    sae.b_dec.data = torch.zeros_like(sae.b_dec.data)

    sae_in = 0.3 * torch.ones(1, 2)
    expected_recons = torch.tensor([[1.2, 1.2]])
    # if we use error term, we should always get the same output as what we put in
    expected_output = sae_in if use_error_term else expected_recons
    out, cache = sae.run_with_cache(sae_in)
    assert torch.allclose(out, expected_output)
    assert torch.allclose(cache["hook_sae_input"], sae_in)
    assert torch.allclose(cache["hook_sae_output"], out)
    assert torch.allclose(cache["hook_sae_recons"], expected_recons)
    if use_error_term:
        assert torch.allclose(
            cache["hook_sae_error"], expected_output - expected_recons
        )

    assert torch.allclose(cache["hook_sae_acts_pre"], torch.tensor([[0.6, 0.6, 0.6]]))
    # the threshold of 1.0 should block the first latent from firing
    assert torch.allclose(cache["hook_sae_acts_post"], torch.tensor([[0.0, 0.6, 0.6]]))


def test_sae_gated_initialization():
    cfg = build_runner_cfg(architecture="gated")
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    assert isinstance(sae.W_enc, nn.Parameter)
    assert isinstance(sae.W_dec, nn.Parameter)
    assert isinstance(sae.b_dec, nn.Parameter)
    assert isinstance(sae.b_gate, nn.Parameter)
    assert isinstance(sae.r_mag, nn.Parameter)
    assert isinstance(sae.b_mag, nn.Parameter)

    assert sae.W_enc.shape == (cfg.sae.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.sae.d_sae, cfg.sae.d_in)
    assert sae.b_dec.shape == (cfg.sae.d_in,)
    assert sae.b_gate.shape == (cfg.sae.d_sae,)
    assert sae.r_mag.shape == (cfg.sae.d_sae,)
    assert sae.b_mag.shape == (cfg.sae.d_sae,)

    assert not torch.allclose(sae.W_enc, torch.zeros_like(sae.W_enc))
    assert not torch.allclose(sae.W_dec, torch.zeros_like(sae.W_dec))
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec))
    assert torch.allclose(sae.b_gate, torch.zeros_like(sae.b_gate))
    assert torch.allclose(sae.r_mag, torch.zeros_like(sae.r_mag))
    assert torch.allclose(sae.b_mag, torch.zeros_like(sae.b_mag))


def test_disable_hooks_temporarily_stops_hooks_from_running():
    cfg = build_runner_cfg(d_in=2, d_sae=3)
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae_in = torch.randn(10, cfg.sae.d_in)

    orig_out, orig_cache = sae.run_with_cache(sae_in)
    with _disable_hooks(sae):
        disabled_out, disabled_cache = sae.run_with_cache(sae_in)
    subseq_out, subseq_cache = sae.run_with_cache(sae_in)

    assert torch.allclose(orig_out, disabled_out)
    assert torch.allclose(orig_out, subseq_out)
    assert disabled_cache.keys() == set()
    for key in orig_cache:
        assert torch.allclose(orig_cache[key], subseq_cache[key])


@pytest.mark.parametrize("architecture", ["standard", "gated", "jumprelu"])
def test_sae_forward_pass_works_with_error_term_and_hooks(architecture: str):
    cfg = build_runner_cfg(architecture=architecture, d_in=32, d_sae=64)
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae.use_error_term = True
    sae_in = torch.randn(10, cfg.sae.d_in)
    original_out, original_cache = sae.run_with_cache(sae_in)

    def ablate_hooked_sae(acts: torch.Tensor, hook: HookPoint):  # noqa: ARG001
        acts[:, :] = 20
        return acts

    with sae.hooks(fwd_hooks=[("hook_sae_acts_post", ablate_hooked_sae)]):
        ablated_out, ablated_cache = sae.run_with_cache(sae_in)

    assert not torch.allclose(original_out, ablated_out, rtol=1e-2)
    assert torch.all(ablated_cache["hook_sae_acts_post"] == 20)
    assert torch.allclose(
        original_cache["hook_sae_error"], ablated_cache["hook_sae_error"], rtol=1e-4
    )


def test_SparseAutoencoder_from_pretrained_loads_from_hugginface_using_shorthand():
    sae, original_cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.0.hook_resid_pre",
        device="cpu",
    )

    assert (
        sae.cfg.neuronpedia_id == "gpt2-small/0-res-jb"
    )  # what we expect from the yml

    # it should match what we get when manually loading from hf
    repo_id = "jbloom/GPT2-Small-SAEs-Reformatted"
    hook_point = "blocks.0.hook_resid_pre"
    filename = f"{hook_point}/sae_weights.safetensors"
    weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
    state_dict = {}
    with safe_open(weight_path, framework="pt", device="cpu") as f:  # type: ignore
        for k in f.keys():  # noqa: SIM118
            state_dict[k] = f.get_tensor(k)

    assert isinstance(sae, SAE)
    assert sae.cfg.model_name == "gpt2-small"
    assert sae.cfg.hook_name == "blocks.0.hook_resid_pre"

    assert isinstance(original_cfg_dict, dict)

    assert isinstance(sparsity, torch.Tensor)
    assert sparsity.shape == (sae.cfg.sae.d_sae,)
    assert sparsity.max() < 0.0

    for k in sae.state_dict():
        if k == "finetuning_scaling_factor":
            continue
        assert torch.allclose(sae.state_dict()[k], state_dict[k])


def test_SparseAutoencoder_from_pretrained_can_load_arbitrary_saes_from_hugginface():
    sae, original_cfg_dict, sparsity = SAE.from_pretrained(
        release="jbloom/GPT2-Small-SAEs-Reformatted",
        sae_id="blocks.0.hook_resid_pre",
        device="cpu",
    )

    # it should match what we get when manually loading from hf
    repo_id = "jbloom/GPT2-Small-SAEs-Reformatted"
    hook_point = "blocks.0.hook_resid_pre"
    filename = f"{hook_point}/sae_weights.safetensors"
    weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
    state_dict = {}
    with safe_open(weight_path, framework="pt", device="cpu") as f:  # type: ignore
        for k in f.keys():  # noqa: SIM118
            state_dict[k] = f.get_tensor(k)

    assert isinstance(sae, SAE)
    assert sae.cfg.model_name == "gpt2-small"
    assert sae.cfg.hook_name == "blocks.0.hook_resid_pre"

    assert isinstance(original_cfg_dict, dict)

    assert isinstance(sparsity, torch.Tensor)
    assert sparsity.shape == (sae.cfg.sae.d_sae,)
    assert sparsity.max() < 0.0

    for k in sae.state_dict():
        if k == "finetuning_scaling_factor":
            continue
        assert torch.allclose(sae.state_dict()[k], state_dict[k])


def test_SparseAutoencoder_from_pretrained_errors_for_invalid_releases():
    with pytest.raises(ValueError):
        SAE.from_pretrained(
            release="wrong",
            sae_id="blocks.0.hook_resid_pre",
            device="cpu",
        )


def test_SparseAutoencoder_from_pretrained_errors_for_invalid_sae_ids():
    with pytest.raises(ValueError):
        SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="wrong",
            device="cpu",
        )


def test_SparseAutoencoder_initialization_standard():
    cfg = build_runner_cfg()

    sae = StandardTrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    assert sae.W_enc.shape == (cfg.sae.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.sae.d_sae, cfg.sae.d_in)
    assert sae.b_enc.shape == (cfg.sae.d_sae,)
    assert sae.b_dec.shape == (cfg.sae.d_in,)
    assert isinstance(sae.activation_fn, torch.nn.ReLU)
    assert sae.device == torch.device("cpu")
    assert sae.dtype == torch.float32

    # biases
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)

    # check if the decoder weight norm is 1 by default
    assert torch.allclose(
        sae.W_dec.norm(dim=1), torch.ones_like(sae.W_dec.norm(dim=1)), atol=1e-6
    )

    #  Default currently shouldn't be tranpose initialization
    unit_normed_W_enc = sae.W_enc / torch.norm(sae.W_enc, dim=0)
    unit_normed_W_dec = sae.W_dec.T
    assert not torch.allclose(unit_normed_W_enc, unit_normed_W_dec, atol=1e-6)


def test_SparseAutoencoder_initialization_gated():
    cfg = build_runner_cfg()
    setattr(cfg, "architecture", "gated")
    sae = GatedTrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    assert sae.W_enc.shape == (cfg.sae.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.sae.d_sae, cfg.sae.d_in)
    assert sae.b_mag.shape == (cfg.sae.d_sae,)
    assert sae.b_gate.shape == (cfg.sae.d_sae,)
    assert sae.r_mag.shape == (cfg.sae.d_sae,)
    assert sae.b_dec.shape == (cfg.sae.d_in,)
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

    #  Default currently shouldn't be tranpose initialization
    unit_normed_W_enc = sae.W_enc / torch.norm(sae.W_enc, dim=0)
    unit_normed_W_dec = sae.W_dec.T
    assert not torch.allclose(unit_normed_W_enc, unit_normed_W_dec, atol=1e-6)


def test_SparseAutoencoder_initialization_jumprelu():
    cfg = build_runner_cfg(architecture="jumprelu")
    sae = JumpReLUTrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    assert sae.W_enc.shape == (cfg.sae.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.sae.d_sae, cfg.sae.d_in)
    assert isinstance(sae.log_threshold, torch.nn.Parameter)
    assert sae.log_threshold.shape == (cfg.sae.d_sae,)
    assert sae.b_enc.shape == (cfg.sae.d_sae,)
    assert sae.b_dec.shape == (cfg.sae.d_in,)
    assert isinstance(sae.activation_fn, torch.nn.ReLU)
    assert sae.device == torch.device("cpu")
    assert sae.dtype == torch.float32

    # biases
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)

    # check if the decoder weight norm is 1 by default
    assert torch.allclose(
        sae.W_dec.norm(dim=1), torch.ones_like(sae.W_dec.norm(dim=1)), atol=1e-6
    )

    #  Default currently shouldn't be tranpose initialization
    unit_normed_W_enc = sae.W_enc / torch.norm(sae.W_enc, dim=0)
    unit_normed_W_dec = sae.W_dec.T
    assert not torch.allclose(unit_normed_W_enc, unit_normed_W_dec, atol=1e-6)


def test_SparseAutoencoder_initialization_orthogonal_enc_dec():
    cfg = build_runner_cfg(decoder_orthogonal_init=True, expansion_factor=2)

    sae = StandardTrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())
    projections = sae.W_dec.T @ sae.W_dec
    mask = ~torch.eye(projections.size(0), dtype=torch.bool)

    assert projections[mask].max() < 0.1

    # initialized weights of biases are 0
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)


def test_SparseAutoencoder_initialization_normalize_decoder_norm():
    cfg = build_runner_cfg(normalize_sae_decoder=True)

    sae = StandardTrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    assert torch.allclose(
        sae.W_dec.norm(dim=1), torch.ones_like(sae.W_dec.norm(dim=1)), atol=1e-6
    )

    # initialized weights of biases are 0
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)


def test_SparseAutoencoder_initialization_encoder_is_decoder_transpose():
    cfg = build_runner_cfg(init_encoder_as_decoder_transpose=True)

    sae = StandardTrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    # If we decoder norms are 1 we need to unit norm W_enc first.
    unit_normed_W_enc = sae.W_enc / torch.norm(sae.W_enc, dim=0)
    unit_normed_W_dec = sae.W_dec.T
    assert torch.allclose(unit_normed_W_enc, unit_normed_W_dec, atol=1e-6)

    # initialized weights of biases are 0
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)


def test_SparseAutoencoder_initialization_enc_dec_T_no_unit_norm():
    cfg = build_runner_cfg(
        init_encoder_as_decoder_transpose=True,
        normalize_sae_decoder=False,
    )

    sae = StandardTrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    assert torch.allclose(sae.W_dec, sae.W_enc.T, atol=1e-6)

    # initialized weights of biases are 0
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)


def test_SparseAutoencoder_initialization_heuristic_init_and_normalize_sae_decoder():
    # assert that an error is raised
    with pytest.raises(ValueError):
        _ = build_runner_cfg(
            decoder_heuristic_init=True,
            normalize_sae_decoder=True,
        )


def test_SparseAutoencoder_initialization_decoder_norm_in_loss_and_normalize_sae_decoder():
    # assert that an error is raised
    with pytest.raises(ValueError):
        _ = build_runner_cfg(
            scale_sparsity_penalty_by_decoder_norm=True,
            normalize_sae_decoder=True,
        )


def test_SparseAutoencoder_initialization_heuristic_init():
    cfg = build_runner_cfg(
        decoder_heuristic_init=True,
        normalize_sae_decoder=False,
    )

    sae = StandardTrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    decoder_norms = sae.W_dec.norm(dim=1)

    # not unit norms
    assert not torch.allclose(
        decoder_norms, torch.ones_like(sae.W_dec.norm(dim=1)), atol=1e-6
    )

    assert torch.allclose(
        decoder_norms, torch.ones_like(decoder_norms) * 0.1, atol=5e-2
    )
