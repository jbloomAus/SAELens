import os
from pathlib import Path
from typing import Any

import einops
import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformer_lens import HookedTransformer

from sae_lens import __version__
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.sparse_autoencoder import (
    SparseAutoencoder,
    _per_item_mse_loss_with_target_norm,
)
from tests.unit.helpers import build_sae_cfg


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "tokenized": False,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_point": "blocks.1.attn.hook_z",
            "hook_point_layer": 1,
            "d_in": 64,
        },
    ],
    ids=[
        "tiny-stories-1M-resid-pre",
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


@pytest.fixture
def sparse_autoencoder(cfg: Any):
    """
    Pytest fixture to create a mock instance of SparseAutoencoder.
    """
    return SparseAutoencoder(cfg)


@pytest.fixture
def model(cfg: LanguageModelSAERunnerConfig):
    return HookedTransformer.from_pretrained(cfg.model_name, device="cpu")


def test_sparse_autoencoder_init(cfg: Any):
    sparse_autoencoder = SparseAutoencoder(cfg)

    assert isinstance(sparse_autoencoder, SparseAutoencoder)

    assert sparse_autoencoder.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sparse_autoencoder.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sparse_autoencoder.b_enc.shape == (cfg.d_sae,)
    assert sparse_autoencoder.b_dec.shape == (cfg.d_in,)

    # assert decoder columns have unit norm
    assert torch.allclose(
        torch.norm(sparse_autoencoder.W_dec, dim=1), torch.ones(cfg.d_sae)
    )


def test_SparseAutoencoder_save_and_load_from_pretrained(tmp_path: Path) -> None:
    cfg = build_sae_cfg(device="cpu")
    model_path = str(tmp_path)
    sparse_autoencoder = SparseAutoencoder(cfg)
    sparse_autoencoder_state_dict = sparse_autoencoder.state_dict()
    sparse_autoencoder.save_model(model_path)

    assert os.path.exists(model_path)

    sparse_autoencoder_loaded = SparseAutoencoder.load_from_pretrained(
        model_path, device="cpu"
    )
    sparse_autoencoder_loaded.cfg.device = "cpu"
    sparse_autoencoder_loaded.cfg.verbose = True
    sparse_autoencoder_loaded.cfg.checkpoint_path = cfg.checkpoint_path
    sparse_autoencoder_loaded_state_dict = sparse_autoencoder_loaded.state_dict()
    # check cfg matches the original
    assert sparse_autoencoder_loaded.cfg == cfg

    # check state_dict matches the original
    for key in sparse_autoencoder.state_dict().keys():
        assert torch.allclose(
            sparse_autoencoder_state_dict[key],
            sparse_autoencoder_loaded_state_dict[key],
        )


def test_SparseAutoencoder_save_and_load_from_pretrained_update_the_sae_lens_version_on_load(
    tmp_path: Path,
):
    cfg = build_sae_cfg(
        device="cpu", sae_lens_training_version="9999.0.0", sae_lens_version="9999.0.0"
    )
    model_path = str(tmp_path)
    sparse_autoencoder = SparseAutoencoder(cfg)
    sparse_autoencoder.save_model(model_path)

    assert os.path.exists(model_path)

    sparse_autoencoder_loaded = SparseAutoencoder.load_from_pretrained(
        model_path, device="cpu"
    )
    # the sae_lens_version should update, but not the training version unless we actually run training
    assert sparse_autoencoder_loaded.cfg.sae_lens_version == __version__
    assert sparse_autoencoder_loaded.cfg.sae_lens_training_version == "9999.0.0"


def test_SparseAutoencoder_save_and_load_from_pretrained_lacks_scaling_factor(
    tmp_path: Path,
) -> None:
    cfg = build_sae_cfg(device="cpu")
    model_path = str(tmp_path)
    sparse_autoencoder = SparseAutoencoder(cfg)
    sparse_autoencoder_state_dict = sparse_autoencoder.state_dict()
    # sometimes old state dicts will be missing the scaling factor
    del sparse_autoencoder_state_dict["scaling_factor"]  # = torch.tensor(0.0)
    sparse_autoencoder.save_model(model_path)

    assert os.path.exists(model_path)

    sparse_autoencoder_loaded = SparseAutoencoder.load_from_pretrained(model_path)
    sparse_autoencoder_loaded.cfg.verbose = True
    sparse_autoencoder_loaded.cfg.checkpoint_path = cfg.checkpoint_path
    sparse_autoencoder_loaded.cfg.device = "cpu"  # might autoload onto mps
    sparse_autoencoder_loaded = sparse_autoencoder_loaded.to("cpu")
    sparse_autoencoder_loaded_state_dict = sparse_autoencoder_loaded.state_dict()
    # check cfg matches the original
    assert sparse_autoencoder_loaded.cfg == cfg

    # check state_dict matches the original
    for key in sparse_autoencoder.state_dict().keys():
        if key == "scaling_factor":
            assert isinstance(cfg.d_sae, int)
            assert torch.allclose(
                torch.ones(cfg.d_sae, dtype=cfg.dtype, device=cfg.device),
                sparse_autoencoder_loaded_state_dict[key],
            )
        else:
            assert torch.allclose(
                sparse_autoencoder_state_dict[key],
                sparse_autoencoder_loaded_state_dict[key],
            )


def test_sparse_autoencoder_encode(sparse_autoencoder: SparseAutoencoder):
    batch_size = 32
    d_in = sparse_autoencoder.d_in
    d_sae = sparse_autoencoder.d_sae

    x = torch.randn(batch_size, d_in)
    feature_acts1 = sparse_autoencoder.encode(x)
    (
        _,
        feature_acts2,
        _,
        _,
        _,
        _,
    ) = sparse_autoencoder.forward(
        x,
    )

    # Check shape
    assert feature_acts1.shape == (batch_size, d_sae)

    # Check values
    assert torch.allclose(feature_acts1, feature_acts2)


def test_sparse_autoencoder_decode(sparse_autoencoder: SparseAutoencoder):
    batch_size = 32
    d_in = sparse_autoencoder.d_in

    x = torch.randn(batch_size, d_in)
    feature_acts = sparse_autoencoder.encode(x)
    sae_out1 = sparse_autoencoder.decode(feature_acts)

    (
        sae_out2,
        _,
        _,
        _,
        _,
        _,
    ) = sparse_autoencoder.forward(
        x,
    )

    assert sae_out1.shape == x.shape
    assert torch.allclose(sae_out1, sae_out2)


def test_sparse_autoencoder_forward(sparse_autoencoder: SparseAutoencoder):
    batch_size = 32
    d_in = sparse_autoencoder.d_in
    d_sae = sparse_autoencoder.d_sae

    x = torch.randn(batch_size, d_in)
    (
        sae_out,
        feature_acts,
        loss,
        mse_loss,
        l1_loss,
        _ghost_grad_loss,
    ) = sparse_autoencoder.forward(
        x,
    )

    assert sae_out.shape == (batch_size, d_in)
    assert feature_acts.shape == (batch_size, d_sae)
    assert loss.shape == ()
    assert mse_loss.shape == ()
    assert l1_loss.shape == ()
    assert torch.allclose(loss, mse_loss + l1_loss)

    expected_mse_loss = (torch.pow((sae_out - x.float()), 2)).mean()

    assert torch.allclose(mse_loss, expected_mse_loss)
    expected_l1_loss = torch.abs(feature_acts).sum(dim=1).mean(dim=(0,))
    assert torch.allclose(l1_loss, sparse_autoencoder.l1_coefficient * expected_l1_loss)

    # check everything has the right dtype
    assert sae_out.dtype == sparse_autoencoder.dtype
    assert feature_acts.dtype == sparse_autoencoder.dtype
    assert loss.dtype == sparse_autoencoder.dtype
    assert mse_loss.dtype == sparse_autoencoder.dtype
    assert l1_loss.dtype == sparse_autoencoder.dtype


def test_sparse_autoencoder_forward_with_mse_loss_norm(
    sparse_autoencoder: SparseAutoencoder,
):
    batch_size = 32
    d_in = sparse_autoencoder.d_in
    d_sae = sparse_autoencoder.d_sae
    sparse_autoencoder.cfg.mse_loss_normalization = "dense_batch"

    x = torch.randn(batch_size, d_in)
    (
        sae_out,
        feature_acts,
        loss,
        mse_loss,
        l1_loss,
        _ghost_grad_loss,
    ) = sparse_autoencoder.forward(
        x,
    )

    assert sae_out.shape == (batch_size, d_in)
    assert feature_acts.shape == (batch_size, d_sae)
    assert loss.shape == ()
    assert mse_loss.shape == ()
    assert l1_loss.shape == ()
    assert torch.allclose(loss, mse_loss + l1_loss)

    x_centred = x - x.mean(dim=0, keepdim=True)
    expected_mse_loss = (
        torch.pow((sae_out - x.float()), 2)
        / (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
    ).mean()
    assert torch.allclose(mse_loss, expected_mse_loss)
    expected_l1_loss = torch.abs(feature_acts).sum(dim=1).mean(dim=(0,))
    assert torch.allclose(l1_loss, sparse_autoencoder.l1_coefficient * expected_l1_loss)

    # check everything has the right dtype
    assert sae_out.dtype == sparse_autoencoder.dtype
    assert feature_acts.dtype == sparse_autoencoder.dtype
    assert loss.dtype == sparse_autoencoder.dtype
    assert mse_loss.dtype == sparse_autoencoder.dtype
    assert l1_loss.dtype == sparse_autoencoder.dtype


def test_SparseAutoencoder_forward_ghost_grad_loss_returns_0_if_no_dead_neurons():
    cfg = build_sae_cfg(d_in=2, d_sae=4, use_ghost_grads=True)
    sae = SparseAutoencoder(cfg)
    sae.train()
    dead_neuron_mask = torch.tensor([False, False, False, False])
    ghost_grad_loss = sae.forward(
        x=torch.randn(3, 2),
        dead_neuron_mask=dead_neuron_mask,
    ).ghost_grad_loss
    assert ghost_grad_loss == 0.0


def test_SparseAutoencoder_forward_ghost_grad_loss_only_adds_gradients_to_dead_neurons():
    cfg = build_sae_cfg(d_in=2, d_sae=4, use_ghost_grads=True)
    sae = SparseAutoencoder(cfg)
    sae.train()
    dead_neuron_mask = torch.tensor([False, True, False, True])
    forward_out = sae.forward(
        x=torch.randn(3, 2),
        dead_neuron_mask=dead_neuron_mask,
    )
    forward_out.ghost_grad_loss.backward()

    # only features 1 and 3 should have non-zero gradients on the encoder weights
    assert sae.W_enc.grad is not None
    assert torch.allclose(sae.W_enc.grad[:, 0], torch.zeros_like(sae.W_enc[:, 0]))
    assert sae.W_enc.grad[:, 1].abs().sum() > 0.001
    assert torch.allclose(sae.W_enc.grad[:, 2], torch.zeros_like(sae.W_enc[:, 2]))
    assert sae.W_enc.grad[:, 3].abs().sum() > 0.001

    # only features 1 and 3 should have non-zero gradients on the decoder weights
    assert sae.W_dec.grad is not None
    assert torch.allclose(sae.W_dec.grad[0, :], torch.zeros_like(sae.W_dec[0, :]))
    assert sae.W_dec.grad[1, :].abs().sum() > 0.001
    assert torch.allclose(sae.W_dec.grad[2, :], torch.zeros_like(sae.W_dec[2, :]))
    assert sae.W_dec.grad[3, :].abs().sum() > 0.001


def test_per_item_mse_loss_with_norm_matches_original_implementation() -> None:
    input = torch.randn(3, 2)
    target = torch.randn(3, 2)
    target_centered = target - target.mean(dim=0, keepdim=True)
    orig_impl_res = (
        torch.pow((input - target.float()), 2)
        / (target_centered**2).sum(dim=-1, keepdim=True).sqrt()
    )
    sae_res = _per_item_mse_loss_with_target_norm(
        input, target, mse_loss_normalization="dense_batch"
    )
    assert torch.allclose(orig_impl_res, sae_res, atol=1e-5)


def test_SparseAutoencoder_forward_can_add_noise_to_hidden_pre() -> None:
    clean_cfg = build_sae_cfg(d_in=2, d_sae=4, noise_scale=0)
    noisy_cfg = build_sae_cfg(d_in=2, d_sae=4, noise_scale=100)
    clean_sae = SparseAutoencoder(clean_cfg)
    noisy_sae = SparseAutoencoder(noisy_cfg)

    input = torch.randn(3, 2)

    clean_output1 = clean_sae.forward(input).sae_out
    clean_output2 = clean_sae.forward(input).sae_out
    noisy_output1 = noisy_sae.forward(input).sae_out
    noisy_output2 = noisy_sae.forward(input).sae_out

    # with no noise, the outputs should be identical
    assert torch.allclose(clean_output1, clean_output2)
    # noisy outputs should be different
    assert not torch.allclose(noisy_output1, noisy_output2)
    assert not torch.allclose(clean_output1, noisy_output1)


def test_SparseAutoencoder_remove_gradient_parallel_to_decoder_directions() -> None:
    cfg = build_sae_cfg(normalize_sae_decoder=True)
    sae = SparseAutoencoder(cfg)
    orig_grad = torch.randn_like(sae.W_dec)
    orig_W_dec = sae.W_dec.clone()
    sae.W_dec.grad = orig_grad.clone()
    sae.remove_gradient_parallel_to_decoder_directions()

    # check that the gradient is orthogonal to the decoder directions
    parallel_component = einops.einsum(
        sae.W_dec.grad,
        sae.W_dec,
        "d_sae d_in, d_sae d_in -> d_sae",
    )
    assert torch.allclose(
        parallel_component, torch.zeros_like(parallel_component), atol=1e-5
    )
    # the decoder weights should not have changed
    assert torch.allclose(sae.W_dec, orig_W_dec)
    # the gradient delta should align with the decoder directions
    grad_delta = orig_grad - sae.W_dec.grad
    assert torch.nn.functional.cosine_similarity(
        sae.W_dec.detach(), grad_delta, dim=1
    ).abs() == pytest.approx(1.0, abs=1e-3)


def test_SparseAutoencoder_get_name_returns_correct_name_from_cfg_vals() -> None:
    cfg = build_sae_cfg(model_name="test_model", hook_point="test_hook_point", d_sae=10)
    sae = SparseAutoencoder(cfg)
    assert sae.get_name() == "sparse_autoencoder_test_model_test_hook_point_10"


def test_SparseAutoencoder_set_decoder_norm_to_unit_norm() -> None:
    cfg = build_sae_cfg(normalize_sae_decoder=True)
    sae = SparseAutoencoder(cfg)
    sae.W_dec.data = 20 * torch.randn_like(sae.W_dec)
    sae.set_decoder_norm_to_unit_norm()
    assert torch.allclose(
        torch.norm(sae.W_dec, dim=1), torch.ones_like(sae.W_dec[:, 0])
    )


def test_SparseAutoencoder_from_pretrained_loads_from_hugginface_using_shorthand():
    sae = SparseAutoencoder.from_pretrained(
        release="gpt2-small-res-jb",
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
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

    assert isinstance(sae, SparseAutoencoder)
    assert sae.cfg.model_name == "gpt2-small"
    assert sae.cfg.hook_point == "blocks.0.hook_resid_pre"

    for k in sae.state_dict().keys():
        if k == "scaling_factor":
            continue
        assert torch.allclose(sae.state_dict()[k], state_dict[k])


def test_SparseAutoencoder_from_pretrained_errors_for_invalid_releases():
    with pytest.raises(ValueError):
        SparseAutoencoder.from_pretrained(
            release="wrong",
            sae_id="blocks.0.hook_resid_pre",
            device="cpu",
        )


def test_SparseAutoencoder_from_pretrained_errors_for_invalid_sae_ids():
    with pytest.raises(ValueError):
        SparseAutoencoder.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="wrong",
            device="cpu",
        )
