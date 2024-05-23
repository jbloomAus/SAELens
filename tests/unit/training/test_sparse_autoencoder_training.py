from typing import Any

import einops
import pytest
import torch
from transformer_lens import HookedTransformer

from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.sparse_autoencoder import TrainingSparseAutoencoder
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
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
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


@pytest.fixture
def training_sae(cfg: Any):
    """
    Pytest fixture to create a mock instance of SparseAutoencoder.
    """
    return TrainingSparseAutoencoder(cfg)


@pytest.fixture
def model(cfg: LanguageModelSAERunnerConfig):
    return HookedTransformer.from_pretrained(cfg.model_name, device="cpu")


def test_sparse_autoencoder_encode(training_sae: TrainingSparseAutoencoder):
    batch_size = 32
    d_in = training_sae.d_in
    d_sae = training_sae.d_sae

    x = torch.randn(batch_size, d_in)
    feature_acts1 = training_sae.encode(x)
    (
        _,
        feature_acts2,
        _,
        _,
        _,
        _,
    ) = training_sae.forward(
        x,
    )

    # Check shape
    assert feature_acts1.shape == (batch_size, d_sae)

    # Check values
    assert torch.allclose(feature_acts1, feature_acts2)


def test_sparse_autoencoder_decode(training_sae: TrainingSparseAutoencoder):
    batch_size = 32
    d_in = training_sae.d_in

    x = torch.randn(batch_size, d_in)
    feature_acts = training_sae.encode(x)
    sae_out1 = training_sae.decode(feature_acts)

    (
        sae_out2,
        _,
        _,
        _,
        _,
        _,
    ) = training_sae.forward(
        x,
    )

    assert sae_out1.shape == x.shape
    assert torch.allclose(sae_out1, sae_out2)


def test_sparse_autoencoder_forward(training_sae: TrainingSparseAutoencoder):
    batch_size = 32
    d_in = training_sae.d_in
    d_sae = training_sae.d_sae

    x = torch.randn(batch_size, d_in)
    (
        sae_out,
        feature_acts,
        loss,
        mse_loss,
        l1_loss,
        _ghost_grad_loss,
    ) = training_sae.forward(
        x,
    )

    assert sae_out.shape == (batch_size, d_in)
    assert feature_acts.shape == (batch_size, d_sae)
    assert loss.shape == ()
    assert mse_loss.shape == ()
    assert l1_loss.shape == ()
    assert torch.allclose(loss, mse_loss + l1_loss)

    expected_mse_loss = (torch.pow((sae_out - x.float()), 2)).sum(dim=-1).mean()

    assert torch.allclose(mse_loss, expected_mse_loss)
    if not training_sae.scale_sparsity_penalty_by_decoder_norm:
        expected_l1_loss = feature_acts.sum(dim=1).mean(dim=(0,))
    else:
        expected_l1_loss = (
            (feature_acts * training_sae.W_dec.norm(dim=1)).norm(dim=1, p=1).mean()
        )
    assert torch.allclose(l1_loss, training_sae.l1_coefficient * expected_l1_loss)

    # check everything has the right dtype
    assert sae_out.dtype == training_sae.dtype
    assert feature_acts.dtype == training_sae.dtype
    assert loss.dtype == training_sae.dtype
    assert mse_loss.dtype == training_sae.dtype
    assert l1_loss.dtype == training_sae.dtype


def test_sparse_autoencoder_forward_with_mse_loss_norm(
    training_sae: TrainingSparseAutoencoder,
):
    batch_size = 32
    d_in = training_sae.d_in
    d_sae = training_sae.d_sae
    training_sae.mse_loss_normalization = "dense_batch"

    x = torch.randn(batch_size, d_in)
    (
        sae_out,
        feature_acts,
        loss,
        mse_loss,
        l1_loss,
        _ghost_grad_loss,
    ) = training_sae.forward(
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
        (
            torch.pow((sae_out - x.float()), 2)
            / (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
        )
        .sum(dim=-1)
        .mean()
    )
    assert torch.allclose(mse_loss, expected_mse_loss)
    if not training_sae.scale_sparsity_penalty_by_decoder_norm:
        expected_l1_loss = feature_acts.sum(dim=1).mean(dim=(0,))
    else:
        expected_l1_loss = (
            (feature_acts * training_sae.W_dec.norm(dim=1)).norm(dim=1, p=1).mean()
        )
    assert torch.allclose(l1_loss, training_sae.l1_coefficient * expected_l1_loss)

    # check everything has the right dtype
    assert sae_out.dtype == training_sae.dtype
    assert feature_acts.dtype == training_sae.dtype
    assert loss.dtype == training_sae.dtype
    assert mse_loss.dtype == training_sae.dtype
    assert l1_loss.dtype == training_sae.dtype


def test_sparse_autoencoder_forward_with_3d_input(
    training_sae: TrainingSparseAutoencoder,
):
    batch_size = 32
    seq_length = 256
    d_in = training_sae.d_in
    d_sae = training_sae.d_sae
    training_sae.mse_loss_normalization = "dense_batch"
    training_sae.lp_norm = 1

    x = torch.randn(batch_size, seq_length, d_in)
    (
        sae_out,
        feature_acts,
        loss,
        mse_loss,
        l1_loss,
        _,
    ) = training_sae.forward(
        x,
    )
    training_sae.lp_norm = 2
    (
        _,
        _,
        _,
        _,
        l2_loss,
        _,
    ) = training_sae.forward(
        x,
    )

    assert sae_out.shape == (batch_size, seq_length, d_in)
    assert feature_acts.shape == (batch_size, seq_length, d_sae)
    assert loss.shape == ()
    assert mse_loss.shape == ()
    assert l1_loss.shape == ()
    assert torch.allclose(loss, mse_loss + l1_loss)

    x_centred = x - x.mean(dim=0, keepdim=True)
    expected_mse_loss = (
        (
            torch.pow((sae_out - x.float()), 2)
            / (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
        )
        .sum(dim=-1)
        .mean()
    )
    assert torch.allclose(mse_loss, expected_mse_loss)

    if training_sae.scale_sparsity_penalty_by_decoder_norm:
        feature_acts = feature_acts * training_sae.W_dec.norm(dim=1)

    expected_l1_loss = feature_acts.sum(dim=-1).mean()
    assert torch.allclose(l1_loss, training_sae.l1_coefficient * expected_l1_loss)

    expected_l2_loss = feature_acts.norm(p=2, dim=-1).mean()
    assert torch.allclose(l2_loss, training_sae.l1_coefficient * expected_l2_loss)

    # check everything has the right dtype
    assert sae_out.dtype == training_sae.dtype
    assert feature_acts.dtype == training_sae.dtype
    assert loss.dtype == training_sae.dtype
    assert mse_loss.dtype == training_sae.dtype
    assert l1_loss.dtype == training_sae.dtype


def test_SparseAutoencoder_forward_ghost_grad_loss_returns_0_if_no_dead_neurons():
    cfg = build_sae_cfg(d_in=2, d_sae=4, use_ghost_grads=True)
    sae = TrainingSparseAutoencoder(cfg)
    sae.train()
    dead_neuron_mask = torch.tensor([False, False, False, False])
    ghost_grad_loss = sae.forward(
        x=torch.randn(3, 2),
        dead_neuron_mask=dead_neuron_mask,
    ).ghost_grad_loss
    assert ghost_grad_loss == 0.0


def test_SparseAutoencoder_forward_ghost_grad_loss_only_adds_gradients_to_dead_neurons():
    cfg = build_sae_cfg(d_in=2, d_sae=4, use_ghost_grads=True)
    sae = TrainingSparseAutoencoder(cfg)
    sae.train()
    dead_neuron_mask = torch.tensor([False, True, False, True])
    forward_out = sae.forward(
        x=torch.randn(3, 2),
        dead_neuron_mask=dead_neuron_mask,
    )
    forward_out.ghost_grad_loss.backward()  # type: ignore

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


def test_per_item_mse_loss_with_norm_matches_original_implementation(
    training_sae: TrainingSparseAutoencoder,
) -> None:
    input = torch.randn(3, 2)
    target = torch.randn(3, 2)
    target_centered = target - target.mean(dim=0, keepdim=True)
    orig_impl_res = (
        torch.pow((input - target.float()), 2)
        / (target_centered**2).sum(dim=-1, keepdim=True).sqrt()
    )
    sae_res = training_sae._per_item_mse_loss_with_target_norm(
        input, target, mse_loss_normalization="dense_batch"
    )
    assert torch.allclose(orig_impl_res, sae_res, atol=1e-5)


def test_SparseAutoencoder_forward_can_add_noise_to_hidden_pre() -> None:
    clean_cfg = build_sae_cfg(d_in=2, d_sae=4, noise_scale=0)
    noisy_cfg = build_sae_cfg(d_in=2, d_sae=4, noise_scale=100)
    clean_sae = TrainingSparseAutoencoder(clean_cfg)
    noisy_sae = TrainingSparseAutoencoder(noisy_cfg)

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
    sae = TrainingSparseAutoencoder(cfg)
    orig_grad = torch.randn_like(sae.W_dec)
    orig_W_dec = sae.W_dec.clone()
    sae.W_dec.grad = orig_grad.clone()
    sae.remove_gradient_parallel_to_decoder_directions()

    # check that the gradient is orthogonal to the decoder directions
    parallel_component = einops.einsum(
        sae.W_dec.grad,
        sae.W_dec.data,
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


def test_SparseAutoencoder_set_decoder_norm_to_unit_norm() -> None:
    cfg = build_sae_cfg(normalize_sae_decoder=True)
    sae = TrainingSparseAutoencoder(cfg)
    sae.W_dec.data = 20 * torch.randn_like(sae.W_dec)
    sae.set_decoder_norm_to_unit_norm()
    assert torch.allclose(
        torch.norm(sae.W_dec, dim=1), torch.ones_like(sae.W_dec[:, 0])
    )
