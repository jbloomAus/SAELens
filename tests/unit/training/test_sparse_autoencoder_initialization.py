import pytest
import torch

from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from tests.unit.helpers import build_sae_cfg


def test_SparseAutoencoder_initialization_standard():
    cfg = build_sae_cfg()

    sae = SparseAutoencoder(cfg)
    assert sae.cfg == cfg
    assert sae.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sae.b_enc.shape == (cfg.d_sae,)
    assert sae.b_dec.shape == (cfg.d_in,)
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
    cfg = build_sae_cfg(decoder_orthogonal_init=True)

    sae = SparseAutoencoder(cfg)
    projections = sae.W_dec.T @ sae.W_dec
    mask = ~torch.eye(projections.size(0), dtype=torch.bool)

    assert projections[mask].max() < 0.1
    assert sae.cfg == cfg

    # initialized weights of biases are 0
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)


def test_SparseAutoencoder_initialization_normalize_decoder_norm():
    cfg = build_sae_cfg(normalize_sae_decoder=True)

    sae = SparseAutoencoder(cfg)

    assert sae.cfg == cfg

    assert torch.allclose(
        sae.W_dec.norm(dim=1), torch.ones_like(sae.W_dec.norm(dim=1)), atol=1e-6
    )

    # initialized weights of biases are 0
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)


def test_SparseAutoencoder_initialization_encoder_is_decoder_transpose():
    cfg = build_sae_cfg(init_encoder_as_decoder_transpose=True)

    sae = SparseAutoencoder(cfg)

    assert sae.cfg == cfg

    # If we decoder norms are 1 we need to unit norm W_enc first.
    unit_normed_W_enc = sae.W_enc / torch.norm(sae.W_enc, dim=0)
    unit_normed_W_dec = sae.W_dec.T
    assert torch.allclose(unit_normed_W_enc, unit_normed_W_dec, atol=1e-6)

    # initialized weights of biases are 0
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)


def test_SparseAutoencoder_initialization_enc_dec_T_no_unit_norm():
    cfg = build_sae_cfg(
        init_encoder_as_decoder_transpose=True,
        normalize_sae_decoder=False,
    )

    sae = SparseAutoencoder(cfg)

    assert sae.cfg == cfg

    assert torch.allclose(sae.W_dec, sae.W_enc.T, atol=1e-6)

    # initialized weights of biases are 0
    assert torch.allclose(sae.b_dec, torch.zeros_like(sae.b_dec), atol=1e-6)
    assert torch.allclose(sae.b_enc, torch.zeros_like(sae.b_enc), atol=1e-6)


def test_SparseAutoencoder_initialization_heuristic_init_and_normalize_sae_decoder():

    # assert that an error is raised
    with pytest.raises(ValueError):
        _ = build_sae_cfg(
            decoder_heuristic_init=True,
            normalize_sae_decoder=True,
        )


def test_SparseAutoencoder_initialization_decoder_norm_in_loss_and_normalize_sae_decoder():

    # assert that an error is raised
    with pytest.raises(ValueError):
        _ = build_sae_cfg(
            scale_sparsity_penalty_by_decoder_norm=True,
            normalize_sae_decoder=True,
        )


def test_SparseAutoencoder_initialization_heuristic_init():
    cfg = build_sae_cfg(
        decoder_heuristic_init=True,
        normalize_sae_decoder=False,
    )

    sae = SparseAutoencoder(cfg)

    decoder_norms = sae.W_dec.norm(dim=1)

    # not unit norms
    assert not torch.allclose(
        decoder_norms, torch.ones_like(sae.W_dec.norm(dim=1)), atol=1e-6
    )

    # check that the mean is close to 0.1
    assert torch.allclose(decoder_norms.mean(), torch.tensor(0.1), atol=5e-2)

    # check that the min isn't less than 0.05
    assert decoder_norms.min().item() >= 0.0499

    # check that the max is less than 1
    assert decoder_norms.max() < 1.001
