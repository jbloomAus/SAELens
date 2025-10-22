import pytest
import torch

from sae_lens.saes.temporal_sae import TemporalSAE
from tests.helpers import build_temporal_sae_cfg


def test_TemporalSAE_initialization():
    cfg = build_temporal_sae_cfg()
    sae = TemporalSAE.from_dict(cfg.to_dict())

    assert isinstance(sae.W_Dec, torch.nn.Parameter)
    assert isinstance(sae.b_dec, torch.nn.Parameter)
    assert sae.W_Dec.shape == (cfg.d_sae, cfg.d_in)
    assert sae.b_dec.shape == (cfg.d_in,)

    if cfg.tied_weights:
        assert sae.W_Enc is None
    else:
        assert isinstance(sae.W_Enc, torch.nn.Parameter)
        assert sae.W_Enc.shape == (cfg.d_in, cfg.d_sae)

    assert len(sae.attn_layers) == cfg.n_attn_layers
    for attn_layer in sae.attn_layers:
        assert attn_layer.n_heads == cfg.n_heads
        assert attn_layer.dimin == cfg.d_sae


@pytest.mark.parametrize("tied_weights", [True, False])
def test_TemporalSAE_forward(tied_weights: bool):
    cfg = build_temporal_sae_cfg(tied_weights=tied_weights)
    sae = TemporalSAE.from_dict(cfg.to_dict())

    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, cfg.d_in)

    reconstruction = sae.forward(x)

    assert reconstruction.shape == x.shape


@pytest.mark.parametrize("tied_weights", [True, False])
def test_TemporalSAE_encode(tied_weights: bool):
    cfg = build_temporal_sae_cfg(
        tied_weights=tied_weights, sae_diff_type="topk", kval_topk=32
    )
    sae = TemporalSAE.from_dict(cfg.to_dict())

    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, cfg.d_in)

    novel_codes = sae.encode(x)

    assert novel_codes.shape == (batch_size, seq_len, cfg.d_sae)
    assert (novel_codes >= 0).all()
    assert cfg.kval_topk is not None
    l0 = (novel_codes != 0).sum(dim=-1).float().mean().item()
    assert l0 <= cfg.kval_topk


def test_TemporalSAE_decode():
    cfg = build_temporal_sae_cfg()
    sae = TemporalSAE.from_dict(cfg.to_dict())

    batch_size = 4
    seq_len = 16
    novel_codes = torch.randn(batch_size, seq_len, cfg.d_sae).relu()

    reconstruction = sae.decode(novel_codes)

    assert reconstruction.shape == (batch_size, seq_len, cfg.d_in)


@pytest.mark.parametrize("sae_diff_type", ["relu", "topk"])
def test_TemporalSAE_sae_diff_type(sae_diff_type: str):
    cfg = build_temporal_sae_cfg(
        sae_diff_type=sae_diff_type,
        kval_topk=32 if sae_diff_type == "topk" else None,
    )
    sae = TemporalSAE.from_dict(cfg.to_dict())

    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, cfg.d_in)

    novel_codes = sae.encode(x)

    assert novel_codes.shape == (batch_size, seq_len, cfg.d_sae)
    assert (novel_codes >= 0).all()

    if sae_diff_type == "topk":
        assert cfg.kval_topk is not None
        l0 = (novel_codes != 0).sum(dim=-1).float()
        assert (l0 <= cfg.kval_topk).all()


@pytest.mark.parametrize("n_attn_layers", [1, 2])
def test_TemporalSAE_n_attn_layers(n_attn_layers: int):
    cfg = build_temporal_sae_cfg(n_attn_layers=n_attn_layers)
    sae = TemporalSAE.from_dict(cfg.to_dict())

    assert len(sae.attn_layers) == n_attn_layers

    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, cfg.d_in)

    reconstruction = sae.forward(x)
    assert reconstruction.shape == x.shape
