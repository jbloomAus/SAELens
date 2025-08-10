import torch
from circuit_tracer.transcoder.cross_layer_transcoder import CrossLayerTranscoder

from sae_lens.saes.clt import StandardCLT, StandardCLTConfig
from tests.helpers import random_init_params


@torch.no_grad()
def create_matching_circuit_tracer_clt(clt: StandardCLT) -> CrossLayerTranscoder:
    assert clt.cfg.d_in == clt.cfg.d_out, "d_in and d_out must be the same"
    ct_clt = CrossLayerTranscoder(
        d_model=clt.cfg.d_in,
        d_transcoder=clt.cfg.d_clt,
        n_layers=clt.cfg.n_layers,
        dtype=torch.float32,
        lazy_decoder=False,
        lazy_encoder=False,
    )
    assert ct_clt.W_enc.shape == clt.W_enc.permute(0, 2, 1).shape
    ct_clt.W_enc.data = clt.W_enc.permute(0, 2, 1).clone()
    assert ct_clt.W_dec is not None
    assert len(ct_clt.W_dec) == len(clt.W_dec)
    for ct_decoder, decoder in zip(ct_clt.W_dec, clt.W_dec):
        assert ct_decoder.shape == decoder.shape
        ct_decoder.data = decoder.clone()
    assert ct_clt.b_enc.shape == clt.b_enc.shape
    ct_clt.b_enc.data = clt.b_enc.clone()
    assert ct_clt.b_dec.shape == clt.b_dec.shape
    ct_clt.b_dec.data = clt.b_dec.clone()
    return ct_clt


def test_clt_forward_matches_circuit_tracer():
    clt_cfg = StandardCLTConfig(
        d_in=64,
        d_out=64,
        d_sae=128,
        n_layers=4,
        apply_b_dec_to_input=False,
    )
    clt = StandardCLT(clt_cfg)
    random_init_params(clt)
    ct_clt = create_matching_circuit_tracer_clt(clt)
    assert ct_clt.forward(torch.randn(4, 2, 64)) == clt.forward(torch.randn(4, 2, 64))
