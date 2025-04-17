import pytest
import torch
from sparsify import SparseCoder, SparseCoderConfig

from sae_lens.saes.sae_base import TrainingSAEConfig
from sae_lens.saes.topk_sae import TopKTrainingSAE
from tests.helpers import build_sae_cfg


def test_TopKTrainingSAE_topk_aux_loss_matches_unnormalized_sparsify_implementation():
    d_in = 128
    d_sae = 192
    cfg = build_sae_cfg(
        d_in=d_in,
        d_sae=d_sae,
        architecture="topk",
        activation_fn_kwargs={"k": 26},
        normalize_sae_decoder=False,
    )

    sae = TopKTrainingSAE(TrainingSAEConfig.from_sae_runner_config(cfg))
    sparse_coder_sae = SparseCoder(
        d_in=d_in, cfg=SparseCoderConfig(num_latents=d_sae, k=26)
    )

    with torch.no_grad():
        # increase b_enc so all features are likely above 0
        # sparsify includes a relu() in their pre_acts, but
        # this is not something we need to try to replicate.
        sae.b_enc.data = sae.b_enc + 100.0
        # make sure all params are the same
        sparse_coder_sae.encoder.weight.data = sae.W_enc.T
        sparse_coder_sae.encoder.bias.data = sae.b_enc
        sparse_coder_sae.b_dec.data = sae.b_dec
        sparse_coder_sae.W_dec.data = sae.W_dec  # type: ignore

    dead_neuron_mask = torch.randn(d_sae) > 0.1
    input_acts = torch.randn(200, d_in)
    input_var = (input_acts - input_acts.mean(0)).pow(2).sum()

    sae_out = sae.training_forward_pass(
        sae_in=input_acts,
        current_l1_coefficient=0.0,
        dead_neuron_mask=dead_neuron_mask,
    )
    comparison_sae_out = sparse_coder_sae.forward(
        input_acts, dead_mask=dead_neuron_mask
    )
    comparison_aux_loss = comparison_sae_out.auxk_loss.detach().item()

    normalization = input_var / input_acts.shape[0]
    raw_aux_loss = sae_out.losses["auxiliary_reconstruction_loss"].item()  # type: ignore
    norm_aux_loss = raw_aux_loss / normalization
    assert norm_aux_loss == pytest.approx(comparison_aux_loss, abs=1e-4)
