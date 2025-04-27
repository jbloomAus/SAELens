import os
from pathlib import Path

import pytest
import torch
from sparsify import SparseCoder, SparseCoderConfig

from sae_lens.saes.sae import SAE, TrainStepInput
from sae_lens.saes.topk_sae import TopKSAE, TopKTrainingSAE, TopKTrainingSAEConfig
from tests.helpers import (
    build_topk_sae_cfg,
    build_topk_sae_training_cfg,
)


def test_TopKTrainingSAE_topk_aux_loss_matches_unnormalized_sparsify_implementation():
    d_in = 128
    d_sae = 192
    cfg = build_topk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=26,
    )

    sae = TopKTrainingSAE(TopKTrainingSAEConfig.from_sae_runner_config(cfg))
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
        step_input=TrainStepInput(
            sae_in=input_acts,
            dead_neuron_mask=dead_neuron_mask,
            coefficients={},
        ),
    )
    comparison_sae_out = sparse_coder_sae.forward(
        input_acts, dead_mask=dead_neuron_mask
    )
    comparison_aux_loss = comparison_sae_out.auxk_loss.detach().item()

    normalization = input_var / input_acts.shape[0]
    raw_aux_loss = sae_out.losses["auxiliary_reconstruction_loss"].item()  # type: ignore
    norm_aux_loss = raw_aux_loss / normalization
    assert norm_aux_loss == pytest.approx(comparison_aux_loss, abs=1e-2)


def test_sae_save_and_load_from_pretrained_topk(tmp_path: Path) -> None:
    cfg = build_topk_sae_cfg(k=30)
    model_path = str(tmp_path)
    sae = TopKSAE(cfg)
    sae_state_dict = sae.state_dict()
    sae.save_model(model_path)

    assert os.path.exists(model_path)

    sae_loaded = SAE.load_from_pretrained(model_path, device="cpu")
    assert isinstance(sae_loaded, TopKSAE)

    sae_loaded_state_dict = sae_loaded.state_dict()

    # check state_dict matches the original
    for key in sae.state_dict():
        assert torch.allclose(
            sae_state_dict[key],
            sae_loaded_state_dict[key],
        )

    sae_in = torch.randn(10, cfg.d_in, device=cfg.device)
    sae_out_1 = sae(sae_in)
    sae_out_2 = sae_loaded(sae_in)
    assert torch.allclose(sae_out_1, sae_out_2)
