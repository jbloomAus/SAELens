import os
from pathlib import Path

import numpy as np
import pytest
import torch
from sparsify import SparseCoder, SparseCoderConfig

from sae_lens.saes.sae import SAE, TrainStepInput
from sae_lens.saes.topk_sae import TopK, TopKSAE, TopKTrainingSAE
from tests.helpers import (
    assert_close,
    build_topk_sae_cfg,
    build_topk_sae_training_cfg,
)


def test_TopKTrainingSAE_topk_aux_loss_matches_unnormalized_sparsify_implementation():
    d_in = 128
    d_sae = 192
    k = 26
    cfg = build_topk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
    )

    sae = TopKTrainingSAE(cfg)
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
    assert norm_aux_loss == pytest.approx(comparison_aux_loss, abs=3e-2)


def test_TopKSAE_save_and_load_from_pretrained(tmp_path: Path) -> None:
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
        assert_close(
            sae_state_dict[key],
            sae_loaded_state_dict[key],
        )

    sae_in = torch.randn(10, cfg.d_in, device=cfg.device)
    sae_out_1 = sae(sae_in)
    sae_out_2 = sae_loaded(sae_in)
    assert_close(sae_out_1, sae_out_2)


def test_TopKTrainingSAE_save_and_load_inference_sae(tmp_path: Path) -> None:
    # Create a training SAE with specific parameter values
    cfg = build_topk_sae_training_cfg(device="cpu", k=30)
    training_sae = TopKTrainingSAE(cfg)

    # Set some known values for testing
    training_sae.W_enc.data = torch.randn_like(training_sae.W_enc.data)
    training_sae.W_dec.data = torch.randn_like(training_sae.W_dec.data)
    training_sae.b_enc.data = torch.randn_like(training_sae.b_enc.data)
    training_sae.b_dec.data = torch.randn_like(training_sae.b_dec.data)

    # Save original state for comparison
    original_W_enc = training_sae.W_enc.data.clone()
    original_W_dec = training_sae.W_dec.data.clone()
    original_b_enc = training_sae.b_enc.data.clone()
    original_b_dec = training_sae.b_dec.data.clone()

    # Save as inference model
    model_path = str(tmp_path)
    training_sae.save_inference_model(model_path)

    assert os.path.exists(model_path)

    # Load as inference SAE
    inference_sae = SAE.load_from_disk(model_path, device="cpu")

    # Should be loaded as TopKSAE
    assert isinstance(inference_sae, TopKSAE)

    # Check that all parameters match
    assert_close(inference_sae.W_enc, original_W_enc)
    assert_close(inference_sae.W_dec, original_W_dec)
    assert_close(inference_sae.b_enc, original_b_enc)
    assert_close(inference_sae.b_dec, original_b_dec)

    # Check that the k parameter is correctly preserved in the config
    assert inference_sae.cfg.k == cfg.k

    # Verify forward pass gives same results
    sae_in = torch.randn(10, cfg.d_in, device="cpu")

    # Get output from training SAE
    training_feature_acts, _ = training_sae.encode_with_hidden_pre(sae_in)
    training_sae_out = training_sae.decode(training_feature_acts)

    # Get output from inference SAE
    inference_feature_acts = inference_sae.encode(sae_in)
    inference_sae_out = inference_sae.decode(inference_feature_acts)

    # Should produce identical outputs
    assert_close(training_feature_acts.to_dense(), inference_feature_acts)
    assert_close(training_sae_out, inference_sae_out, rtol=1e-4, atol=1e-4)

    # Test the full forward pass
    training_full_out = training_sae(sae_in)
    inference_full_out = inference_sae(sae_in)
    assert_close(training_full_out, inference_full_out, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("num_dims", [1, 2, 3, 4, 5])
def test_topK_sparse_activations(num_dims: bool):
    # Validate that the sparse top-K intermediate output (COO format)
    # we use to accelerate the decoder matches the dense top-K output.
    dims = (np.arange(1, num_dims + 1) + 3).tolist()
    dims[-1] = 1024
    for k in [1, 10, 100, 1000]:
        topk_sparse = TopK(k, use_sparse_activations=True)
        topk_dense = TopK(k, use_sparse_activations=False)
        x = torch.randn(*dims) + 50.0
        sparse_x = topk_sparse(x)
        assert sparse_x.is_sparse
        sparse_x = sparse_x.to_dense()
        dense_x = topk_dense(x)
        assert_close(dense_x, sparse_x)


@pytest.mark.parametrize("num_dims", [1, 2, 3, 4, 5])
def test_topK_activation_sparse_mm(num_dims: int):
    # Validate that our decoder produces the same output when using the sparse intermediates
    # as when using the dense intermediates.
    d_in = 128
    d_sae = 1024
    dims = (np.arange(1, num_dims + 1) + 3).tolist()
    dims[-1] = d_sae

    cfg = build_topk_sae_training_cfg(
        d_in=d_in,
        d_sae=d_sae,
        k=26,
    )

    sae = TopKTrainingSAE(cfg)

    with torch.no_grad():
        # increase b_enc so all features are likely above 0
        # sparsify includes a relu() in their pre_acts, but
        # this is not something we need to try to replicate.
        sae.b_enc.data = sae.b_enc + 100.0

    for k in [1, 10, 100, 1000]:
        topk_sparse = TopK(k, use_sparse_activations=True)
        topk_dense = TopK(k, use_sparse_activations=False)
        x = torch.randn(*dims) + 50.0
        sparse_x = topk_sparse(x)
        sae_out_sparse = sae.decode(sparse_x)
        dense_x = topk_dense(x)
        sae_out_dense = sae.decode(dense_x)
        assert_close(sae_out_sparse, sae_out_dense, rtol=1e-4, atol=5e-4)


def test_TopKTrainingSAE_sparse_activations_config():
    # Check that our config is respected in both training & inference SAEs
    cfg = build_topk_sae_training_cfg(k=100, use_sparse_activations=True)
    sae = TopKTrainingSAE(cfg)
    assert sae.activation_fn.use_sparse_activations  # type: ignore
    assert sae.cfg.use_sparse_activations

    cfg = build_topk_sae_training_cfg(k=100, use_sparse_activations=False)
    sae = TopKTrainingSAE(cfg)
    assert not sae.activation_fn.use_sparse_activations  # type: ignore
    assert not sae.cfg.use_sparse_activations
