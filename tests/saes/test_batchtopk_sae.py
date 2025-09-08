import os
from pathlib import Path

import torch
import torch.nn as nn

from sae_lens.saes.batchtopk_sae import (
    BatchTopK,
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.saes.sae import SAE, TrainStepInput
from tests.helpers import (
    assert_close,
    assert_not_close,
    build_batchtopk_sae_training_cfg,
)


def test_BatchTopK_with_same_number_of_top_features_per_batch():
    batch_topk = BatchTopK(k=2)
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0]])

    # Expected output after:
    # 1. ReLU: [[1, 0, 3, 0], [0, 6, 0, 8]]
    # 2. Flatten: [1, 0, 3, 0, 0, 6, 0, 8]
    # 3. Top 4 values (k=2 * batch_size=2): [8, 6, 3, 1]
    # 4. Reshape back to original shape
    expected = torch.tensor([[1.0, 0.0, 3.0, 0.0], [0.0, 6.0, 0.0, 8.0]])

    output = batch_topk(x)

    assert output.shape == x.shape
    assert_close(output, expected)

    # Check that exactly k*batch_size values are non-zero
    assert (output != 0).sum() == batch_topk.k * x.shape[0]


def test_BatchTopK_with_imbalanced_top_features_per_batch():
    batch_topk = BatchTopK(k=2)
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # All positive values
            [-5.0, -6.0, 0.5, -8.0],  # Only one small positive value
        ]
    )

    # Expected output after:
    # 1. ReLU: [[1, 2, 3, 4], [0, 0, 0.5, 0]]
    # 2. Flatten: [1, 2, 3, 4, 0, 0, 0.5, 0]
    # 3. Top 4 values (k=2 * batch_size=2): [4, 3, 2, 1]
    # 4. Reshape back to original shape
    expected = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],  # Gets 4 non-zero values
            [0.0, 0.0, 0.0, 0.0],  # Gets 0 non-zero values
        ]
    )

    output = batch_topk(x)
    assert_close(output, expected)

    # Check that exactly k*batch_size values are non-zero across all batches
    assert (output != 0).sum() == batch_topk.k * x.shape[0]


def test_BatchTopK_with_float_k():
    batch_topk = BatchTopK(k=1.5)  # Float k value
    x = torch.tensor([[1.0, -2.0, 3.0, -4.0], [-5.0, 6.0, -7.0, 8.0]])

    # Expected output after:
    # 1. ReLU: [[1, 0, 3, 0], [0, 6, 0, 8]]
    # 2. Flatten: [1, 0, 3, 0, 0, 6, 0, 8]
    # 3. Top 3 values (k=1.5 * batch_size=2 = 3): [8, 6, 3]
    # 4. Reshape back to original shape
    expected = torch.tensor([[0.0, 0.0, 3.0, 0.0], [0.0, 6.0, 0.0, 8.0]])

    output = batch_topk(x)

    assert output.shape == x.shape
    assert_close(output, expected)

    # Check that exactly int(k*batch_size) values are non-zero
    expected_nonzero_count = int(batch_topk.k * x.shape[0])
    assert (output != 0).sum() == expected_nonzero_count


def test_BatchTopK_output_must_be_positive():
    batch_topk = BatchTopK(k=2)
    x = torch.tensor(
        [
            [-1.0, 2.0, -3.0, -4.0],  # Only 1 positive value
            [5.0, -6.0, -7.0, -8.0],  # Only 1 positive value
        ]
    )
    # Total positive values (2) < k*batch_size (4)

    # Expected output after:
    # 1. ReLU: [[0, 2, 0, 0], [5, 0, 0, 0]]
    # 2. Flatten: [0, 2, 0, 0, 5, 0, 0, 0]
    # 3. Top 4 values (k=2 * batch_size=2): [5, 2, 0, 0]
    # 4. Reshape back to original shape
    expected = torch.tensor(
        [
            [0.0, 2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0],
        ]
    )

    output = batch_topk(x)

    assert_close(output, expected)
    assert (output >= 0).all()

    # Check that number of non-zero values equals number of positive inputs
    # (which is less than k*batch_size)
    assert (output != 0).sum() == (x > 0).sum()


def test_BatchTopKTrainingSAEConfig_accepts_a_float_k():
    cfg = BatchTopKTrainingSAEConfig(k=1.5, d_in=10, d_sae=10)
    assert cfg.k == 1.5
    assert cfg.to_dict()["k"] == 1.5


def test_BatchTopKTrainingSAE_initialization():
    cfg = build_batchtopk_sae_training_cfg(device="cpu")
    sae = BatchTopKTrainingSAE.from_dict(cfg.to_dict())
    assert isinstance(sae.W_enc, nn.Parameter)
    assert isinstance(sae.W_dec, nn.Parameter)
    assert isinstance(sae.b_enc, nn.Parameter)
    assert isinstance(sae.b_dec, nn.Parameter)
    assert isinstance(sae.topk_threshold, torch.Tensor)

    assert sae.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sae.b_enc.shape == (cfg.d_sae,)
    assert sae.b_dec.shape == (cfg.d_in,)
    assert sae.topk_threshold.shape == ()

    # encoder/decoder should be initialized, everything else should be 0s
    assert_not_close(sae.W_enc, torch.zeros_like(sae.W_enc))
    assert_not_close(sae.W_dec, torch.zeros_like(sae.W_dec))
    assert_close(sae.b_dec, torch.zeros_like(sae.b_dec))
    assert_close(sae.b_enc, torch.zeros_like(sae.b_enc))
    assert sae.topk_threshold.item() == 0.0


def test_BatchTopKTrainingSAE_save_and_load_inference_sae(tmp_path: Path) -> None:
    # Create a training SAE with specific parameter values
    cfg = build_batchtopk_sae_training_cfg(device="cpu")
    training_sae = BatchTopKTrainingSAE(cfg)

    # Set some known values for testing
    training_sae.W_enc.data = torch.randn_like(training_sae.W_enc.data)
    training_sae.W_dec.data = torch.randn_like(training_sae.W_dec.data)
    training_sae.b_enc.data = torch.randn_like(training_sae.b_enc.data)
    training_sae.b_dec.data = torch.randn_like(training_sae.b_dec.data)

    sae_in = torch.randn(30, training_sae.cfg.d_in)
    train_step_input = TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
    )

    # run some test data through to learn the correct threshold
    for _ in range(500):
        training_sae.training_forward_pass(train_step_input)

    # Save original state for comparison
    original_W_enc = training_sae.W_enc.data.clone()
    original_W_dec = training_sae.W_dec.data.clone()
    original_b_enc = training_sae.b_enc.data.clone()
    original_b_dec = training_sae.b_dec.data.clone()
    original_threshold = training_sae.topk_threshold.item()

    # Save as inference model
    model_path = str(tmp_path)
    training_sae.save_inference_model(model_path)

    assert os.path.exists(model_path)

    # Load as inference SAE
    inference_sae = SAE.load_from_disk(model_path, device="cpu")

    # Should be loaded as JumpReLUSAE
    assert isinstance(inference_sae, JumpReLUSAE)

    # Check that all parameters match
    assert_close(inference_sae.W_enc, original_W_enc)
    assert_close(inference_sae.W_dec, original_W_dec)
    assert_close(inference_sae.b_enc, original_b_enc)
    assert_close(inference_sae.b_dec, original_b_dec)

    # Check that topk_threshold was converted to threshold
    assert_close(
        inference_sae.threshold,
        original_threshold * torch.ones_like(inference_sae.b_enc),
    )

    # Get output from training SAE
    training_feature_acts, _ = training_sae.encode_with_hidden_pre(sae_in)
    training_sae_out = training_sae.decode(training_feature_acts)

    # Get output from inference SAE
    inference_feature_acts = inference_sae.encode(sae_in)
    inference_sae_out = inference_sae.decode(inference_feature_acts)

    # Should produce identical outputs
    assert_close(training_feature_acts, inference_feature_acts)
    assert_close(training_sae_out, inference_sae_out)

    # Test the full forward pass
    training_full_out = training_sae(sae_in)
    inference_full_out = inference_sae(sae_in)
    assert_close(training_full_out, inference_full_out)


def test_BatchTopKTrainingSAE_training_step_updates_threshold() -> None:
    # Create BatchTopKTrainingSAE
    cfg = build_batchtopk_sae_training_cfg(device="cpu")
    sae = BatchTopKTrainingSAE(cfg)

    # Set initial threshold
    initial_threshold = 0.0
    sae.topk_threshold = torch.tensor(initial_threshold)
    sae.b_enc.data = torch.randn(sae.cfg.d_sae) + 10

    # Create input that will produce non-zero activations
    sae_in = torch.randn(4, sae.cfg.d_in)
    train_step_input = TrainStepInput(
        sae_in=sae_in,
        coefficients={},
        dead_neuron_mask=None,
    )

    # Do training step
    with torch.no_grad():
        output = sae.training_forward_pass(train_step_input)

    # Verify threshold changed
    assert sae.topk_threshold != initial_threshold

    # Verify threshold is included in metrics dict
    assert "topk_threshold" in output.metrics
    assert output.metrics["topk_threshold"] == sae.topk_threshold


def test_BatchTopKTrainingSAEConfig_get_inference_sae_cfg_dict() -> None:
    cfg = build_batchtopk_sae_training_cfg(device="cpu")
    sae = BatchTopKTrainingSAE(cfg)

    inference_config = sae.cfg.get_inference_sae_cfg_dict()

    # Should convert to JumpReLU architecture
    assert inference_config["architecture"] == "jumprelu"

    # Should preserve key config fields
    assert inference_config["d_in"] == cfg.d_in
    assert inference_config["d_sae"] == cfg.d_sae
    assert inference_config["dtype"] == cfg.dtype
    assert inference_config["device"] == cfg.device

    # Should not have BatchTopK-specific fields
    assert "topk_threshold_lr" not in inference_config


def test_BatchTopKTrainingSAE_process_state_dict_for_saving_inference() -> None:
    cfg = build_batchtopk_sae_training_cfg(device="cpu")
    sae = BatchTopKTrainingSAE(cfg)

    # Set a known threshold value
    threshold_value = 0.5
    sae.topk_threshold = torch.tensor(threshold_value)

    # Get state dict and process it
    state_dict = sae.state_dict()
    sae.process_state_dict_for_saving_inference(state_dict)

    # topk_threshold should be removed
    assert "topk_threshold" not in state_dict

    # threshold should be added with correct shape and value
    assert "threshold" in state_dict
    expected_threshold = torch.ones_like(sae.b_enc) * threshold_value
    assert_close(state_dict["threshold"], expected_threshold)
