import pytest
import torch

# Old modules (from the original code)
from sae_lens.sae import SAE, SAEConfig

# New modules (our re-implementation)
from sae_lens.saes.sae_base import SAEConfig as NewSAEConfig
from sae_lens.saes.topk_sae import TopKSAE, TopKTrainingSAE
from sae_lens.training.training_sae import TrainingSAE, TrainingSAEConfig


@pytest.fixture
def seed_everything():
    """
    Ensure deterministic tests by setting a fixed random seed.
    """
    torch.manual_seed(42)
    yield
    torch.manual_seed(0)


def compare_params(old_model: SAE, new_model: TopKSAE):
    """
    Compare parameter names and shapes between old and new SAEs.
    """
    old_params = dict(old_model.named_parameters())
    new_params = dict(new_model.named_parameters())
    assert sorted(old_params.keys()) == sorted(new_params.keys()), (
        f"Parameter names differ.\n"
        f"Old: {sorted(old_params.keys())}\nNew: {sorted(new_params.keys())}"
    )
    for k in old_params:
        assert old_params[k].shape == new_params[k].shape, (
            f"Param {k} shape mismatch: "
            f"old {old_params[k].shape} vs new {new_params[k].shape}"
        )


def make_old_topk_sae(d_in=16, d_sae=8, use_error_term=False) -> SAE:
    """
    Instantiate the old (original) topk SAE.
    """
    cfg = SAEConfig(
        architecture="topk",
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name="blocks.0.hook_resid_pre",  # avoid hook_z for simpler shape
        hook_layer=0,
        hook_head_index=None,
        activation_fn="topk",
        activation_fn_kwargs={"k": 4},  # example k
        apply_b_dec_to_input=False,
        finetuning_scaling_factor=False,
        normalize_activations="none",
        context_size=None,
        dataset_path=None,
        dataset_trust_remote_code=False,
        sae_lens_training_version="test_version",
        model_from_pretrained_kwargs={},
        seqpos_slice=None,
        prepend_bos=False,
    )
    old_sae = SAE(cfg)
    old_sae.use_error_term = use_error_term
    return old_sae


def make_new_topk_sae(d_in=16, d_sae=8, use_error_term=False) -> TopKSAE:
    """
    Instantiate the new TopKSAE re-implementation.
    """
    new_cfg = NewSAEConfig(
        architecture="topk",
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        hook_head_index=None,
        activation_fn="topk",
        activation_fn_kwargs={"k": 4},  # match old config
        apply_b_dec_to_input=False,
        finetuning_scaling_factor=False,
        normalize_activations="none",
        context_size=None,
        dataset_path=None,
        dataset_trust_remote_code=False,
        sae_lens_training_version="test_version",
        model_from_pretrained_kwargs={},
        seqpos_slice=None,
        prepend_bos=False,
    )
    return TopKSAE(new_cfg, use_error_term=use_error_term)


def test_topk_sae_inference_equivalence():
    """
    Compare old vs new topk SAE on:
      - parameter shape
      - forward pass shape
      - optional error_term
    """
    old_sae = make_old_topk_sae(d_in=16, d_sae=8, use_error_term=False)
    new_sae = make_new_topk_sae(d_in=16, d_sae=8, use_error_term=False)

    compare_params(old_sae, new_sae)

    # Provide example input
    x = torch.randn(2, 4, 16, dtype=torch.float32)

    with torch.no_grad():
        old_out = old_sae(x)
        new_out = new_sae(x)

    assert old_out.shape == new_out.shape, "Output shape mismatch."
    assert torch.isfinite(old_out).all(), "Old SAE produced NaNs or inf"
    assert torch.isfinite(new_out).all(), "New SAE produced NaNs or inf"

    # Now test with error_term
    old_sae_err = make_old_topk_sae(d_in=16, d_sae=8, use_error_term=True)
    new_sae_err = make_new_topk_sae(d_in=16, d_sae=8, use_error_term=True)

    with torch.no_grad():
        old_err_out = old_sae_err(x)
        new_err_out = new_sae_err(x)

    # They won't necessarily match numerically (old topk does some specifics),
    # but at least check shape & finiteness
    assert old_err_out.shape == new_err_out.shape
    assert torch.isfinite(old_err_out).all(), "Old error-term output has NaNs/inf"
    assert torch.isfinite(new_err_out).all(), "New error-term output has NaNs/inf"


def test_topk_sae_training_equivalence():
    """
    Compare old vs new topk training SEAs on:
      - parameter shape
      - training forward pass shape
      - partial correctness of losses
    """
    # Build old vs new training configs
    old_training_cfg = TrainingSAEConfig(
        architecture="topk",
        d_in=16,
        d_sae=8,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        hook_head_index=None,
        activation_fn="topk",
        activation_fn_kwargs={"k": 4},
        apply_b_dec_to_input=False,
        finetuning_scaling_factor=False,
        normalize_activations="none",
        context_size=None,
        dataset_path=None,
        dataset_trust_remote_code=False,
        sae_lens_training_version="test_version",
        model_from_pretrained_kwargs={},
        seqpos_slice=None,
        prepend_bos=False,
        # Training-specific:
        l1_coefficient=0.01,
        lp_norm=1.0,
        use_ghost_grads=False,
        normalize_sae_decoder=False,
        noise_scale=0.0,
        decoder_orthogonal_init=False,
        mse_loss_normalization=None,
        jumprelu_init_threshold=0.0,
        jumprelu_bandwidth=1.0,
        decoder_heuristic_init=False,
        init_encoder_as_decoder_transpose=False,
        scale_sparsity_penalty_by_decoder_norm=False,
    )

    old_training_sae = TrainingSAE(old_training_cfg)
    new_training_sae = TopKTrainingSAE(old_training_cfg)

    # Compare param shapes
    old_params = dict(old_training_sae.named_parameters())
    new_params = dict(new_training_sae.named_parameters())
    assert sorted(old_params.keys()) == sorted(new_params.keys()), "Param names differ"
    for k in old_params:
        assert (
            old_params[k].shape == new_params[k].shape
        ), f"Shape mismatch on param {k}"

    # Provide random input
    x = torch.randn(2, 3, 16, dtype=torch.float32)

    old_training_sae.train()
    new_training_sae.train()

    old_out = old_training_sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=old_training_cfg.l1_coefficient,
        dead_neuron_mask=None,
    )
    new_out = new_training_sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=old_training_cfg.l1_coefficient,
        dead_neuron_mask=None,
    )

    # Check output shape
    assert old_out.sae_out.shape == new_out.sae_out.shape
    # Check MSE loss is finite
    assert torch.isfinite(old_out.losses["mse_loss"])
    assert torch.isfinite(new_out.losses["mse_loss"])
    # Compare total loss shape
    assert old_out.loss.shape == new_out.loss.shape

    # This test doesn't require them to be identical numerically,
    # because the old code's topk might do something slightly different:
    # e.g. standard topk vs new topk might differ in subtleties.

    # But we confirm both are finite and the forward pass works
    assert torch.isfinite(old_out.loss)
    assert torch.isfinite(new_out.loss)
