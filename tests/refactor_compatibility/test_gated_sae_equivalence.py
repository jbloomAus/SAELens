import pytest
import torch

# Old modules
from sae_lens.sae import SAE, SAEConfig
from sae_lens.training.training_sae import TrainingSAE, TrainingSAEConfig

# New Gated modules
from sae_lens.saes.gated_sae import GatedSAE, GatedTrainingSAE
from sae_lens.saes.sae_base import SAEConfig as NewSAEConfig, TrainingSAEConfig as NewTrainingSAEConfig

@pytest.fixture
def seed_everything():
    """
    Ensure deterministic behavior for tests by setting a fixed random seed.
    """
    torch.manual_seed(42)
    yield
    torch.manual_seed(0)

def make_old_gated_sae(d_in=16, d_sae=8, use_error_term=False) -> SAE:
    """
    Helper to instantiate an old Gated SAE instance for testing.
    This creates an old SAE with architecture='gated'.
    """
    old_cfg = SAEConfig(
        architecture="gated",
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        hook_head_index=None,
        activation_fn_str="relu",
        activation_fn_kwargs={},
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
    old_sae = SAE(old_cfg)
    # Note: Gated SAE doesn't support error term, so this should be False
    old_sae.use_error_term = use_error_term
    return old_sae

def make_new_gated_sae(d_in=16, d_sae=8, use_error_term=False) -> GatedSAE:
    """
    Helper to instantiate a new GatedSAE instance for testing (inference only).
    """
    new_cfg = NewSAEConfig(
        architecture="gated",
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        hook_head_index=None,
        activation_fn_str="relu",
        activation_fn_kwargs={},
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
    sae = GatedSAE(new_cfg, use_error_term=use_error_term)
    return sae

def compare_params(old_sae: SAE, new_sae: GatedSAE):
    """
    Compare parameter names and shapes between the old Gated SAE and the new GatedSAE.
    """
    old_params = dict(old_sae.named_parameters())
    new_params = dict(new_sae.named_parameters())
    old_keys = sorted(old_params.keys())
    new_keys = sorted(new_params.keys())

    assert old_keys == new_keys, (
        f"Parameter names differ.\nOld: {old_keys}\nNew: {new_keys}"
    )

    for key in old_keys:
        v_old = old_params[key]
        v_new = new_params[key]
        assert v_old.shape == v_new.shape, (
            f"Param {key} shape mismatch: old {v_old.shape}, new {v_new.shape}"
        )

def debug_fold_w_dec_norm():
    """Debug the fold_W_dec_norm function for both implementations."""
    print("\n=== DEBUGGING FOLD_W_DEC_NORM ===")
    
    # Create instances with identical parameters
    old_sae = make_old_gated_sae(use_error_term=False)
    new_sae = make_new_gated_sae(use_error_term=False)
    
    # Set identical random parameters
    old_params = dict(old_sae.named_parameters())
    new_params = dict(new_sae.named_parameters())
    
    with torch.no_grad():
        for k in sorted(old_params.keys()):
            rand = torch.rand_like(old_params[k])
            old_params[k].copy_(rand)
            new_params[k].copy_(rand)
    
    # Print initial decoder norm values
    old_w_dec_norm = old_sae.W_dec.norm(dim=-1)
    new_w_dec_norm = new_sae.W_dec.norm(dim=-1)
    
    print(f"Initial W_dec norms - OLD: {old_w_dec_norm[:5]} NEW: {new_w_dec_norm[:5]}")
    
    # Save parameter values before folding
    params_before_old = {k: v.clone() for k, v in old_params.items()}
    params_before_new = {k: v.clone() for k, v in new_params.items()}
    
    # Perform the folding
    print("Calling fold_W_dec_norm...")
    old_sae.fold_W_dec_norm()
    new_sae.fold_W_dec_norm()
    
    # Check parameter values after folding
    for k in sorted(old_params.keys()):
        # Calculate relative changes
        old_change = torch.norm(old_params[k] - params_before_old[k]) / torch.norm(params_before_old[k])
        new_change = torch.norm(new_params[k] - params_before_new[k]) / torch.norm(params_before_new[k])
        
        print(f"Parameter {k}:")
        print(f"  OLD change: {old_change.item():.6f}, NEW change: {new_change.item():.6f}")
        
        if not torch.allclose(old_params[k], new_params[k], atol=1e-5):
            print(f"  !! MISMATCH !! Max diff: {torch.max(torch.abs(old_params[k] - new_params[k])).item()}")
            # Sample a few values
            flat_old = old_params[k].flatten()
            flat_new = new_params[k].flatten()
            if len(flat_old) > 5:
                print(f"  Sample OLD: {flat_old[:5]}")
                print(f"  Sample NEW: {flat_new[:5]}")
    
    # Check encoder and decoder norms after folding
    old_w_dec_norm_after = old_sae.W_dec.norm(dim=-1)
    new_w_dec_norm_after = new_sae.W_dec.norm(dim=-1)
    
    print(f"W_dec norms after folding - OLD: {old_w_dec_norm_after[:5]} NEW: {new_w_dec_norm_after[:5]}")
    
    # Check forward pass output
    x = torch.randn(2, 3, 16, dtype=torch.float32)
    old_out = old_sae(x)
    new_out = new_sae(x)
    
    print(f"Output shapes - OLD: {old_out.shape}, NEW: {new_out.shape}")
    print(f"Output max difference: {torch.max(torch.abs(old_out - new_out)).item()}")
    print("=== END DEBUGGING FOLD_W_DEC_NORM ===\n")

@pytest.mark.parametrize("use_error_term", [False])
def test_gated_inference_equivalence(seed_everything, use_error_term):
    """
    Test that the old vs new Gated SAEs match in parameter shape and forward pass outputs.
    Note: Gated SAE doesn't support error_term=True.
    """
    old_sae = make_old_gated_sae(d_in=16, d_sae=8, use_error_term=use_error_term)
    new_sae = make_new_gated_sae(d_in=16, d_sae=8, use_error_term=use_error_term)

    # Ensure parameters are identical before comparing outputs
    with torch.no_grad():
        old_params = dict(old_sae.named_parameters())
        new_params = dict(new_sae.named_parameters())
        for k in sorted(old_params.keys()):
            new_params[k].copy_(old_params[k])

    # Compare parameter shapes
    compare_params(old_sae, new_sae)

    # Provide a random input
    x = torch.randn(2, 4, 16, dtype=torch.float32)

    with torch.no_grad():
        old_out = old_sae(x)
        new_out = new_sae(x)

    assert old_out.shape == new_out.shape, "Output shape mismatch."
    assert torch.isfinite(old_out).all(), "Old output contains non-finite values."
    assert torch.isfinite(new_out).all(), "New output contains non-finite values."
    
    # Now they really should match, since we forcibly aligned params
    assert torch.allclose(old_out, new_out, atol=1e-5), \
        "Outputs differ between old and new implementations."

@pytest.mark.parametrize("fold_fn", ["fold_W_dec_norm", "fold_activation_norm_scaling_factor"])
def test_gated_fold_equivalence(seed_everything, fold_fn):
    """
    Test that folding functions (fold_W_dec_norm or fold_activation_norm_scaling_factor)
    on old vs new Gated SAE yields consistent results on forward passes.
    
    This is especially important for fold_W_dec_norm, which has gated-specific logic.
    """
    # First, run debugging if this is fold_W_dec_norm
    if fold_fn == "fold_W_dec_norm":
        debug_fold_w_dec_norm()
    old_sae = make_old_gated_sae(use_error_term=False)
    new_sae = make_new_gated_sae(use_error_term=False)
    compare_params(old_sae, new_sae)

    # We'll line up parameters by name so that p_old and p_new actually match in shape.
    old_params = dict(old_sae.named_parameters())
    new_params = dict(new_sae.named_parameters())

    # fill random data so that norms differ
    with torch.no_grad():
        for k in sorted(old_params.keys()):
            rand = torch.rand_like(old_params[k])
            old_params[k].copy_(rand)
            new_params[k].copy_(rand)

    # Now call the fold function
    if fold_fn == "fold_W_dec_norm":
        old_sae.fold_W_dec_norm()
        new_sae.fold_W_dec_norm()
    elif fold_fn == "fold_activation_norm_scaling_factor":
        scale_factor = 2.0
        old_sae.fold_activation_norm_scaling_factor(scale_factor)
        new_sae.fold_activation_norm_scaling_factor(scale_factor)

    # Provide input, compare outputs
    x = torch.randn(2, 3, 16, dtype=torch.float32)
    old_out = old_sae(x)
    new_out = new_sae(x)

    assert old_out.shape == new_out.shape, f"Output shape mismatch after {fold_fn}"
    
    # Compare the actual values - they should match closely after folding
    assert torch.allclose(old_out, new_out, atol=1e-5), \
        f"{fold_fn} produces different results between old and new implementations"
    
    # Also check the folded parameters directly
    for k in sorted(old_params.keys()):
        assert torch.allclose(old_params[k], new_params[k], atol=1e-5), \
            f"Parameter {k} differs after {fold_fn}"

def test_gated_run_hooks_equivalence(seed_everything):
    """
    Compare hooking behavior for GatedSAE. We'll check that hooking triggers
    the same number of calls in old_sae vs new_sae and that output shapes match.
    """
    class Counter:
        def __init__(self):
            self.count = 0
        def inc(self, *args, **kwargs):
            self.count += 1

    old_sae = make_old_gated_sae()
    new_sae = make_new_gated_sae()

    old_c = Counter()
    new_c = Counter()

    old_hooks_to_add = [k for k, _ in old_sae.hook_dict.items()]
    new_hooks_to_add = [k for k, _ in new_sae.hook_dict.items()]

    for name in old_hooks_to_add:
        old_sae.add_hook(name, old_c.inc, dir="fwd")
    for name in new_hooks_to_add:
        new_sae.add_hook(name, new_c.inc, dir="fwd")

    x = torch.randn(2, 4, 16, dtype=torch.float32)
    old_out = old_sae(x)
    new_out = new_sae(x)
    assert old_out.shape == new_out.shape, "Mismatch in forward shape with hooking test."

    assert old_c.count > 0, "No hooks triggered in old Gated SAE"
    assert new_c.count > 0, "No hooks triggered in new Gated SAE"
    assert old_c.count == new_c.count, "Different number of hooks called in old vs new implementation"

#####################################
# Training Equivalence
#####################################

def make_old_gated_training_sae(d_in=16, d_sae=8) -> TrainingSAE:
    """
    Helper to instantiate an old TrainingSAE configured as Gated for testing.
    """
    old_training_cfg = TrainingSAEConfig(
        architecture="gated",
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        hook_head_index=None,
        activation_fn_str="relu",
        activation_fn_kwargs={},
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
        # training fields
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
    return TrainingSAE(old_training_cfg)

def make_new_gated_training_sae(d_in=16, d_sae=8) -> GatedTrainingSAE:
    """
    Helper to instantiate a new GatedTrainingSAE instance.
    """
    new_training_cfg = NewTrainingSAEConfig(
        architecture="gated",
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        hook_head_index=None,
        activation_fn_str="relu",
        activation_fn_kwargs={},
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
    return GatedTrainingSAE(new_training_cfg)

def test_gated_training_equivalence(seed_everything):
    """
    Test that old vs new Gated SAEs match in training behavior.
    We'll check the outputs, losses, and ensure numeric equivalence.
    """
    old_sae = make_old_gated_training_sae()
    new_sae = make_new_gated_training_sae()

    # Ensure parameters are identical before comparing outputs
    with torch.no_grad():
        old_params = dict(old_sae.named_parameters())
        new_params = dict(new_sae.named_parameters())
        for k in sorted(old_params.keys()):
            new_params[k].copy_(old_params[k])

    # Put in training mode
    old_sae.train()
    new_sae.train()

    # Compare parameters
    compare_params(old_sae, new_sae)

    # Create consistent random data
    batch_size, seq_len, d_in = 2, 4, 16
    x = torch.randn(batch_size, seq_len, d_in, dtype=torch.float32)

    # Get training outputs
    old_out = old_sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=old_sae.cfg.l1_coefficient,
        dead_neuron_mask=None
    )
    new_out = new_sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=new_sae.cfg.l1_coefficient,
        dead_neuron_mask=None
    )

    # Check output shapes
    assert old_out.sae_out.shape == new_out.sae_out.shape, \
        "Gated training output shape mismatch."
    
    # Check all values are finite
    assert torch.isfinite(old_out.sae_out).all(), "Old Gated training out is not finite."
    assert torch.isfinite(new_out.sae_out).all(), "New Gated training out is not finite."
    assert torch.isfinite(old_out.loss).all(), "Old Gated training loss is not finite."
    assert torch.isfinite(new_out.loss).all(), "New Gated training loss is not finite."
    
    # Check for losses present in both
    assert "mse_loss" in old_out.losses, "Missing MSE loss in old implementation"
    assert "mse_loss" in new_out.losses, "Missing MSE loss in new implementation"
    
    # Gated SAE has specific losses
    assert "l1_loss" in old_out.losses.keys() or "auxiliary_reconstruction_loss" in old_out.losses.keys(), \
        "Old Gated training missing expected loss terms."
    assert "l1_loss" in new_out.losses or "gating_aux_loss" in new_out.losses, \
        "New Gated training missing expected loss terms."
    
    # Check if training forward pass is equivalent
    assert torch.allclose(old_out.sae_out, new_out.sae_out, atol=1e-5), \
        "Output differs between old and new Gated implementation"
    assert torch.allclose(old_out.loss, new_out.loss, atol=1e-5), \
        "Loss differs between old and new Gated implementation" 