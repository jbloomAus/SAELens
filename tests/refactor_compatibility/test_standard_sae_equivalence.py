import einops
import pytest
import torch

# Old modules
from sae_lens.sae import SAE, SAEConfig

# New modules
from sae_lens.saes.sae_base import SAEConfig as NewSAEConfig
from sae_lens.saes.standard_sae import StandardSAE, StandardTrainingSAE
from sae_lens.training.training_sae import TrainingSAE, TrainingSAEConfig


@pytest.fixture
def seed_everything():
    """
    Ensure deterministic behavior for tests by setting a fixed random seed.
    """
    torch.manual_seed(42)
    yield
    torch.manual_seed(0)


def make_old_sae(
    architecture="standard", d_in=16, d_sae=8, use_error_term=False
) -> SAE:
    """
    Helper to instantiate an old SAE instance for testing.
    Note: use a hook_name that does NOT end with '_z', so that we avoid
    the old code's auto-flattening logic that changes shapes unexpectedly.
    """
    old_cfg = SAEConfig(
        architecture=architecture,
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        # Use 'hook_resid_pre' to avoid hooking on '_z' and flattening
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        hook_head_index=None,
        activation_fn="relu",
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
    old_sae.use_error_term = use_error_term
    return old_sae


def make_new_sae(
    architecture="standard", d_in=16, d_sae=8, use_error_term=False
) -> StandardSAE:
    """
    Helper to instantiate a new StandardSAE instance for testing.
    Mirror the same hook_name that does NOT end with '_z'.
    """
    new_cfg = NewSAEConfig(
        architecture=architecture,
        d_in=d_in,
        d_sae=d_sae,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name="blocks.0.hook_resid_pre",
        hook_layer=0,
        hook_head_index=None,
        activation_fn="relu",
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
    return StandardSAE(new_cfg, use_error_term=use_error_term)


def compare_params(old_sae: SAE, new_sae: StandardSAE):
    """
    Compare parameter names and shapes between the old SAE and the new StandardSAE.
    """
    old_params = dict(old_sae.named_parameters())
    new_params = dict(new_sae.named_parameters())
    old_keys = sorted(old_params.keys())
    new_keys = sorted(new_params.keys())

    assert (
        old_keys == new_keys
    ), f"Parameter names differ.\nOld: {old_keys}\nNew: {new_keys}"

    for key in old_keys:
        v_old = old_params[key]
        v_new = new_params[key]
        assert (
            v_old.shape == v_new.shape
        ), f"Param {key} shape mismatch: old {v_old.shape}, new {v_new.shape}"


@pytest.mark.parametrize("architecture", ["standard", "gated", "jumprelu"])
def test_standard_sae_inference_equivalence(architecture):  # type: ignore
    """
    Extended test of old vs new SAE inference equivalence across multiple architectures:
      - compare param shape
      - compare forward pass output shapes & ensure finite
      - optionally test error_term usage
    """
    if architecture in ("gated", "jumprelu"):
        # The new code has separate classes for these; for brevity, we skip here.
        pytest.skip(
            f"{architecture} is tested in new classes separately, skipping for demonstration."
        )

    old_sae = make_old_sae(
        architecture=architecture, d_in=16, d_sae=8, use_error_term=False
    )
    new_sae = make_new_sae(
        architecture=architecture, d_in=16, d_sae=8, use_error_term=False
    )
    compare_params(old_sae, new_sae)

    # Provide a random input
    x = torch.randn(2, 4, 16, dtype=torch.float32)

    with torch.no_grad():
        old_out = old_sae(x)
        new_out = new_sae(x)

    assert old_out.shape == new_out.shape, "Output shape mismatch."
    assert torch.isfinite(old_out).all()
    assert torch.isfinite(new_out).all()

    # Now test error_term usage
    old_sae_error = make_old_sae(
        architecture=architecture, d_in=16, d_sae=8, use_error_term=True
    )
    new_sae_error = make_new_sae(
        architecture=architecture, d_in=16, d_sae=8, use_error_term=True
    )

    with torch.no_grad():
        old_err_out = old_sae_error(x)
        new_err_out = new_sae_error(x)

    assert old_err_out.shape == new_err_out.shape
    # standard architecture can match exactly
    if architecture == "standard":
        assert torch.allclose(
            old_err_out, new_err_out, atol=1e-5
        ), "Mismatch in old/new output with error term (standard arch)"


@pytest.mark.parametrize(
    "fold_fn", ["fold_W_dec_norm", "fold_activation_norm_scaling_factor"]
)
def test_standard_sae_fold_equivalence(fold_fn):
    """
    Test that calling fold functions (like fold_W_dec_norm or fold_activation_norm_scaling_factor)
    on old vs new yields consistent results on forward passes.
    """
    old_sae = make_old_sae(architecture="standard", use_error_term=False)
    new_sae = make_new_sae(architecture="standard", use_error_term=False)
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
        # pick a random scaling factor
        scale_factor = 2.0
        old_sae.fold_activation_norm_scaling_factor(scale_factor)
        new_sae.fold_activation_norm_scaling_factor(scale_factor)

    # Provide input, compare outputs
    x = torch.randn(2, 3, 16, dtype=torch.float32)
    old_out = old_sae(x)
    new_out = new_sae(x)
    assert old_out.shape == new_out.shape
    assert torch.allclose(
        old_out, new_out, atol=1e-5
    ), f"{fold_fn} mismatch between old and new"


def test_standard_sae_run_hooks_equivalence():
    """
    Compare hooking behavior (similar to test_hooked_sae).
    We'll check that hooking triggers the same number of calls in old_sae vs new_sae
    and that output shapes match.
    """

    class Counter:
        def __init__(self):
            self.count = 0

        def inc(self, *args, **kwargs):
            self.count += 1

    old_sae = make_old_sae()
    new_sae = make_new_sae()

    # define hooks in the same way:
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
    assert (
        old_out.shape == new_out.shape
    ), "Mismatch in forward shape with hooking test."

    # We don't require an identical count if the new code has more internal sub-hooks,
    # but both should have triggered at least once.
    assert old_c.count > 0, "No hooks triggered in old SAE"
    assert new_c.count > 0, "No hooks triggered in new SAE"


@pytest.mark.parametrize(
    "hook_name",
    [
        "blocks.0.hook_resid_pre",  # standard case
        "blocks.0.attn.hook_z",  # hook_z case
    ],
)
def test_standard_sae_hook_z_equivalence(hook_name):
    """
    Test that both old and new SAEs handle hook_z cases correctly.
    For hook_z, inputs should be reshaped from (..., n_heads, d_head) to (..., n_heads*d_head).
    """
    # For hook_z case, d_in should be n_heads * d_head
    n_heads = 8
    d_head = 8
    d_in = n_heads * d_head  # Always use flattened dimension for SAE config

    def make_old_sae_with_hook(hook_name, d_in) -> SAE:
        cfg = SAEConfig(
            architecture="standard",
            d_in=d_in,  # Always use flattened dimension
            d_sae=32,
            dtype="float32",
            device="cpu",
            model_name="test_model",
            hook_name=hook_name,
            hook_layer=0,
            hook_head_index=None,
            activation_fn="relu",
            activation_fn_kwargs={},
            apply_b_dec_to_input=False,  # Important: set to False to avoid shape issues
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
        return SAE(cfg)

    def make_new_sae_with_hook(hook_name, d_in) -> StandardSAE:
        cfg = NewSAEConfig(
            architecture="standard",
            d_in=d_in,  # Always use flattened dimension
            d_sae=32,
            dtype="float32",
            device="cpu",
            model_name="test_model",
            hook_name=hook_name,
            hook_layer=0,
            hook_head_index=None,
            activation_fn="relu",
            activation_fn_kwargs={},
            apply_b_dec_to_input=False,  # Important: set to False to avoid shape issues
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
        return StandardSAE(cfg)

    old_sae = make_old_sae_with_hook(hook_name, d_in)
    new_sae = make_new_sae_with_hook(hook_name, d_in)

    # Create input with appropriate shape
    batch_size = 2
    seq_len = 4
    if hook_name.endswith("_z"):
        # Input shape for attention hook: (batch, seq, heads, d_head)
        x = torch.randn(batch_size, seq_len, n_heads, d_head)

        # New SAE needs explicit hook_z handling turned on
        new_sae.turn_on_forward_pass_hook_z_reshaping()
        # Set d_head for proper reshaping
        new_sae.d_head = d_head
    else:
        # Standard shape for other hooks: (batch, seq, d_in)
        x = torch.randn(batch_size, seq_len, d_in)

    # Test forward pass shapes
    with torch.no_grad():
        old_out = old_sae(x)
        new_out = new_sae(x)

    assert (
        old_out.shape == new_out.shape == x.shape
    ), f"Output shape mismatch for {hook_name}. Old: {old_out.shape}, New: {new_out.shape}, Input: {x.shape}"

    # For hook_z case, verify the internal reshape happened
    if hook_name.endswith("_z"):
        # Run with caching to check internal shapes
        old_out, old_cache = old_sae.run_with_cache(x)
        new_out, new_cache = new_sae.run_with_cache(x)

        # Both should have flattened internal activations
        assert (
            old_cache["hook_sae_input"].shape[-1] == n_heads * d_head
        ), "Old SAE didn't flatten hook_z input correctly"
        assert (
            new_cache["hook_sae_input"].shape[-1] == n_heads * d_head
        ), "New SAE didn't flatten hook_z input correctly"

        # Verify encoder output shape is correct
        assert (
            old_cache["hook_sae_acts_pre"].shape[-1] == old_sae.cfg.d_sae
        ), "Old SAE encoder output shape incorrect"
        assert (
            new_cache["hook_sae_acts_pre"].shape[-1] == new_sae.cfg.d_sae
        ), "New SAE encoder output shape incorrect"

    # Clean up hook_z mode if it was enabled
    if hook_name.endswith("_z"):
        new_sae.turn_off_forward_pass_hook_z_reshaping()


@pytest.mark.parametrize(
    "hook_name",
    [
        "blocks.0.hook_resid_pre",  # standard case
        "blocks.0.attn.hook_z",  # hook_z case
    ],
)
def test_standard_sae_training_hook_z_equivalence(hook_name):
    """
    Test that training works correctly with hook_z reshaping.
    """
    # For hook_z case, d_in should be n_heads * d_head
    n_heads = 8
    d_head = 8
    d_in = n_heads * d_head  # Always use flattened dimension

    old_training_cfg = TrainingSAEConfig(
        architecture="standard",
        d_in=d_in,  # Always use flattened dimension
        d_sae=32,
        dtype="float32",
        device="cpu",
        model_name="test_model",
        hook_name=hook_name,
        hook_layer=0,
        hook_head_index=None,
        activation_fn="relu",
        activation_fn_kwargs={},
        apply_b_dec_to_input=False,  # Important: set to False
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
    new_training_sae = StandardTrainingSAE(old_training_cfg)

    # Create appropriate input shape and prepare data for each implementation
    batch_size = 2
    seq_len = 4
    if hook_name.endswith("_z"):
        # For new SAE: Keep original shape, it will handle reshaping internally
        new_input_data = torch.randn(batch_size, seq_len, n_heads, d_head)
        # For old SAE: Pre-flatten the input as it expects
        old_input_data = einops.rearrange(
            new_input_data.clone(), "... n_heads d_head -> ... (n_heads d_head)"
        )

        # Turn on hook_z reshaping for new SAE only (old one expects pre-flattened input)
        new_training_sae.turn_on_forward_pass_hook_z_reshaping()
        new_training_sae.d_head = d_head
    else:
        # For non-hook_z case, both use the same shape
        new_input_data = torch.randn(batch_size, seq_len, d_in)
        old_input_data = new_input_data.clone()

    old_training_sae.train()
    new_training_sae.train()

    # Forward pass with appropriate input for each
    old_out = old_training_sae.training_forward_pass(
        sae_in=old_input_data,
        current_l1_coefficient=old_training_cfg.l1_coefficient,
        dead_neuron_mask=None,
    )
    new_out = new_training_sae.training_forward_pass(
        sae_in=new_input_data,
        current_l1_coefficient=old_training_cfg.l1_coefficient,
        dead_neuron_mask=None,
    )

    # Check shapes match input
    if hook_name.endswith("_z"):
        assert old_out.sae_out.shape == old_input_data.shape
        assert new_out.sae_out.shape == new_input_data.shape
    else:
        assert old_out.sae_out.shape == new_out.sae_out.shape == new_input_data.shape

    # Check losses exist and are finite
    assert "mse_loss" in old_out.losses and "mse_loss" in new_out.losses
    assert torch.isfinite(old_out.losses["mse_loss"])
    assert torch.isfinite(new_out.losses["mse_loss"])

    # Check sparsity loss (name differs between implementations)
    old_sparsity_loss = old_out.losses.get("aux_loss", old_out.losses.get("l1_loss"))
    new_sparsity_loss = new_out.losses.get("aux_loss", new_out.losses.get("l1_loss"))
    assert torch.isfinite(old_sparsity_loss)
    assert torch.isfinite(new_sparsity_loss)

    # For hook_z case, verify the internal shapes
    if hook_name.endswith("_z"):
        # Run with caching to check internal shapes
        old_out, old_cache = old_training_sae.run_with_cache(old_input_data)
        new_out, new_cache = new_training_sae.run_with_cache(new_input_data)

        # Check input shapes
        assert (
            old_cache["hook_sae_input"].shape[-1] == n_heads * d_head
        ), "Old SAE input shape incorrect"
        assert (
            new_cache["hook_sae_input"].shape[-1] == n_heads * d_head
        ), "New SAE input shape incorrect"

        # Verify encoder output shape is correct
        assert (
            old_cache["hook_sae_acts_pre"].shape[-1] == old_training_sae.cfg.d_sae
        ), "Old SAE encoder output shape incorrect"
        assert (
            new_cache["hook_sae_acts_pre"].shape[-1] == new_training_sae.cfg.d_sae
        ), "New SAE encoder output shape incorrect"

    if hook_name.endswith("_z"):
        new_training_sae.turn_off_forward_pass_hook_z_reshaping()
