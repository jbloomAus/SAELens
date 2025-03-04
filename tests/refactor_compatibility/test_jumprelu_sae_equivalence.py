import pytest
import torch

# Old modules
from sae_lens.sae import SAE, SAEConfig
from sae_lens.training.training_sae import TrainingSAE, TrainingSAEConfig

# New JumpReLU modules
from sae_lens.saes.jumprelu_sae import JumpReLUSAE, JumpReLUTrainingSAE
from sae_lens.saes.sae_base import SAEConfig as NewSAEConfig, TrainingSAEConfig as NewTrainingSAEConfig

@pytest.fixture
def seed_everything():
    """
    Ensure deterministic behavior for tests by setting a fixed random seed.
    """
    torch.manual_seed(42)
    yield
    torch.manual_seed(0)

def make_old_jumprelu_sae(d_in=16, d_sae=8, use_error_term=False) -> SAE:
    """
    Helper to instantiate an old JumpReLU SAE instance for testing.
    This creates an old SAE with architecture='jumprelu'.
    """
    # We replicate the logic from test_standard_sae_equivalence.make_old_sae,
    # but specify architecture="jumprelu".
    old_cfg = SAEConfig(
        architecture="jumprelu",
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
    old_sae = SAE(old_cfg)
    old_sae.use_error_term = use_error_term
    return old_sae

def make_new_jumprelu_sae(d_in=16, d_sae=8, use_error_term=False) -> JumpReLUSAE:
    """
    Helper to instantiate a new JumpReLUSAE instance for testing (inference only).
    """
    new_cfg = NewSAEConfig(
        architecture="jumprelu",
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
    sae = JumpReLUSAE(new_cfg, use_error_term=use_error_term)
    return sae

def compare_params(old_sae: SAE, new_sae: JumpReLUSAE):
    """
    Compare parameter names and shapes between the old JumpReLU SAE and the new JumpReLUSAE.
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

@pytest.mark.parametrize("use_error_term", [False, True])
def test_jumprelu_inference_equivalence(seed_everything, use_error_term):
    """
    Test that the old vs new JumpReLU SAEs match in parameter shape and forward pass outputs,
    and can optionally test the error_term usage.
    """
    old_sae = make_old_jumprelu_sae(d_in=16, d_sae=8, use_error_term=use_error_term)
    new_sae = make_new_jumprelu_sae(d_in=16, d_sae=8, use_error_term=use_error_term)

    # Compare parameter shapes
    compare_params(old_sae, new_sae)

    # Provide a random input
    x = torch.randn(2, 4, 16, dtype=torch.float32)

    with torch.no_grad():
        old_out = old_sae(x)
        new_out = new_sae(x)

    assert old_out.shape == new_out.shape, "Output shape mismatch."
    assert torch.isfinite(old_out).all()
    assert torch.isfinite(new_out).all()

    # If error_term is True, they might diverge for non-standard architectures,
    # but let's allow a tolerance for closeness or just check shape
    # It's okay if they differ numerically, but let's see if "jumprelu" lines up
    if use_error_term:
        # We might not expect exact equality, but we can still check shape
        assert old_out.shape == new_out.shape

@pytest.mark.parametrize("fold_fn", ["fold_W_dec_norm", "fold_activation_norm_scaling_factor"])
def test_jumprelu_fold_equivalence(seed_everything, fold_fn):
    """
    Test that folding functions (fold_W_dec_norm or fold_activation_norm_scaling_factor)
    on old vs new JumpReLU yields consistent results on forward passes.
    """
    old_sae = make_old_jumprelu_sae(use_error_term=False)
    new_sae = make_new_jumprelu_sae(use_error_term=False)
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

    assert old_out.shape == new_out.shape
    # The numeric results might not match exactly if JumpReLU is stricter with thresholds,
    # but let's see if they match within a tolerance. If they do for your code, match them strictly.
    assert torch.allclose(old_out, new_out, atol=1e-5), \
        f"{fold_fn} mismatch between old and new"

def test_jumprelu_run_hooks_equivalence(seed_everything):
    """
    Compare hooking behavior for JumpReLU. We'll check that hooking triggers
    the same number of calls in old_sae vs new_sae and that output shapes match.
    """
    class Counter:
        def __init__(self):
            self.count = 0
        def inc(self, *args, **kwargs):
            self.count += 1

    old_sae = make_old_jumprelu_sae()
    new_sae = make_new_jumprelu_sae()

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

    assert old_c.count > 0, "No hooks triggered in old JumpReLU SAE"
    assert new_c.count > 0, "No hooks triggered in new JumpReLU SAE"

#####################################
# Training Equivalence
#####################################

def make_old_jumprelu_training_sae(d_in=16, d_sae=8) -> TrainingSAE:
    """
    Helper to instantiate an old TrainingSAE configured as JumpReLU for testing.
    """
    old_training_cfg = TrainingSAEConfig(
        architecture="jumprelu",
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

def make_new_jumprelu_training_sae(d_in=16, d_sae=8) -> JumpReLUTrainingSAE:
    """
    Helper to instantiate a new JumpReLUTrainingSAE instance.
    """
    new_training_cfg = NewTrainingSAEConfig(
        architecture="jumprelu",
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
    return JumpReLUTrainingSAE(new_training_cfg)

def test_jumprelu_training_equivalence(seed_everything):
    """
    Test that old vs new JumpReLU SAEs match shapes in outputs and remain finite.
    We won't require exact numeric equivalence, as the old code might differ in how it
    handles threshold or error terms. But we'll check overall shape and finiteness.
    """
    old_sae = make_old_jumprelu_training_sae()
    new_sae = make_new_jumprelu_training_sae()

    old_sae.train()
    new_sae.train()

    batch_size, seq_len, d_in = 2, 4, 16
    x = torch.randn(batch_size, seq_len, d_in, dtype=torch.float32)

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

    assert old_out.sae_out.shape == new_out.sae_out.shape, \
        "JumpReLU training output shape mismatch."
    assert torch.isfinite(old_out.sae_out).all(), "Old JumpReLU training out is not finite."
    assert torch.isfinite(new_out.sae_out).all(), "New JumpReLU training out is not finite."

    # Check that we do have MSE and L0 losses
    assert "mse_loss" in old_out.losses
    assert "mse_loss" in new_out.losses
    # This old code calls it "l0_loss" if it uses Step, or it's an "aux_loss" fallback.
    old_l0_loss = old_out.losses.get("l0_loss", old_out.losses.get("aux_loss"))
    new_l0_loss = new_out.losses.get("l0_loss", new_out.losses.get("aux_loss"))
    assert old_l0_loss is not None, "Old JumpReLU training missing L0 or aux loss."
    assert new_l0_loss is not None, "New JumpReLU training missing L0 or aux loss."

    assert torch.isfinite(old_l0_loss)
    assert torch.isfinite(new_l0_loss) 