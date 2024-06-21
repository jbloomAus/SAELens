import einops
import pytest
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from sae_lens import HookedSAETransformer
from sae_lens.sae import Transcoder, TranscoderConfig

MODEL = "solu-1l"
prompt = "Hello World!"


class Counter:
    def __init__(self):
        self.count = 0

    def inc(self, *args, **kwargs):  # type: ignore
        self.count += 1


@pytest.fixture(scope="module")
def model():
    model = HookedSAETransformer.from_pretrained(MODEL, device="cpu")
    yield model
    model.reset_saes()


@pytest.fixture(scope="module")
def original_logits(model: HookedTransformer):
    return model(prompt)


def get_hooked_mlp_transcoder(model: HookedTransformer, layer: int) -> Transcoder:
    """Helper function to get a hooked MLP transcoder for a given layer of the model."""
    site_to_size = {
        "ln2.hook_normalized": model.cfg.d_model,
        "hook_mlp_out": model.cfg.d_model,
    }

    site_in = "ln2.hook_normalized"
    site_out = "hook_mlp_out"
    hook_name = f"blocks.{layer}.{site_in}"
    hook_name_out = f"blocks.{layer}.{site_out}"
    d_in = site_to_size[site_in]
    d_out = site_to_size[site_out]

    tc_cfg = TranscoderConfig(
        d_in=d_in,
        d_out=d_out,
        d_sae=d_in * 2,
        dtype="float32",
        device="cpu",
        model_name=MODEL,
        hook_name=hook_name,
        hook_name_out=hook_name_out,
        hook_layer=layer,
        hook_layer_out=layer,
        hook_head_index=None,
        hook_head_index_out=None,
        activation_fn_str="relu",
        prepend_bos=True,
        context_size=128,
        dataset_path="test",
        apply_b_dec_to_input=False,
        finetuning_scaling_factor=False,
        sae_lens_training_version=None,
        normalize_activations="none",
    )

    return Transcoder(tc_cfg)


@pytest.fixture(
    scope="module",
)
def hooked_transcoder(
    model: HookedTransformer,
) -> Transcoder:
    return get_hooked_mlp_transcoder(model, 0)


def test_forward_reconstructs_input(
    model: HookedTransformer, hooked_transcoder: Transcoder
):
    """Verfiy that the Transcoder returns an output with the same shape as the input activations."""

    # NOTE: In general, we do not expect the output of the transcoder to be equal to the input activations.
    # However, for MLP transcoders specifically, the shapes do match.
    act_name = hooked_transcoder.cfg.hook_name
    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    sae_output = hooked_transcoder(x)
    assert sae_output.shape == x.shape


def test_run_with_cache(model: HookedTransformer, hooked_transcoder: Transcoder):
    """Verifies that run_with_cache caches Transcoder activations"""

    act_name = hooked_transcoder.cfg.hook_name
    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    sae_output, cache = hooked_transcoder.run_with_cache(x)
    assert sae_output.shape == x.shape

    assert "hook_sae_input" in cache
    assert "hook_sae_acts_pre" in cache
    assert "hook_sae_acts_post" in cache
    assert "hook_sae_recons" in cache
    assert "hook_sae_output" in cache


def test_run_with_hooks(model: HookedTransformer, hooked_transcoder: Transcoder):
    """Verifies that run_with_hooks works with Transcoder activations"""

    c = Counter()
    act_name = hooked_transcoder.cfg.hook_name

    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    sae_hooks = [
        "hook_sae_input",
        "hook_sae_acts_pre",
        "hook_sae_acts_post",
        "hook_sae_recons",
        "hook_sae_output",
    ]

    sae_output = hooked_transcoder.run_with_hooks(
        x, fwd_hooks=[(sae_hook_name, c.inc) for sae_hook_name in sae_hooks]
    )
    assert sae_output.shape == x.shape

    assert c.count == len(sae_hooks)


@pytest.mark.xfail
def test_error_term(model: HookedTransformer, hooked_transcoder: Transcoder):
    """Verifies that that if we use error_terms, HookedTranscoder returns an output that is equal tdef test_feature_grads_with_error_term(model: HookedTransformer, hooked_transcoder: SparseAutoencoderBase):
    o the input activations."""

    act_name = hooked_transcoder.cfg.hook_name
    hooked_transcoder.use_error_term = True

    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    sae_output = hooked_transcoder(x)
    assert sae_output.shape == x.shape
    assert torch.allclose(sae_output, x, atol=1e-6)

    """Verifies that pytorch backward computes the correct feature gradients when using error_terms. Motivated by the need to compute feature gradients for attribution patching."""

    act_name = hooked_transcoder.cfg.hook_name
    hooked_transcoder.use_error_term = True

    # Get input activations
    _, cache = model.run_with_cache(prompt, names_filter=act_name)
    x = cache[act_name]

    # Cache gradients with respect to feature acts
    hooked_transcoder.reset_hooks()
    grad_cache = {}

    def backward_cache_hook(act: torch.Tensor, hook: HookPoint):
        grad_cache[hook.name] = act.detach()

    hooked_transcoder.add_hook("hook_sae_acts_post", backward_cache_hook, "bwd")
    hooked_transcoder.add_hook("hook_sae_output", backward_cache_hook, "bwd")

    sae_output = hooked_transcoder(x)
    assert torch.allclose(sae_output, x, atol=1e-6)
    value = sae_output.sum()
    value.backward()
    hooked_transcoder.reset_hooks()

    # Compute gradient analytically
    if act_name.endswith("hook_z"):
        reshaped_output_grad = einops.rearrange(
            grad_cache["hook_sae_output"], "... n_heads d_head -> ... (n_heads d_head)"
        )
        analytic_grad = reshaped_output_grad @ hooked_transcoder.W_dec.T
    else:
        analytic_grad = grad_cache["hook_sae_output"] @ hooked_transcoder.W_dec.T

    # Compare analytic gradient with pytorch computed gradient
    assert torch.allclose(grad_cache["hook_sae_acts_post"], analytic_grad, atol=1e-6)
