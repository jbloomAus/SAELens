from typing import cast

import pytest
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer

from sae_lens import SparseAutoencoder
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes
from sae_lens.toolkit.sae_patcher import SAEPatcher

device = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def model() -> HookedTransformer:
    return cast(HookedTransformer, HookedTransformer.from_pretrained("gpt2").to(device))


@pytest.fixture(scope="module")
def sae() -> SparseAutoencoder:
    return get_gpt2_res_jb_saes()[0]["blocks.8.hook_resid_pre"].to(device)


@pytest.fixture(scope="module")
def prompt() -> str:
    return "Hello world"


def test_sae_patcher_hook_forward_hook_only(
    model: HookedTransformer, sae: SparseAutoencoder, prompt: str
):
    orig_loss = model(prompt, return_type="loss")
    sae_patcher = SAEPatcher(sae)

    with model.hooks(
        fwd_hooks=[sae_patcher.get_forward_hook()],
    ):
        patched_loss = model(prompt, return_type="loss")

    assert torch.isclose(orig_loss, patched_loss, atol=1e-6)


def test_sae_patcher_backward_hook_only(
    model: HookedTransformer, sae: SparseAutoencoder, prompt: str
):
    orig_loss = model(prompt, return_type="loss")
    sae_patcher = SAEPatcher(sae)

    with model.hooks(
        bwd_hooks=[sae_patcher.get_backward_hook()],
    ):
        patched_loss = model(prompt, return_type="loss")

    assert torch.isclose(orig_loss, patched_loss, atol=1e-6)


def get_grads(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    grads = []
    for name, param in model.named_parameters():
        assert param.grad is not None
        grads.append((name, param.grad))
    return grads


def test_sae_patcher_preserves_cached_model_activations(
    model: HookedTransformer, sae: SparseAutoencoder, prompt: str
):
    _, orig_cache = model.run_with_cache(prompt, return_type="loss")
    sae_patcher = SAEPatcher(sae)

    with model.hooks(
        fwd_hooks=[sae_patcher.get_forward_hook()],
        bwd_hooks=[sae_patcher.get_backward_hook()],
    ):
        _, patched_cache = model.run_with_cache(prompt, return_type="loss")

    for orig_name, patched_name in zip(orig_cache, patched_cache):
        assert orig_name == patched_name
        orig_act = orig_cache[orig_name]
        patched_act = patched_cache[patched_name]
        assert torch.allclose(orig_act, patched_act, atol=1e-5)


def test_sae_patcher_preserves_model_grad(
    model: HookedTransformer, sae: SparseAutoencoder, prompt: str
):
    orig_loss = model(prompt, return_type="loss")
    orig_loss.backward()
    orig_grads = get_grads(model)
    sae_patcher = SAEPatcher(sae)

    with model.hooks(
        fwd_hooks=[sae_patcher.get_forward_hook()],
        bwd_hooks=[sae_patcher.get_backward_hook()],
    ):
        patched_loss = model(prompt, return_type="loss")
        patched_loss.backward()
        patched_grads = get_grads(model)

    for (orig_name, orig_grad), (patched_name, patched_grad) in zip(
        orig_grads, patched_grads
    ):
        assert orig_name == patched_name
        assert torch.allclose(orig_grad, patched_grad, atol=1e-5)


def test_sae_patcher_fields_have_grad(
    model: HookedTransformer, sae: SparseAutoencoder, prompt: str
):
    sae_patcher = SAEPatcher(sae)
    assert sae_patcher.sae_feature_acts.grad is None
    assert sae_patcher.sae_errors.grad is None

    with model.hooks(
        fwd_hooks=[sae_patcher.get_forward_hook()],
        bwd_hooks=[sae_patcher.get_backward_hook()],
    ):
        patched_loss = model(prompt, return_type="loss")
        patched_loss.backward()

    assert sae_patcher.sae_feature_acts.grad is not None
    assert sae_patcher.sae_errors.grad is not None


def test_sae_patcher_get_node_values(
    model: HookedTransformer, sae: SparseAutoencoder, prompt: str
):

    # Get the batch size and number of tokens hackily
    loss = model(prompt, return_type="loss", loss_per_token=True)
    assert len(loss.shape) == 2
    n_batch, n_token = loss.shape
    # Next-token prediction loss shape is actually n_token-1 so we add one
    n_token += 1

    sae_patcher = SAEPatcher(sae)
    with model.hooks(
        fwd_hooks=[sae_patcher.get_forward_hook()],
        bwd_hooks=[sae_patcher.get_backward_hook()],
    ):
        patched_loss = model(prompt, return_type="loss")
        patched_loss.backward()

    feature_node_vals = sae_patcher.get_node_values("feature")
    assert feature_node_vals.shape == (n_batch, n_token, sae.d_sae)
    error_node_vals = sae_patcher.get_node_values("error")
    assert error_node_vals.shape == (n_batch, n_token, sae.d_in)
    all_node_vals = sae_patcher.get_node_values("all")
    assert all_node_vals.shape == (n_batch, n_token, sae.d_sae + sae.d_in)


def test_sae_patcher_get_node_grads(
    model: HookedTransformer, sae: SparseAutoencoder, prompt: str
):

    # Get the batch size and number of tokens hackily
    loss = model(prompt, return_type="loss", loss_per_token=True)
    assert len(loss.shape) == 2
    n_batch, n_token = loss.shape
    # Next-token prediction loss shape is actually n_token-1 so we add one
    n_token += 1

    sae_patcher = SAEPatcher(sae)
    with model.hooks(
        fwd_hooks=[sae_patcher.get_forward_hook()],
        bwd_hooks=[sae_patcher.get_backward_hook()],
    ):
        patched_loss = model(prompt, return_type="loss")
        patched_loss.backward()

    feature_node_grads = sae_patcher.get_node_grads("feature")
    assert feature_node_grads.shape == (n_batch, n_token, sae.d_sae)
    error_node_grads = sae_patcher.get_node_grads("error")
    assert error_node_grads.shape == (n_batch, n_token, sae.d_in)
    all_node_grads = sae_patcher.get_node_grads("all")
    assert all_node_grads.shape == (n_batch, n_token, sae.d_sae + sae.d_in)
