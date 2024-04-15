from typing import Callable, Protocol

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from sae_lens import SparseAutoencoder


class TransformerLensForwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        raise NotImplementedError


class TransformerLensBackwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> tuple[torch.Tensor]:
        raise NotImplementedError


ForwardHookData = tuple[str, TransformerLensForwardHook]
BackwardHookData = tuple[str, TransformerLensBackwardHook]


class SAEPatchHook:
    """A hook for a HookedTransformer to patch an SAE into the computational graph"""

    sae: SparseAutoencoder
    sae_feature_acts: Float[torch.Tensor, "n_batch n_token d_sae"]
    sae_errors: Float[torch.Tensor, "n_batch n_token d_model"]

    def __init__(self, sae: SparseAutoencoder):
        self.sae = sae
        self.sae_feature_acts = torch.tensor([])
        self.sae_errors = torch.tensor([])

    def _forward_hook_fn(
        self,
        orig: Float[torch.Tensor, "n_batch n_token d_model"],
        hook: HookPoint,
    ) -> Float[torch.Tensor, "n_batch n_token d_model"]:
        """Forward hook to patch the SAE into the computational graph

        orig: Original activations
        """

        a_orig = orig
        z_sae = self.sae.encode(a_orig)
        # keep pyright happy
        assert isinstance(z_sae, torch.Tensor)
        a_sae = self.sae.decode(z_sae)
        a_err = a_orig - a_sae.detach()

        # Track the gradients
        assert z_sae.requires_grad
        z_sae.retain_grad()
        assert a_err.requires_grad
        a_err.retain_grad()

        # Store values for later use
        self.sae_feature_acts = z_sae
        self.sae_errors = a_err

        a_rec = a_sae + a_err
        return a_rec

    def _backward_hook_fn(
        self,
        orig: Float[torch.Tensor, "n_batch n_token d_model"],
        hook: HookPoint,
    ) -> tuple[Float[torch.Tensor, "n_batch n_token d_model"]]:
        """Implement pass-through gradients

        orig: gradient w.r.t output
        return: gradient w.r.t input
        """
        # NOTE: Transformer lens 1.5.4 un-tuples the gradient before passing it in
        # So we need to re-tuple it here
        return (orig,)

    def get_forward_hook(self) -> ForwardHookData:
        """Return a forward hook that patches the activation."""
        return (self.sae.cfg.hook_point, self._forward_hook_fn)

    def get_backward_hook(self) -> BackwardHookData:
        """Return a backward hook that patches the gradients."""
        return (self.sae.cfg.hook_point, self._backward_hook_fn)

    def get_ie_atp_of_sae_features(
        self, z_patch: Float[torch.Tensor, "n_batch n_token d_sae"]
    ) -> Float[torch.Tensor, "n_batch n_token d_sae"]:
        grad = self.sae_feature_acts.grad
        val = self.sae_feature_acts
        return grad * (z_patch - val)

    def get_ie_atp_of_sae_errors(
        self, a_patch: Float[torch.Tensor, "n_batch n_token d_model"]
    ) -> Float[torch.Tensor, "n_batch n_token d_model"]:
        grad = self.sae_errors.grad
        val = self.sae_errors.grad
        return grad * (a_patch - val)

    # TODO: implement integrated gradients


def compute_indirect_effect(
    model: HookedTransformer,
    sae: SparseAutoencoder,
    metric_fn: Callable[[HookedTransformer, str], Float[torch.Tensor, ""]],
    x_orig: str,
    z_patch: Float[torch.Tensor, "n_batch n_token d_sae"] | None = None,
    err_patch: Float[torch.Tensor, "n_batch n_token d_model"] | None = None,
) -> tuple[
    Float[torch.Tensor, "n_batch n_token d_sae"],
    Float[torch.Tensor, "n_batch n_token d_model"],
]:
    sae_patch_hook = SAEPatchHook(sae)
    with model.hooks(fwd_hooks=[sae_patch_hook.get_forward_hook()]):
        metric = metric_fn(model, x_orig)
        metric.backward()

    if z_patch is None:
        z_patch = torch.zeros_like(sae_patch_hook.sae_feature_acts)
    if err_patch is None:
        err_patch = torch.zeros_like(sae_patch_hook.sae_errors)

    ie_atp_feat = sae_patch_hook.get_ie_atp_of_sae_features(z_patch)
    ie_atp_err = sae_patch_hook.get_ie_atp_of_sae_errors(err_patch)
    return ie_atp_feat, ie_atp_err
