from typing import Literal, Protocol

import torch
from jaxtyping import Float
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

NodeType = Literal["feature", "error", "all"]


class SAEPatcher:
    """Patches an SAE into the computational graph of a HookedTransformer

    Usage:

    sae_patcher = SAEPatcher(sae)
    with model.hooks(
        fwd_hooks=[sae_patcher.get_forward_hook()]
        bwd_hooks=[sae_patcher.get_backward_hook()]
    ):
        ...
    """

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

    @property
    def sae_nodes(self) -> Float[torch.Tensor, "n_batch n_token (d_sae + d_model)"]:
        return torch.cat([self.sae_feature_acts, self.sae_errors], dim=-1)

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

    def get_node_values(
        self, node_type: NodeType = "all"
    ) -> Float[torch.Tensor, "n_batch n_token n_nodes"]:
        if node_type == "feature":
            return self.sae_feature_acts.detach()
        elif node_type == "error":
            return self.sae_errors.detach()
        elif node_type == "all":
            return torch.cat([self.sae_feature_acts, self.sae_errors], dim=-1).detach()
        else:
            raise ValueError(f"Invalid node_type: {node_type}")

    def get_node_grads(
        self, node_type: NodeType = "all"
    ) -> Float[torch.Tensor, "n_batch n_token n_nodes"]:

        if node_type == "feature":
            if self.sae_feature_acts.grad is None:
                raise RuntimeError("Gradients are not available.")
            return self.sae_feature_acts.grad.detach()
        elif node_type == "error":
            if self.sae_errors.grad is None:
                raise RuntimeError("Gradients are not available.")
            return self.sae_errors.grad.detach()
        elif node_type == "all":
            if self.sae_feature_acts.grad is None or self.sae_errors.grad is None:
                raise RuntimeError("Gradients are not available.")
            return torch.cat(
                [self.sae_feature_acts.grad, self.sae_errors.grad], dim=-1
            ).detach()
        else:
            raise ValueError(f"Invalid node_type: {node_type}")
