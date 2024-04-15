from dataclasses import dataclass
from typing import Callable, Protocol

import torch
from jaxtyping import Float
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from sae_lens import SparseAutoencoder, SparseAutoencoderDictionary


class TransformerLensForwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> torch.Tensor:
        raise NotImplementedError


class TransformerLensBackwardHook(Protocol):
    def __call__(self, orig: torch.Tensor, hook: HookPoint) -> tuple[torch.Tensor]:
        raise NotImplementedError


ForwardHookData = tuple[str, TransformerLensForwardHook]
BackwardHookData = tuple[str, TransformerLensBackwardHook]


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


def indirect_effect_attrib(
    t_orig: Float[torch.Tensor, "n_batch n_token d"],
    t_patch: Float[torch.Tensor, "n__batch n_token d"],
    grad_m_t_at_t_orig: Float[torch.Tensor, "n_batch n_token d"],
):
    return grad_m_t_at_t_orig * (t_patch - t_orig)


@dataclass
class SAELayerAttributionScores:
    name: str
    feature_scores: Float[torch.Tensor, "n_batch n_token d_sae"]
    error_scores: Float[torch.Tensor, "n_batch n_token d_model"]


def get_sae_features_and_errors_for_input(
    model: HookedTransformer,
    sae_dict: SparseAutoencoderDictionary,
    metric_fn: Callable[[HookedTransformer, str], Float[torch.Tensor, ""]],
    x: str,
) -> tuple[
    dict[str, Float[torch.Tensor, "n_batch n_token d_sae"]],
    dict[str, Float[torch.Tensor, "n_batch n_token d_model"]],
]:
    # Run model on original input
    patchers = {name: SAEPatcher(sae) for name, sae in sae_dict}
    with model.hooks(
        fwd_hooks=[p.get_forward_hook() for p in patchers.values()],
        bwd_hooks=[p.get_backward_hook() for p in patchers.values()],
    ):
        metric = metric_fn(model, x)
        metric.backward()

    # Collate the sae features and errors
    orig_features = {
        name: patcher.sae_feature_acts for name, patcher in patchers.items()
    }
    orig_errors = {name: patcher.sae_errors for name, patcher in patchers.items()}
    return orig_features, orig_errors


def compute_node_indirect_effect(
    model: HookedTransformer,
    sae_dict: SparseAutoencoderDictionary,
    metric_fn: Callable[[HookedTransformer, str], Float[torch.Tensor, ""]],
    x_orig: str,
    x_patch: str | None = None,
) -> tuple[
    dict[str, Float[torch.Tensor, "n_batch n_token d_sae"]],
    dict[str, Float[torch.Tensor, "n_batch n_token d_model"]],
]:
    """Compute node indirect effects for a given input

    Here, nodes are the features and errors of the SAEs
    and the indirect effect is computed using first-order Taylor approximation.

    Returns:
    - feature_scores: The scores for each feature.
      A dict of {name: [batch, token, d_sae]} of feature scores
    - error_scores: The scores for each error
    """
    orig_features, orig_errors = get_sae_features_and_errors_for_input(
        model, sae_dict, metric_fn, x_orig
    )

    # If no patch is provided, return zeros
    if x_patch is None:
        patch_features = {
            name: torch.zeros_like(act) for name, act in orig_features.items()
        }
        patch_errors = {
            name: torch.zeros_like(err) for name, err in orig_errors.items()
        }
    else:
        patch_features, patch_errors = get_sae_features_and_errors_for_input(
            model, sae_dict, metric_fn, x_patch
        )

    # Compute feature scores
    feature_scores = {}
    for name, orig_act in orig_features.items():
        grad_m_orig_act = orig_act.grad
        assert grad_m_orig_act is not None
        patch_act = patch_features[name]
        feature_scores[name] = indirect_effect_attrib(
            orig_act, patch_act, grad_m_orig_act
        )

    # Compute error scores
    error_scores = {}
    for name, orig_err in orig_errors.items():
        grad_m_orig_err = orig_err.grad
        assert grad_m_orig_err is not None
        patch_err = patch_errors[name]
        error_scores[name] = indirect_effect_attrib(
            orig_err, patch_err, grad_m_orig_err
        )

    return feature_scores, error_scores


# def compute_edge_indirect_effect(
#     model: HookedTransformer,
#     sae_dict: SparseAutoencoderDictionary,
#     metric_fn: Callable[[HookedTransformer, str], Float[torch.Tensor, ""]],
#     x_orig: str,
#     x_patch: str | None = None,
# ) -> tuple[
#     dict[str, Float[torch.Tensor, "n_batch n_token d_sae"]],
#     dict[str, Float[torch.Tensor, "n_batch n_token d_model"]],
# ]:
#     pass
