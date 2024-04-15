from typing import Protocol

import torch
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint

from sae_lens import SparseAutoencoder


class TransformerLensHook(Protocol):
    def __call__(
        self, original_activations: torch.Tensor, hook: HookPoint
    ) -> torch.Tensor:
        raise NotImplementedError


class SAEPatchHook(TransformerLensHook):
    """A hook for a HookedTransformer to patch an SAE into the computational graph"""

    sae: SparseAutoencoder
    sae_feature_acts: Float[torch.Tensor, "n_batch n_token d_sae"]
    sae_errors: Float[torch.Tensor, "n_batch n_token d_model"]

    def __init__(self, sae: SparseAutoencoder):
        self.sae = sae
        self.sae_feature_acts = torch.tensor([])
        self.sae_errors = torch.tensor([])

    def __call__(
        self,
        original_activations: Float[torch.Tensor, "n_batch n_token d_model"],
        hook: HookPoint,
    ) -> Float[torch.Tensor, "n_batch n_token d_model"]:

        a_orig = original_activations
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
