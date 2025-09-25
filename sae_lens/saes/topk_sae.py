"""Inference-only TopKSAE variant, similar in spirit to StandardSAE but using a TopK-based activation."""

from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Float
from torch import nn
from transformer_lens.hook_points import HookPoint
from typing_extensions import override

from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainCoefficientConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
    _disable_hooks,
)


class SparseHookPoint(HookPoint):
    """
    A HookPoint that takes in a sparse tensor.
    Overrides TransformerLens's HookPoint.
    """

    def __init__(self, d_sae: int):
        super().__init__()
        self.d_sae = d_sae

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        using_hooks = (
            self._forward_hooks is not None
            or self._backward_hooks is not None
            and len(self._forward_hooks) > 0
            and len(self._backward_hooks) > 0
        )
        if using_hooks and x.is_sparse:
            return x.to_dense()
        return x  # if no hooks are being used, use passthrough


class TopK(nn.Module):
    """
    A simple TopK activation that zeroes out all but the top K elements along the last dimension,
    and applies ReLU to the top K elements.
    """

    use_sparse_activations: bool

    def __init__(
        self,
        k: int,
        use_sparse_activations: bool = True,
    ):
        super().__init__()
        self.k = k
        self.use_sparse_activations = use_sparse_activations

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        1) Select top K elements along the last dimension.
        2) Apply ReLU.
        3) Zero out all other entries.
        """
        topk_values, topk_indices = torch.topk(x, k=self.k, dim=-1, sorted=False)
        values = topk_values.relu()
        if self.use_sparse_activations:
            # Produce a COO sparse tensor (use sparse matrix multiply in decode)
            assert (
                x.ndim >= 2
            ), f"Expected pre-topK tensor to have at least 2 dimensions, got tensor of shape {x.shape}"
            x = x.view(-1, x.shape[-1])
            M, _ = x.shape
            sparse_indices = torch.stack(
                [
                    torch.arange(M, device=x.device).repeat_interleave(self.k),
                    topk_indices.flatten(),
                ]
            )
            return torch.sparse_coo_tensor(
                sparse_indices, values.flatten(), tuple(x.shape)
            )
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, values)
        return result


@dataclass
class TopKSAEConfig(SAEConfig):
    """
    Configuration class for a TopKSAE.
    """

    k: int = 100
    use_sparse_activations: bool = True

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class TopKSAE(SAE[TopKSAEConfig]):
    """
    An inference-only sparse autoencoder using a "topk" activation function.
    It uses linear encoder and decoder layers, applying the TopK activation
    to the hidden pre-activation in its encode step.
    """

    b_enc: nn.Parameter

    def __init__(self, cfg: TopKSAEConfig, use_error_term: bool = False):
        """
        Args:
            cfg: SAEConfig defining model size and behavior.
            use_error_term: Whether to apply the error-term approach in the forward pass.
        """
        super().__init__(cfg, use_error_term)
        if self.cfg.use_sparse_activations:
            self.hook_sae_acts_post = SparseHookPoint(self.cfg.d_sae)
        self.setup()

    @override
    def initialize_weights(self) -> None:
        # Initialize encoder weights and bias.
        super().initialize_weights()
        _init_weights_topk(self)

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Converts input x into feature activations.
        Uses topk activation under the hood.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        # The BaseSAE already sets self.activation_fn to TopK(...) if config requests topk.
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    def decode(
        self,
        feature_acts: Float[torch.Tensor, "... d_sae"],
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Reconstructs the input from topk feature activations.
        Applies optional finetuning scaling, hooking to recons, out normalization,
        and optional head reshaping.
        """
        if self.cfg.use_sparse_activations and feature_acts.ndim >= 3:
            raise ValueError(
                "Sparse activations are only supported for 2D activations. Use .disable_sparse_activations() to support arbitrary activation dims."
            )
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    def disable_sparse_activations(self) -> None:
        self.cfg.use_sparse_activations = False
        if isinstance(self.activation_fn, TopK):
            self.activation_fn.use_sparse_activations = False

    def enable_sparse_activations(self) -> None:
        self.cfg.use_sparse_activations = True
        if isinstance(self.activation_fn, TopK):
            self.activation_fn.use_sparse_activations = True

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SAE."""
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        if self.use_error_term:
            with torch.no_grad():
                # Recompute without hooks for true error term
                with _disable_hooks(self):
                    feature_acts_clean = self.encode(x)
                    x_reconstruct_clean = self.decode(feature_acts_clean)
                sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            sae_out = sae_out + sae_error

        return self.hook_sae_output(sae_out)

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return TopK(self.cfg.k, use_sparse_activations=self.cfg.use_sparse_activations)

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        raise NotImplementedError(
            "Folding W_dec_norm is not safe for TopKSAEs, as this may change the topk activations"
        )


@dataclass
class TopKTrainingSAEConfig(TrainingSAEConfig):
    """
    Configuration class for training a TopKTrainingSAE.
    """

    k: int = 100
    use_sparse_activations: bool = True
    aux_loss_coefficient: float = 1.0

    @override
    @classmethod
    def architecture(cls) -> str:
        return "topk"


class TopKTrainingSAE(TrainingSAE[TopKTrainingSAEConfig]):
    """
    TopK variant with training functionality. Calculates a topk-related auxiliary loss, etc.
    """

    b_enc: nn.Parameter

    def __init__(self, cfg: TopKTrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        if self.cfg.use_sparse_activations:
            self.hook_sae_acts_post = SparseHookPoint(self.cfg.d_sae)
        self.setup()

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        _init_weights_topk(self)

    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        """
        Similar to the base training method: calculate pre-activations, then apply TopK.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        # Apply the TopK activation function (already set in self.activation_fn if config is "topk")
        if self.cfg.use_sparse_activations and isinstance(
            self.hook_sae_acts_post,
            SparseHookPoint,
        ):
            feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        else:
            feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts, hidden_pre

    @override
    def decode(
        self,
        feature_acts: Float[torch.Tensor, "... d_sae"],
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Decodes feature activations back into input space,
        applying optional finetuning scale, hooking, out normalization, etc.
        """
        if self.cfg.use_sparse_activations and feature_acts.ndim >= 3:
            raise ValueError(
                "Sparse activations are only supported for 2D activations. Use .disable_sparse_activations() to support arbitrary activation dims."
            )
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    def disable_sparse_activations(self) -> None:
        self.cfg.use_sparse_activations = False
        if isinstance(self.activation_fn, TopK):
            self.activation_fn.use_sparse_activations = False

    def enable_sparse_activations(self) -> None:
        self.cfg.use_sparse_activations = True
        if isinstance(self.activation_fn, TopK):
            self.activation_fn.use_sparse_activations = True

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SAE."""
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        if self.use_error_term:
            with torch.no_grad():
                # Recompute without hooks for true error term
                with _disable_hooks(self):
                    feature_acts_clean = self.encode(x)
                    x_reconstruct_clean = self.decode(feature_acts_clean)
                sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            sae_out = sae_out + sae_error

        return self.hook_sae_output(sae_out)

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Calculate the auxiliary loss for dead neurons
        topk_loss = self.calculate_topk_aux_loss(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            hidden_pre=hidden_pre,
            dead_neuron_mask=step_input.dead_neuron_mask,
        )
        return {"auxiliary_reconstruction_loss": topk_loss}

    @override
    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        raise NotImplementedError(
            "Folding W_dec_norm is not safe for TopKSAEs, as this may change the topk activations"
        )

    @override
    def get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return TopK(self.cfg.k, use_sparse_activations=self.cfg.use_sparse_activations)

    @override
    def get_coefficients(self) -> dict[str, TrainCoefficientConfig | float]:
        return {}

    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Calculate TopK auxiliary loss.

        This auxiliary loss encourages dead neurons to learn useful features by having
        them reconstruct the residual error from the live neurons. It's a key part of
        preventing neuron death in TopK SAEs.
        """
        # Mostly taken from https://github.com/EleutherAI/sae/blob/main/sae/sae.py, except without variance normalization
        # NOTE: checking the number of dead neurons will force a GPU sync, so performance can likely be improved here
        if dead_neuron_mask is None or (num_dead := int(dead_neuron_mask.sum())) == 0:
            return sae_out.new_tensor(0.0)
        residual = (sae_in - sae_out).detach()

        # Heuristic from Appendix B.1 in the paper
        k_aux = sae_in.shape[-1] // 2

        # Reduce the scale of the loss if there are a small number of dead latents
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        auxk_acts = _calculate_topk_aux_acts(
            k_aux=k_aux,
            hidden_pre=hidden_pre,
            dead_neuron_mask=dead_neuron_mask,
        )

        # Encourage the top ~50% of dead latents to predict the residual of the
        # top k living latents
        recons = self.decode(auxk_acts)
        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()
        return self.cfg.aux_loss_coefficient * scale * auxk_loss


def _calculate_topk_aux_acts(
    k_aux: int,
    hidden_pre: torch.Tensor,
    dead_neuron_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Helper method to calculate activations for the auxiliary loss.

    Args:
        k_aux: Number of top dead neurons to select
        hidden_pre: Pre-activation values from encoder
        dead_neuron_mask: Boolean mask indicating which neurons are dead

    Returns:
        Tensor with activations for only the top-k dead neurons, zeros elsewhere
    """

    # Don't include living latents in this loss
    auxk_latents = torch.where(dead_neuron_mask[None], hidden_pre, -torch.inf)
    # Top-k dead latents
    auxk_topk = auxk_latents.topk(k_aux, sorted=False)
    # Set the activations to zero for all but the top k_aux dead latents
    auxk_acts = torch.zeros_like(hidden_pre)
    auxk_acts.scatter_(-1, auxk_topk.indices, auxk_topk.values)
    # Set activations to zero for all but top k_aux dead latents
    return auxk_acts


def _init_weights_topk(
    sae: SAE[TopKSAEConfig] | TrainingSAE[TopKTrainingSAEConfig],
) -> None:
    sae.b_enc = nn.Parameter(
        torch.zeros(sae.cfg.d_sae, dtype=sae.dtype, device=sae.device)
    )
