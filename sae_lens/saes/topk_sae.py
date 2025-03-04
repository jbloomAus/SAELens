"""Inference-only TopKSAE variant, similar in spirit to StandardSAE but using a TopK-based activation."""

import torch
from torch import nn
from jaxtyping import Float
from typing import Callable, Any, Optional

from sae_lens.saes.sae_base import BaseSAE, SAEConfig
from sae_lens.saes.sae_base import BaseTrainingSAE, TrainStepOutput

class TopK(nn.Module):
    """
    A simple TopK activation that zeroes out all but the top K elements along the last dimension,
    then optionally applies a post-activation function (e.g., ReLU).
    """
    def __init__(
        self,
        k: int,
        postact_fn: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
    ):
        super().__init__()
        self.k = k
        self.postact_fn = postact_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1) Select top K elements along the last dimension.
        2) Apply post-activation (often ReLU).
        3) Zero out all other entries.
        """
        topk = torch.topk(x, k=self.k, dim=-1)
        values = self.postact_fn(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result


class TopKSAE(BaseSAE):
    """
    An inference-only sparse autoencoder using a "topk" activation function.
    It uses linear encoder and decoder layers, applying the TopK activation
    to the hidden pre-activation in its encode step.
    """

    def __init__(self, cfg: SAEConfig, use_error_term: bool = False):
        """
        Args:
            cfg: SAEConfig defining model size and behavior.
            use_error_term: Whether to apply the error-term approach in the forward pass.
        """
        super().__init__(cfg, use_error_term)

    def initialize_weights(self) -> None:
        """
        Initializes weights and biases for encoder/decoder similarly to the standard SAE,
        that is:
          - b_enc, b_dec are zero-initialized
          - W_enc, W_dec are Kaiming Uniform
        """
        # encoder bias
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        # decoder bias
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

        # encoder weight
        w_enc_data = torch.empty(
            self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_enc_data)
        self.W_enc = nn.Parameter(w_enc_data)

        # decoder weight
        w_dec_data = torch.empty(
            self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_dec_data)
        self.W_dec = nn.Parameter(w_dec_data)

    def encode(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_sae"]:
        """
        Converts input x into feature activations.
        Uses topk activation from the config (cfg.activation_fn == "topk")
        under the hood.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        # The BaseSAE already sets self.activation_fn to TopK(...) if config requests topk.
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts

    def decode(self, feature_acts: Float[torch.Tensor, "... d_sae"]) -> Float[torch.Tensor, "... d_in"]:
        """
        Reconstructs the input from topk feature activations.
        Applies optional finetuning scaling, hooking to recons, out normalization,
        and optional head reshaping.
        """
        scaled_features = self.apply_finetuning_scaling_factor(feature_acts)
        sae_out_pre = scaled_features @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    def _get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self.cfg.activation_fn == "topk":
            if "k" not in self.cfg.activation_fn_kwargs:
                raise ValueError("TopK activation function requires a k value.")
            k = self.cfg.activation_fn_kwargs.get("k", 1)  # Default k to 1 if not provided
            postact_fn = self.cfg.activation_fn_kwargs.get(
                "postact_fn", nn.ReLU()
            )  # Default post-activation to ReLU if not provided
            return TopK(k, postact_fn)
        # Otherwise, return the "standard" handling from BaseSAE
        return super()._get_activation_fn()


class TopKTrainingSAE(BaseTrainingSAE):
    """
    TopK variant with training functionality. Injects noise during training, optionally
    calculates a topk-related auxiliary loss, etc.
    """

    def initialize_weights(self) -> None:
        """Very similar to TopKSAE, using zero biases + Kaiming Uniform weights."""
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

        w_enc_data = torch.empty(
            self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_enc_data)
        self.W_enc = nn.Parameter(w_enc_data)

        w_dec_data = torch.empty(
            self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_dec_data)
        self.W_dec = nn.Parameter(w_dec_data)

    def encode_with_hidden_pre(
        self,
        x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        """
        Similar to the base training method: cast input, optionally add noise, then apply TopK.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        # Inject noise if training
        if self.training and self.cfg.noise_scale > 0:
            hidden_pre_noised = hidden_pre + torch.randn_like(hidden_pre) * self.cfg.noise_scale
        else:
            hidden_pre_noised = hidden_pre

        # Apply the TopK activation function (already set in self.activation_fn if config is "topk")
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre_noised))
        return feature_acts, hidden_pre_noised

    def encode(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_sae"]:
        """
        For inference, just encode without returning hidden_pre. 
        (training_forward_pass calls encode_with_hidden_pre).
        """
        feature_acts, _ = self.encode_with_hidden_pre(x)
        return feature_acts

    def decode(
        self,
        feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Decodes feature activations back into input space, 
        applying optional finetuning scale, hooking, out normalization, etc.
        """
        scaled_features = self.apply_finetuning_scaling_factor(feature_acts)
        sae_out_pre = scaled_features @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)
    
    def calculate_aux_loss(
        self,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: Optional[torch.Tensor],
        current_l1_coefficient: float,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate auxiliary loss for TopK SAE.
        
        For TopK SAEs, the auxiliary loss is the topk auxiliary reconstruction loss,
        which encourages dead neurons to learn useful features.
        """
        # Standard L1 loss
        weighted_feature_acts = feature_acts
        if self.cfg.scale_sparsity_penalty_by_decoder_norm:
            weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
        
        sparsity = weighted_feature_acts.norm(p=self.cfg.lp_norm, dim=-1)
        l1_loss = (current_l1_coefficient * sparsity).mean()
        
        # Calculate auxiliary reconstruction loss if dead neurons present
        aux_recon_loss = torch.tensor(0.0, device=sae_in.device, dtype=sae_in.dtype)
        if dead_neuron_mask is not None:
            # ... topk specific implementation ...
            # This should match the logic from the tests
            k_aux = self.cfg.d_sae // 2  # Half the number of neurons
            aux_acts = self._calculate_topk_aux_acts(k_aux, hidden_pre, dead_neuron_mask)
            aux_out = aux_acts @ self.W_dec + self.b_dec
            aux_recon_loss = ((aux_out - sae_in) ** 2).sum(dim=-1).mean()
        
        # Update the losses dictionary in the parent class
        self.current_losses = {
            "l1_loss": l1_loss,
            "auxiliary_reconstruction_loss": aux_recon_loss,
        }
        
        return l1_loss + aux_recon_loss

    def _get_activation_fn(self):
        if self.cfg.activation_fn == "topk":
            if "k" not in self.cfg.activation_fn_kwargs:
                raise ValueError("TopK activation function requires a k value.")
            k = self.cfg.activation_fn_kwargs.get("k", 1)
            postact_fn = self.cfg.activation_fn_kwargs.get("postact_fn", nn.ReLU())
            return TopK(k, postact_fn)
        return super()._get_activation_fn()

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: Optional[torch.Tensor] = None,
    ) -> TrainStepOutput:
        """
        Forward pass specific to TopK SAE architecture, ensuring consistent loss dictionary keys.
        """
        # Encode and decode
        feature_acts, hidden_pre = self.encode_with_hidden_pre(sae_in)
        sae_out = self.decode(feature_acts)
        
        # MSE loss calculation
        per_item_mse_loss = self.mse_loss_fn(sae_out, sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()
        
        # TopK-specific auxiliary loss
        aux_loss = self.calculate_topk_aux_loss(
            sae_in=sae_in,
            sae_out=sae_out,
            hidden_pre=hidden_pre,
            dead_neuron_mask=dead_neuron_mask,
        )
        
        # Use consistent keys for losses
        losses = {
            "mse_loss": mse_loss,
            "auxiliary_reconstruction_loss": aux_loss,
        }
        
        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=mse_loss + aux_loss,
            losses=losses,
        )

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
        # Check if we have any dead neurons to work with
        if dead_neuron_mask is not None and (num_dead := int(dead_neuron_mask.sum())) > 0:
            residual = sae_in - sae_out

            # Heuristic from Appendix B.1 in the paper - use ~50% of features
            k_aux = hidden_pre.shape[-1] // 2

            # Reduce the scale of the loss if there are fewer dead neurons than k_aux
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Calculate the activations for the top-k dead neurons
            auxk_acts = self._calculate_topk_aux_acts(k_aux, hidden_pre, dead_neuron_mask)

            # Encourage the top dead latents to predict the residual
            recons = self.decode(auxk_acts)
            auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()
            return scale * auxk_loss
        
        # If no dead neurons or mask not provided, return zero loss
        return torch.tensor(0.0, device=self.device)

    def _calculate_topk_aux_acts(
        self,
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
        # Don't include living latents in this loss (set them to -inf so they won't be selected)
        auxk_latents = torch.where(dead_neuron_mask[None], hidden_pre, torch.tensor(-float('inf'), device=hidden_pre.device))
        
        # Find topk values among dead neurons
        auxk_topk = auxk_latents.topk(k_aux, dim=-1, sorted=False)
        
        # Create a tensor of zeros, then place the topk values at their proper indices
        auxk_acts = torch.zeros_like(hidden_pre)
        auxk_acts.scatter_(-1, auxk_topk.indices, auxk_topk.values)
        
        return auxk_acts 