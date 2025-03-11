from typing import Any, Optional

import torch
from jaxtyping import Float
from torch import nn

from sae_lens.saes.sae_base import (
    BaseSAE,
    BaseTrainingSAE,
    SAEConfig,
    TrainingSAEConfig,
    TrainStepOutput,
)


class GatedSAE(BaseSAE):
    """
    GatedSAE is an inference-only implementation of a Sparse Autoencoder (SAE)
    using a gated linear encoder and a standard linear decoder.
    """

    def __init__(self, cfg: SAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        # Ensure b_enc does not exist for the gated architecture
        self.b_enc = None

    def initialize_weights(self) -> None:
        """
        Initialize weights exactly as in the original SAE class for gated architecture.
        """
        # Use the same initialization methods and values as in original SAE
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
                )
            )
        )
        self.b_gate = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        # Ensure r_mag is initialized to zero as in original
        self.r_mag = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.b_mag = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        # Decoder parameters with same initialization as original
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
        )
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

        # after defining b_gate, b_mag, etc.:
        self.b_enc = None

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Encode the input tensor into the feature space using a gated encoder.
        This must match the original encode_gated implementation from SAE class.
        """
        # Preprocess the SAE input (casting type, applying hooks, normalization)
        sae_in = self.process_sae_in(x)

        # Gating path exactly as in original SAE.encode_gated
        gating_pre_activation = sae_in @ self.W_enc + self.b_gate
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path (weight sharing with gated encoder)
        magnitude_pre_activation = self.hook_sae_acts_pre(
            sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
        )
        feature_magnitudes = self.activation_fn(magnitude_pre_activation)

        # Combine gating and magnitudes
        return self.hook_sae_acts_post(active_features * feature_magnitudes)

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Decode the feature activations back into the input space:
          1) Apply optional finetuning scaling.
          2) Linear transform plus bias.
          3) Run any reconstruction hooks and out-normalization if configured.
          4) If the SAE was reshaping hook_z activations, reshape back.
        """
        # 1) optional finetuning scaling
        scaled_features = self.apply_finetuning_scaling_factor(feature_acts)
        # 2) linear transform
        sae_out_pre = scaled_features @ self.W_dec + self.b_dec
        # 3) hooking and normalization
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        # 4) reshape if needed (hook_z)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @torch.no_grad()
    def fold_W_dec_norm(self):
        """Override to handle gated-specific parameters."""
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T

        # Gated-specific parameters need special handling
        self.r_mag.data = self.r_mag.data * W_dec_norms.squeeze()
        self.b_gate.data = self.b_gate.data * W_dec_norms.squeeze()
        self.b_mag.data = self.b_mag.data * W_dec_norms.squeeze()

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Normalize decoder columns to unit norm."""
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def initialize_decoder_norm_constant_norm(self, norm: float = 0.1):
        """Initialize decoder with constant norm."""
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data *= norm


class GatedTrainingSAE(BaseTrainingSAE):
    """
    GatedTrainingSAE is a concrete implementation of BaseTrainingSAE for the "gated" SAE architecture.
    It implements:
      - initialize_weights: sets up gating parameters (as in GatedSAE) plus optional training-specific init.
      - encode: calls encode_with_hidden_pre (standard training approach).
      - decode: linear transformation + hooking, same as GatedSAE or StandardTrainingSAE.
      - encode_with_hidden_pre: gating logic + optional noise injection for training.
      - calculate_aux_loss: includes an auxiliary reconstruction path and gating-based sparsity penalty.
      - training_forward_pass: calls encode_with_hidden_pre, decode, and sums up MSE + gating losses.
    """

    def __init__(self, cfg: TrainingSAEConfig, use_error_term: bool = False):
        if use_error_term:
            raise ValueError(
                "GatedSAE does not support `use_error_term`. Please set `use_error_term=False`."
            )
        super().__init__(cfg, use_error_term)
        # Also ensure no b_enc for the training version
        self.b_enc = None

    def initialize_weights(self) -> None:
        # Reuse the gating parameter initialization from GatedSAE:
        GatedSAE.initialize_weights(self)  # type: ignore

        # Additional training-specific logic, e.g. orthogonal init or heuristics:
        if self.cfg.decoder_orthogonal_init:
            self.W_dec.data = nn.init.orthogonal_(self.W_dec.data.T).T
        elif self.cfg.decoder_heuristic_init:
            self.W_dec.data = torch.rand(
                self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
            )
            self.initialize_decoder_norm_constant_norm()
        if self.cfg.init_encoder_as_decoder_transpose:
            self.W_enc.data = self.W_dec.data.T.clone().contiguous()
        if self.cfg.normalize_sae_decoder:
            with torch.no_grad():
                self.set_decoder_norm_to_unit_norm()

        # after finishing:
        self.b_enc = None

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        # For inference usage, just compute the feature activations
        features, _ = self.encode_with_hidden_pre(x)
        return features

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Decode feature activations to input space, with hooking + optional scaling.
        Matches GatedSAE.decode but includes any training logic if needed.
        """
        scaled_features = self.apply_finetuning_scaling_factor(feature_acts)
        sae_out_pre = scaled_features @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        """
        Gated forward pass with pre-activation (for training).
        We also inject noise if self.training is True.
        """
        sae_in = self.process_sae_in(x)

        # Gating path
        gating_pre_activation = sae_in @ self.W_enc + self.b_gate
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path
        magnitude_pre_activation = sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
        if self.training:
            magnitude_pre_activation += (
                torch.randn_like(magnitude_pre_activation) * self.cfg.noise_scale
            )
        magnitude_pre_activation = self.hook_sae_acts_pre(magnitude_pre_activation)

        feature_magnitudes = self.activation_fn(magnitude_pre_activation)

        # Combine gating path and magnitude path
        feature_acts = self.hook_sae_acts_post(active_features * feature_magnitudes)

        # Return both the final feature activations and the pre-activation (for logging or penalty)
        return feature_acts, magnitude_pre_activation

    def calculate_aux_loss(
        self,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: Optional[torch.Tensor],
        current_l1_coefficient: float,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Get the input tensor from kwargs
        sae_in = kwargs.get("sae_in")
        if sae_in is None:
            raise ValueError("sae_in is required for gated auxiliary loss calculation")

        # Re-center the input if apply_b_dec_to_input is set
        sae_in_centered = sae_in - (self.b_dec * self.cfg.apply_b_dec_to_input)

        # The gating pre-activation (pi_gate) for the auxiliary path
        pi_gate = sae_in_centered @ self.W_enc + self.b_gate
        pi_gate_act = torch.relu(pi_gate)

        # L1-like penalty scaled by W_dec norms
        l1_loss = (
            current_l1_coefficient
            * torch.sum(pi_gate_act * self.W_dec.norm(dim=1), dim=-1).mean()
        )

        # Aux reconstruction: reconstruct x purely from gating path
        via_gate_reconstruction = pi_gate_act @ self.W_dec + self.b_dec
        aux_recon_loss = (via_gate_reconstruction - sae_in).pow(2).sum(dim=-1).mean()

        # Return both losses separately
        return {"l1_loss": l1_loss, "auxiliary_reconstruction_loss": aux_recon_loss}

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: Optional[torch.Tensor] = None,
    ) -> TrainStepOutput:
        """
        Perform a forward pass for training that exactly matches original implementation.
        """
        feature_acts, hidden_pre = self.encode_with_hidden_pre(sae_in)
        sae_out = self.decode(feature_acts)

        # MSE Loss calculation
        per_item_mse_loss = self.mse_loss_fn(sae_out, sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        # Calculate auxiliary losses, passing sae_in as a keyword argument
        aux_losses = self.calculate_aux_loss(
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            dead_neuron_mask=dead_neuron_mask,
            current_l1_coefficient=current_l1_coefficient,
            sae_in=sae_in,
        )

        # Total loss and losses dictionary
        total_loss = (
            mse_loss
            + aux_losses["l1_loss"]
            + aux_losses["auxiliary_reconstruction_loss"]
        )

        # Structure losses as in original implementation
        losses = {
            "mse_loss": mse_loss,
            "l1_loss": aux_losses["l1_loss"],
            "auxiliary_reconstruction_loss": aux_losses[
                "auxiliary_reconstruction_loss"
            ],
        }

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Set decoder norms to unit norm"""
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def initialize_decoder_norm_constant_norm(self, norm: float = 0.1):
        """Initialize decoder with constant norm"""
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data *= norm
