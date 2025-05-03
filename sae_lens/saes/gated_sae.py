from typing import Any

import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import nn

from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
)


class GatedSAE(SAE):
    """
    GatedSAE is an inference-only implementation of a Sparse Autoencoder (SAE)
    using a gated linear encoder and a standard linear decoder.
    """

    b_gate: nn.Parameter
    b_mag: nn.Parameter
    r_mag: nn.Parameter

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
    def initialize_decoder_norm_constant_norm(self, norm: float = 0.1):
        """Initialize decoder with constant norm."""
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data *= norm


class GatedTrainingSAE(TrainingSAE):
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

    b_gate: nn.Parameter  # type: ignore
    b_mag: nn.Parameter  # type: ignore
    r_mag: nn.Parameter  # type: ignore

    def __init__(self, cfg: TrainingSAEConfig, use_error_term: bool = False):
        if use_error_term:
            raise ValueError(
                "GatedSAE does not support `use_error_term`. Please set `use_error_term=False`."
            )
        super().__init__(cfg, use_error_term)

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
        if self.training and self.cfg.noise_scale > 0:
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
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Re-center the input if apply_b_dec_to_input is set
        sae_in_centered = step_input.sae_in - (
            self.b_dec * self.cfg.apply_b_dec_to_input
        )

        # The gating pre-activation (pi_gate) for the auxiliary path
        pi_gate = sae_in_centered @ self.W_enc + self.b_gate
        pi_gate_act = torch.relu(pi_gate)

        # L1-like penalty scaled by W_dec norms
        l1_loss = (
            step_input.current_l1_coefficient
            * torch.sum(pi_gate_act * self.W_dec.norm(dim=1), dim=-1).mean()
        )

        # Aux reconstruction: reconstruct x purely from gating path
        via_gate_reconstruction = pi_gate_act @ self.W_dec + self.b_dec
        aux_recon_loss = (
            (via_gate_reconstruction - step_input.sae_in).pow(2).sum(dim=-1).mean()
        )

        # Return both losses separately
        return {"l1_loss": l1_loss, "auxiliary_reconstruction_loss": aux_recon_loss}

    def log_histograms(self) -> dict[str, NDArray[Any]]:
        """Log histograms of the weights and biases."""
        b_gate_dist = self.b_gate.detach().float().cpu().numpy()
        b_mag_dist = self.b_mag.detach().float().cpu().numpy()
        return {
            **super().log_histograms(),
            "weights/b_gate": b_gate_dist,
            "weights/b_mag": b_mag_dist,
        }

    @torch.no_grad()
    def initialize_decoder_norm_constant_norm(self, norm: float = 0.1):
        """Initialize decoder with constant norm"""
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data *= norm
