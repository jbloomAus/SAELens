from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from jaxtyping import Float
from torch import nn
from typing_extensions import override

from sae_lens.saes.sae import (
    SAE,
    SAEConfig,
    TrainCoefficientConfig,
    TrainingSAE,
    TrainingSAEConfig,
    TrainStepInput,
    TrainStepOutput,
)
from sae_lens.util import filter_valid_dataclass_fields


def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class Step(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x > threshold).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        threshold_grad = torch.sum(
            -(1.0 / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )
        return None, threshold_grad, None


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output  # We don't apply STE to x input
        threshold_grad = torch.sum(
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


@dataclass
class JumpReLUSAEConfig(SAEConfig):
    """
    Configuration class for a JumpReLUSAE.
    """

    @override
    @classmethod
    def architecture(cls) -> str:
        return "jumprelu"


class JumpReLUSAE(SAE[JumpReLUSAEConfig]):
    """
    JumpReLUSAE is an inference-only implementation of a Sparse Autoencoder (SAE)
    using a JumpReLU activation. For each unit, if its pre-activation is
    <= threshold, that unit is zeroed out; otherwise, it follows a user-specified
    activation function (e.g., ReLU etc.).

    It implements:
      - initialize_weights: sets up parameters, including a threshold.
      - encode: computes the feature activations using JumpReLU.
      - decode: reconstructs the input from the feature activations.

    The BaseSAE.forward() method automatically calls encode and decode,
    including any error-term processing if configured.
    """

    b_enc: nn.Parameter
    threshold: nn.Parameter

    def __init__(self, cfg: JumpReLUSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    @override
    def initialize_weights(self) -> None:
        super().initialize_weights()
        self.threshold = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Encode the input tensor into the feature space using JumpReLU.
        The threshold parameter determines which units remain active.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        # 1) Apply the base "activation_fn" from config (e.g., ReLU).
        base_acts = self.activation_fn(hidden_pre)

        # 2) Zero out any unit whose (hidden_pre <= threshold).
        #    We cast the boolean mask to the same dtype for safe multiplication.
        jump_relu_mask = (hidden_pre > self.threshold).to(base_acts.dtype)

        # 3) Multiply the normally activated units by that mask.
        return self.hook_sae_acts_post(base_acts * jump_relu_mask)

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Decode the feature activations back to the input space.
        Follows the same steps as StandardSAE: apply scaling, transform, hook, and optionally reshape.
        """
        sae_out_pre = feature_acts @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @torch.no_grad()
    def fold_W_dec_norm(self):
        """
        Override to properly handle threshold adjustment with W_dec norms.
        When we scale the encoder weights, we need to scale the threshold
        by the same factor to maintain the same sparsity pattern.
        """
        # Save the current threshold before calling parent method
        current_thresh = self.threshold.clone()

        # Get W_dec norms that will be used for scaling
        W_dec_norms = self.W_dec.norm(dim=-1)

        # Call parent implementation to handle W_enc, W_dec, and b_enc adjustment
        super().fold_W_dec_norm()

        # Scale the threshold by the same factor as we scaled b_enc
        # This ensures the same features remain active/inactive after folding
        self.threshold.data = current_thresh * W_dec_norms


@dataclass
class JumpReLUTrainingSAEConfig(TrainingSAEConfig):
    """
    Configuration class for training a JumpReLUTrainingSAE.
    """

    jumprelu_init_threshold: float = 0.01
    jumprelu_bandwidth: float = 0.05
    l0_coefficient: float = 1.0
    l0_warm_up_steps: int = 0

    @override
    @classmethod
    def architecture(cls) -> str:
        return "jumprelu"


class JumpReLUTrainingSAE(TrainingSAE[JumpReLUTrainingSAEConfig]):
    """
    JumpReLUTrainingSAE is a training-focused implementation of a SAE using a JumpReLU activation.

    Similar to the inference-only JumpReLUSAE, but with:
      - A learnable log-threshold parameter (instead of a raw threshold).
      - Forward passes that add noise during training, if configured.
      - A specialized auxiliary loss term for sparsity (L0 or similar).

    Methods of interest include:
    - initialize_weights: sets up W_enc, b_enc, W_dec, b_dec, and log_threshold.
    - encode_with_hidden_pre_jumprelu: runs a forward pass for training, optionally adding noise.
    - training_forward_pass: calculates MSE and auxiliary losses, returning a TrainStepOutput.
    """

    b_enc: nn.Parameter
    log_threshold: nn.Parameter

    def __init__(self, cfg: JumpReLUTrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

        # We'll store a bandwidth for the training approach, if needed
        self.bandwidth = cfg.jumprelu_bandwidth

        # In typical JumpReLU training code, we may track a log_threshold:
        self.log_threshold = nn.Parameter(
            torch.ones(self.cfg.d_sae, dtype=self.dtype, device=self.device)
            * np.log(cfg.jumprelu_init_threshold)
        )

    @override
    def initialize_weights(self) -> None:
        """
        Initialize parameters like the base SAE, but also add log_threshold.
        """
        super().initialize_weights()
        # Encoder Bias
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

    @property
    def threshold(self) -> torch.Tensor:
        """
        Returns the parameterized threshold > 0 for each unit.
        threshold = exp(log_threshold).
        """
        return torch.exp(self.log_threshold)

    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        sae_in = self.process_sae_in(x)

        hidden_pre = sae_in @ self.W_enc + self.b_enc
        feature_acts = JumpReLU.apply(hidden_pre, self.threshold, self.bandwidth)

        return feature_acts, hidden_pre  # type: ignore

    @override
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Calculate architecture-specific auxiliary loss terms."""
        l0 = torch.sum(Step.apply(hidden_pre, self.threshold, self.bandwidth), dim=-1)  # type: ignore
        l0_loss = (step_input.coefficients["l0"] * l0).mean()
        return {"l0_loss": l0_loss}

    @override
    def get_coefficients(self) -> dict[str, float | TrainCoefficientConfig]:
        return {
            "l0": TrainCoefficientConfig(
                value=self.cfg.l0_coefficient,
                warm_up_steps=self.cfg.l0_warm_up_steps,
            ),
        }

    @torch.no_grad()
    def fold_W_dec_norm(self):
        """
        Override to properly handle threshold adjustment with W_dec norms.
        """
        # Save the current threshold before we call the parent method
        current_thresh = self.threshold.clone()

        # Get W_dec norms
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)

        # Call parent implementation to handle W_enc and W_dec adjustment
        super().fold_W_dec_norm()

        # Fix: Use squeeze() instead of squeeze(-1) to match old behavior
        self.log_threshold.data = torch.log(current_thresh * W_dec_norms.squeeze())

    def _create_train_step_output(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        loss: torch.Tensor,
        losses: dict[str, torch.Tensor],
    ) -> TrainStepOutput:
        """
        Helper to produce a TrainStepOutput from the trainer.
        The old code expects a method named _create_train_step_output().
        """
        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=loss,
            losses=losses,
        )

    @torch.no_grad()
    def initialize_decoder_norm_constant_norm(self, norm: float = 0.1):
        """Initialize decoder with constant norm"""
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data *= norm

    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        """Convert log_threshold to threshold for saving"""
        if "log_threshold" in state_dict:
            threshold = torch.exp(state_dict["log_threshold"]).detach().contiguous()
            del state_dict["log_threshold"]
            state_dict["threshold"] = threshold

    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        """Convert threshold to log_threshold for loading"""
        if "threshold" in state_dict:
            threshold = state_dict["threshold"]
            del state_dict["threshold"]
            state_dict["log_threshold"] = torch.log(threshold).detach().contiguous()

    def to_inference_config_dict(self) -> dict[str, Any]:
        return filter_valid_dataclass_fields(
            self.cfg.to_dict(), JumpReLUSAEConfig, ["architecture"]
        )
