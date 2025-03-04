import torch
from torch import nn
import numpy as np
from typing import Tuple, Optional, Any, Union
from jaxtyping import Float

from sae_lens.saes.sae_base import BaseSAE
from sae_lens.saes.sae_base import SAEConfig

from sae_lens.saes.sae_base import BaseTrainingSAE
from sae_lens.saes.sae_base import TrainingSAEConfig
from sae_lens.saes.sae_base import TrainStepOutput


class JumpReLUSAE(BaseSAE):
    """
    JumpReLUSAE is an inference-only implementation of a Sparse Autoencoder (SAE)
    using a JumpReLU activation. For each unit, if its pre-activation is
    <= threshold, that unit is zeroed out; otherwise, it follows a user-specified
    activation function (e.g., ReLU, tanh-relu, etc.).

    It implements:
      - initialize_weights: sets up parameters, including a threshold.
      - encode: computes the feature activations using JumpReLU.
      - decode: reconstructs the input from the feature activations.
      
    The BaseSAE.forward() method automatically calls encode and decode,
    including any error-term processing if configured.
    """

    def __init__(self, cfg: SAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    def initialize_weights(self) -> None:
        """
        Initialize encoder and decoder weights, as well as biases.
        Additionally, include a learnable `threshold` parameter that
        determines when units "turn on" for the JumpReLU.
        """
        # Biases
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

        # Threshold for JumpReLU
        # You can pick a default initialization (e.g., zeros means unit is off unless hidden_pre > 0)
        # or see the training version for more advanced init with log_threshold, etc.
        self.threshold = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        # Encoder and Decoder weights
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

    def encode(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_sae"]:
        """
        Encode the input tensor into the feature space using JumpReLU.
        The threshold parameter determines which units remain active.
        """
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)

        # 1) Apply the base "activation_fn" from config (e.g., ReLU, tanh-relu).
        base_acts = self.activation_fn(hidden_pre)

        # 2) Zero out any unit whose (hidden_pre <= threshold).
        #    We cast the boolean mask to the same dtype for safe multiplication.
        jump_relu_mask = (hidden_pre > self.threshold).to(base_acts.dtype)

        # 3) Multiply the normally activated units by that mask.
        feature_acts = self.hook_sae_acts_post(base_acts * jump_relu_mask)
        return feature_acts

    def decode(self, feature_acts: Float[torch.Tensor, "... d_sae"]) -> Float[torch.Tensor, "... d_in"]:
        """
        Decode the feature activations back to the input space.
        Follows the same steps as StandardSAE: apply scaling, transform, hook, and optionally reshape.
        """
        scaled_features = self.apply_finetuning_scaling_factor(feature_acts)
        sae_out_pre = scaled_features @ self.W_dec + self.b_dec
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


class JumpReLUTrainingSAE(BaseTrainingSAE):
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

    def __init__(self, cfg: TrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

        # We'll store a bandwidth for the training approach, if needed
        self.bandwidth = cfg.jumprelu_bandwidth

        # In typical JumpReLU training code, we may track a log_threshold:
        self.log_threshold = nn.Parameter(
            torch.ones(self.cfg.d_sae, dtype=self.dtype, device=self.device)
            * np.log(cfg.jumprelu_init_threshold)
        )

    def initialize_weights(self) -> None:
        """
        Initialize parameters like the base SAE, but also add log_threshold.
        """
        # Encoder Bias
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        # Decoder Bias
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )
        # W_enc
        w_enc_data = torch.nn.init.kaiming_uniform_(
            torch.empty(
                self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
            )
        )
        self.W_enc = nn.Parameter(w_enc_data)

        # W_dec
        w_dec_data = torch.nn.init.kaiming_uniform_(
            torch.empty(
                self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
            )
        )
        self.W_dec = nn.Parameter(w_dec_data)

        # Optionally apply orthogonal or heuristic init
        if self.cfg.decoder_orthogonal_init:
            self.W_dec.data = nn.init.orthogonal_(self.W_dec.data.T).T
        elif self.cfg.decoder_heuristic_init:
            self.W_dec.data = torch.rand(
                self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
            )
            self.initialize_decoder_norm_constant_norm()

        # Optionally transpose
        if self.cfg.init_encoder_as_decoder_transpose:
            self.W_enc.data = self.W_dec.data.T.clone().contiguous()

        # Optionally normalize columns of W_dec
        if self.cfg.normalize_sae_decoder:
            with torch.no_grad():
                self.set_decoder_norm_to_unit_norm()

    @property
    def threshold(self) -> torch.Tensor:
        """
        Returns the parameterized threshold > 0 for each unit.
        threshold = exp(log_threshold).
        """
        return torch.exp(self.log_threshold)

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Overridden version of the standard encoding that uses the JumpReLU approach
        for inference. For training, detailed logic is in encode_with_hidden_pre_jumprelu.
        """
        feature_acts, _ = self.encode_with_hidden_pre_jumprelu(x)
        return feature_acts

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Decode function is largely identical to StandardTrainingSAE: 
        apply finetuning scale, linear transform, recons hook, out normalization, 
        and reshape if needed for heads.
        """
        scaled_features = self.apply_finetuning_scaling_factor(feature_acts)
        sae_out_pre = scaled_features @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    def encode_with_hidden_pre_jumprelu(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        """
        Training-aware version of the JumpReLU SAE encoding. Adds noise if self.training
        is True. Then applies a JumpReLU step: base_acts = activation_fn(pre), 
        and zero out units with pre <= threshold.
        """
        sae_in = self.process_sae_in(x)

        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        if self.training and self.cfg.noise_scale > 0:
            hidden_pre = hidden_pre + torch.randn_like(hidden_pre) * self.cfg.noise_scale

        # Use the base activation configured (e.g. ReLU, tanh-relu, etc.)
        base_acts = self.activation_fn(hidden_pre)

        # Multiply by (pre > threshold)
        jump_relu_mask = (hidden_pre > self.threshold).to(base_acts.dtype)
        feature_acts = self.hook_sae_acts_post(base_acts * jump_relu_mask)

        return feature_acts, hidden_pre

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: Optional[torch.Tensor] = None,
    ):
        """
        Main training pass. 
        1) Encodes with JumpReLU, 
        2) Decodes, 
        3) Computes MSE, 
        4) Computes L0 loss from (pre > threshold), 
        5) Returns the combined loss.
        """
        feature_acts, hidden_pre = self.encode_with_hidden_pre_jumprelu(sae_in)
        sae_out = self.decode(feature_acts)

        # MSE Loss
        per_item_mse_loss = self.mse_loss_fn(sae_out, sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        # L0 Loss: Count how many times each unit is > threshold with Step.apply
        # from the old approach, or simply sum boolean mask
        step_mask = (hidden_pre > self.threshold).to(sae_out.dtype)
        # L0 penalty is basically sum of step_mask. 
        # Typically scaled by current_l1_coefficient
        l0_loss = (current_l1_coefficient * step_mask.sum(dim=-1)).mean()

        loss = mse_loss + l0_loss
        losses_dict = {
            "mse_loss": mse_loss,
            "l0_loss": l0_loss,
            "l1_loss": l0_loss,  # Add this for backward compatibility
        }

        # Return the train step output
        return self._create_train_step_output(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=loss,
            losses=losses_dict,
        ) 

    def encode_with_hidden_pre(
        self,
        x: Float[torch.Tensor, "... d_in"]
    ) -> Tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        # In master code, we have a dedicated method named encode_with_hidden_pre_jumprelu
        # so let's just call it here.
        return self.encode_with_hidden_pre_jumprelu(x)

    def calculate_aux_loss(
        self,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: Optional[torch.Tensor],
        current_l1_coefficient: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Calculate L0 penalty (number of units above threshold).
        """
        step_mask = (hidden_pre > self.threshold).to(feature_acts.dtype)
        l0_loss = (current_l1_coefficient * step_mask.sum(dim=-1)).mean()
        return l0_loss

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
        losses: dict[str, Union[float, torch.Tensor]],
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
    def set_decoder_norm_to_unit_norm(self):
        """Set decoder norms to unit norm"""
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

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
