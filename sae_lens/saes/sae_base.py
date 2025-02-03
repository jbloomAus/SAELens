"""Base classes for Sparse Autoencoders (SAEs)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Callable, Optional, Tuple, TypeVar
import warnings

import einops
import torch
from jaxtyping import Float
from torch import nn
from transformer_lens.hook_points import HookedRootModule, HookPoint

from sae_lens.config import DTYPE_MAP

T = TypeVar("T", bound="BaseSAE")


@dataclass
class SAEConfig:
    """Base configuration for SAE models."""
    architecture: str
    d_in: int
    d_sae: int
    dtype: str
    device: str
    model_name: str
    hook_name: str
    hook_layer: int
    hook_head_index: Optional[int]
    activation_fn_str: str
    activation_fn_kwargs: dict[str, Any]
    apply_b_dec_to_input: bool
    finetuning_scaling_factor: bool
    normalize_activations: str
    context_size: Optional[int]
    dataset_path: Optional[str]
    dataset_trust_remote_code: bool
    sae_lens_training_version: str
    model_from_pretrained_kwargs: dict[str, Any]
    seqpos_slice: Optional[tuple[int, ...]]
    prepend_bos: bool

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAEConfig":
        valid_field_names = {field.name for field in fields(cls)}
        valid_config_dict = {
            key: val for key, val in config_dict.items() if key in valid_field_names
        }
        return cls(**valid_config_dict)

@dataclass
class TrainStepOutput:
    """Output from a training step."""
    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    hidden_pre: torch.Tensor
    loss: torch.Tensor  # we need to call backwards on this
    losses: dict[str, float | torch.Tensor]

class BaseSAE(HookedRootModule, ABC):
    """Abstract base class for all SAE architectures."""
    
    cfg: SAEConfig
    dtype: torch.dtype
    device: torch.device
    use_error_term: bool
    
    def __init__(self, cfg: SAEConfig, use_error_term: bool = False):
        """Initialize the SAE."""
        super().__init__()
        
        self.cfg = cfg
        
        if cfg.model_from_pretrained_kwargs:
            warnings.warn(
                "\nThis SAE has non-empty model_from_pretrained_kwargs. "
                "\nFor optimal performance, load the model like so:\n"
                "model = HookedSAETransformer.from_pretrained_no_processing(..., **cfg.model_from_pretrained_kwargs)",
                category=UserWarning,
                stacklevel=1,
            )

        self.dtype = DTYPE_MAP[cfg.dtype]
        self.device = torch.device(cfg.device)
        self.use_error_term = use_error_term
        
        # Set up activation function
        self.activation_fn = self._get_activation_fn()
        
        # Initialize weights
        self.initialize_weights()

        # Handle presence / absence of scaling factor
        if self.cfg.finetuning_scaling_factor:
            self.apply_finetuning_scaling_factor = (
                lambda x: x * self.finetuning_scaling_factor
            )
        else:
            self.apply_finetuning_scaling_factor = lambda x: x
        
        # Set up hooks
        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint()
        self.hook_sae_output = HookPoint()
        self.hook_sae_recons = HookPoint()
        self.hook_sae_error = HookPoint()
        
        # Set up activation normalization
        self._setup_activation_normalization()

        # Add reshape functions
        self.reshape_fn_in = lambda x: x
        self.reshape_fn_out = lambda x: x
        
        self.setup()  # Required for HookedRootModule
    
    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float):
        self.W_enc.data *= scaling_factor  # type: ignore
        self.W_dec.data /= scaling_factor  # type: ignore
        self.b_dec.data /= scaling_factor  # type: ignore
        self.cfg.normalize_activations = "none"
    
    def _get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get the activation function specified in config."""
        if self.cfg.activation_fn_str == "relu":
            return torch.nn.ReLU()
        elif self.cfg.activation_fn_str == "tanh-relu":
            def tanh_relu(input: torch.Tensor) -> torch.Tensor:
                input = torch.relu(input)
                return torch.tanh(input)
            return tanh_relu
        raise ValueError(f"Unknown activation function: {self.cfg.activation_fn_str}")
    
    def _setup_activation_normalization(self):
        """Set up activation normalization functions based on config."""
        if self.cfg.normalize_activations == "constant_norm_rescale":
            def run_time_activation_norm_fn_in(x: torch.Tensor) -> torch.Tensor:
                self.x_norm_coeff = (self.cfg.d_in**0.5) / x.norm(dim=-1, keepdim=True)
                return x * self.x_norm_coeff

            def run_time_activation_norm_fn_out(x: torch.Tensor) -> torch.Tensor:
                x = x / self.x_norm_coeff  # type: ignore
                del self.x_norm_coeff
                return x

            self.run_time_activation_norm_fn_in = run_time_activation_norm_fn_in
            self.run_time_activation_norm_fn_out = run_time_activation_norm_fn_out
            
        elif self.cfg.normalize_activations == "layer_norm":
            def run_time_activation_ln_in(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
                mu = x.mean(dim=-1, keepdim=True)
                x = x - mu
                std = x.std(dim=-1, keepdim=True)
                x = x / (std + eps)
                self.ln_mu = mu
                self.ln_std = std
                return x

            def run_time_activation_ln_out(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:  # noqa: ARG001
                return x * self.ln_std + self.ln_mu  # type: ignore

            self.run_time_activation_norm_fn_in = run_time_activation_ln_in
            self.run_time_activation_norm_fn_out = run_time_activation_ln_out
        else:
            self.run_time_activation_norm_fn_in = lambda x: x
            self.run_time_activation_norm_fn_out = lambda x: x
    
    @abstractmethod
    def initialize_weights(self):
        """Initialize model weights."""
        pass
    
    @abstractmethod
    def encode(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_sae"]:
        """Encode input tensor to feature space."""
        pass
    
    @abstractmethod
    def decode(self, feature_acts: Float[torch.Tensor, "... d_sae"]) -> Float[torch.Tensor, "... d_in"]:
        """Decode feature activations back to input space."""
        pass
    
    def turn_on_forward_pass_hook_z_reshaping(self):
        """For attention head outputs (hook_z) - reshape to 2D"""
        self.reshape_fn_in = lambda x: einops.rearrange(x, "b s n h -> (b s) (n h)")
        self.reshape_fn_out = lambda x: einops.rearrange(
            x, "(b s) (n h) -> b s n h", 
            n=self.cfg.hook_head_index is not None
        )

    def turn_off_forward_pass_hook_z_reshaping(self):
        self.reshape_fn_in = lambda x: x
        self.reshape_fn_out = lambda x: x
    
    def process_sae_in(self, sae_in: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_in"]:
        sae_in = sae_in.to(self.dtype)
        sae_in = self.reshape_fn_in(sae_in)  # Add reshape
        sae_in = self.hook_sae_input(sae_in)
        sae_in = self.run_time_activation_norm_fn_in(sae_in)
        return sae_in - (self.b_dec * self.cfg.apply_b_dec_to_input)  # type: ignore
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SAE."""
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)
        
        if self.use_error_term:
            with torch.no_grad():
                # Recompute without hooks for true error term
                feature_acts_clean = self.encode(x)
                x_reconstruct_clean = self.decode(feature_acts_clean)
                sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            sae_out = sae_out + sae_error
            
        return self.hook_sae_output(sae_out)
    
    @torch.no_grad()
    def fold_W_dec_norm(self):
        """Fold decoder norms into encoder."""
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T
        self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()


class BaseTrainingSAE(BaseSAE, ABC):
    """Abstract base class for training versions of SAEs."""
    
    cfg: TrainingSAEConfig
    
    def __init__(self, cfg: TrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        self.mse_loss_fn = self._get_mse_loss_fn()
    
    @abstractmethod
    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        """Encode with access to pre-activation values for training."""
        pass
    
    @abstractmethod
    def calculate_aux_loss(self, **kwargs) -> torch.Tensor:
        """Calculate architecture-specific auxiliary loss terms."""
        pass
    
    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: Optional[torch.Tensor] = None,
    ) -> TrainStepOutput:
        """Forward pass during training."""
        feature_acts, hidden_pre = self.encode_with_hidden_pre(sae_in)
        sae_out = self.decode(feature_acts)
        
        # Calculate MSE loss
        per_item_mse_loss = self.mse_loss_fn(sae_out, sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()
        
        # Calculate auxiliary losses
        aux_loss = self.calculate_aux_loss(
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            dead_neuron_mask=dead_neuron_mask,
            current_l1_coefficient=current_l1_coefficient,
        )
        
        losses = {
            "mse_loss": mse_loss,
            "aux_loss": aux_loss,
        }
        
        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=mse_loss + aux_loss,
            losses=losses,
        )
    
    def _get_mse_loss_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get the MSE loss function based on config."""
        def standard_mse_loss_fn(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.mse_loss(preds, target, reduction="none")

        def batch_norm_mse_loss_fn(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            target_centered = target - target.mean(dim=0, keepdim=True)
            normalization = target_centered.norm(dim=-1, keepdim=True)
            return torch.nn.functional.mse_loss(preds, target, reduction="none") / (normalization + 1e-6)

        if self.cfg.mse_loss_normalization == "dense_batch":
            return batch_norm_mse_loss_fn
        return standard_mse_loss_fn