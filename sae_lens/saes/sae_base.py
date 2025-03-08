"""Base classes for Sparse Autoencoders (SAEs)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Optional, Tuple, TypeVar
import warnings

import einops
import torch
from jaxtyping import Float
from torch import nn
from transformer_lens.hook_points import HookedRootModule, HookPoint
from contextlib import contextmanager
from sae_lens.config import DTYPE_MAP
from sae_lens.config import LanguageModelSAERunnerConfig

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
    activation_fn: str
    activation_fn_kwargs: dict[str, Any]
    apply_b_dec_to_input: bool
    finetuning_scaling_factor: bool
    normalize_activations: str
    context_size: Optional[int]
    dataset_path: Optional[str]
    dataset_trust_remote_code: bool
    sae_lens_training_version: str
    model_from_pretrained_kwargs: dict[str, Any] = field(default_factory=dict)
    seqpos_slice: Optional[tuple[int, ...]] = None
    prepend_bos: bool = False
    neuronpedia_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAEConfig":
        valid_field_names = {field.name for field in fields(cls)}
        valid_config_dict = {
            key: val for key, val in config_dict.items() if key in valid_field_names
        }
        
        # Ensure seqpos_slice is a tuple
        if "seqpos_slice" in valid_config_dict and valid_config_dict["seqpos_slice"] is not None:
            if isinstance(valid_config_dict["seqpos_slice"], list):
                valid_config_dict["seqpos_slice"] = tuple(valid_config_dict["seqpos_slice"])
        
        return cls(**valid_config_dict)

@dataclass
class TrainStepOutput:
    """Output from a training step."""
    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    hidden_pre: torch.Tensor
    loss: torch.Tensor  # we need to call backwards on this
    losses: dict[str, torch.Tensor]

class BaseSAE(HookedRootModule, ABC):
    """Abstract base class for all SAE architectures."""
    
    cfg: SAEConfig
    dtype: torch.dtype
    device: torch.device
    use_error_term: bool
    
    # For type checking only - don't provide default values
    # These will be initialized by subclasses
    W_enc: nn.Parameter
    W_dec: nn.Parameter
    b_dec: nn.Parameter
    
    # Architecture-specific parameters that may be present depending on subclass
    # Don't provide default values here
    # b_enc: Optional[nn.Parameter]
    b_gate: Optional[nn.Parameter]
    b_mag: Optional[nn.Parameter]
    r_mag: Optional[nn.Parameter]
    
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

        # handle hook_z reshaping if needed.
        if self.cfg.hook_name.endswith("_z"):
            # print(f"Setting up hook_z reshaping for {self.cfg.hook_name}")
            self.turn_on_forward_pass_hook_z_reshaping()
        else:
            # print(f"No hook_z reshaping needed for {self.cfg.hook_name}")
            self.turn_off_forward_pass_hook_z_reshaping()
        
        # Set up activation normalization
        self._setup_activation_normalization()
        
        self.setup()  # Required for HookedRootModule
    
    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float):
        self.W_enc.data *= scaling_factor  # type: ignore
        self.W_dec.data /= scaling_factor  # type: ignore
        self.b_dec.data /= scaling_factor  # type: ignore
        self.cfg.normalize_activations = "none"
    
    def _get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get the activation function specified in config."""
        return self._get_activation_fn_static(self.cfg.activation_fn, **(self.cfg.activation_fn_kwargs or {}))

    @staticmethod
    def _get_activation_fn_static(activation_fn: str, **kwargs) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get the activation function from a string specification."""
        if activation_fn == "relu":
            return torch.nn.ReLU()
        elif activation_fn == "tanh-relu":
            def tanh_relu(input: torch.Tensor) -> torch.Tensor:
                input = torch.relu(input)
                return torch.tanh(input)
            return tanh_relu
        elif activation_fn == "topk":
            if "k" not in kwargs:
                raise ValueError("TopK activation function requires a k value.")
            k = kwargs.get("k", 1)  # Default k to 1 if not provided
            
            def topk_fn(x: torch.Tensor) -> torch.Tensor:
                topk = torch.topk(x.flatten(start_dim=-1), k=k, dim=-1)
                values = torch.relu(topk.values)
                result = torch.zeros_like(x.flatten(start_dim=-1))
                result.scatter_(-1, topk.indices, values)
                return result.view_as(x)
            
            return topk_fn
        raise ValueError(f"Unknown activation function: {activation_fn}")
    
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
        if not self.cfg.hook_name.endswith("_z"):
            raise ValueError("This method should only be called for hook_z SAEs.")

        # print(f"Turning on hook_z reshaping for {self.cfg.hook_name}")
        
        def reshape_fn_in(x: torch.Tensor):
            # print(f"reshape_fn_in input shape: {x.shape}")
            self.d_head = x.shape[-1]
            # print(f"Setting d_head to: {self.d_head}")
            self.reshape_fn_in = lambda x: einops.rearrange(
                x, "... n_heads d_head -> ... (n_heads d_head)"
            )
            reshaped = einops.rearrange(x, "... n_heads d_head -> ... (n_heads d_head)")
            # print(f"reshape_fn_in output shape: {reshaped.shape}")
            return reshaped

        self.reshape_fn_in = reshape_fn_in
        self.reshape_fn_out = lambda x, d_head: einops.rearrange(
            x, "... (n_heads d_head) -> ... n_heads d_head", d_head=d_head
        )
        self.hook_z_reshaping_mode = True
        # print(f"hook_z reshaping turned on, self.d_head={getattr(self, 'd_head', None)}")

    def turn_off_forward_pass_hook_z_reshaping(self):
        self.reshape_fn_in = lambda x: x
        self.reshape_fn_out = lambda x, d_head: x  # noqa: ARG005
        self.d_head = None
        self.hook_z_reshaping_mode = False
    
    def process_sae_in(self, sae_in: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_in"]:
        # print(f"Input shape to process_sae_in: {sae_in.shape}")
        # print(f"self.cfg.hook_name: {self.cfg.hook_name}")
        # print(f"self.b_dec shape: {self.b_dec.shape}")
        # print(f"Hook z reshaping mode: {getattr(self, 'hook_z_reshaping_mode', False)}")
        
        sae_in = sae_in.to(self.dtype)
        
        # print(f"Shape before reshape_fn_in: {sae_in.shape}")
        sae_in = self.reshape_fn_in(sae_in)
        # print(f"Shape after reshape_fn_in: {sae_in.shape}")
        
        sae_in = self.hook_sae_input(sae_in)
        sae_in = self.run_time_activation_norm_fn_in(sae_in)
        
        # Here's where the error happens
        bias_term = self.b_dec * self.cfg.apply_b_dec_to_input
        # print(f"Bias term shape: {bias_term.shape}")
        
        return sae_in - bias_term
    
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
    
    @torch.no_grad()
    def fold_W_dec_norm(self):
        """Fold decoder norms into encoder."""
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T
        
        # Only update b_enc if it exists (standard/jumprelu architectures)
        if hasattr(self, 'b_enc') and isinstance(self.b_enc, nn.Parameter):
            self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()


@dataclass(kw_only=True)
class TrainingSAEConfig(SAEConfig):
    # Sparsity Loss Calculations
    l1_coefficient: float
    lp_norm: float
    use_ghost_grads: bool
    normalize_sae_decoder: bool
    noise_scale: float
    decoder_orthogonal_init: bool
    mse_loss_normalization: Optional[str]
    jumprelu_init_threshold: float
    jumprelu_bandwidth: float
    decoder_heuristic_init: bool
    init_encoder_as_decoder_transpose: bool
    scale_sparsity_penalty_by_decoder_norm: bool

    @classmethod
    def from_sae_runner_config(
        cls, cfg: LanguageModelSAERunnerConfig
    ) -> "TrainingSAEConfig":
        return cls(
            # base config
            architecture=cfg.architecture,
            d_in=cfg.d_in,
            d_sae=cfg.d_sae,  # type: ignore
            dtype=cfg.dtype,
            device=cfg.device,
            model_name=cfg.model_name,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            hook_head_index=cfg.hook_head_index,
            activation_fn=cfg.activation_fn,
            activation_fn_kwargs=cfg.activation_fn_kwargs,
            apply_b_dec_to_input=cfg.apply_b_dec_to_input,
            finetuning_scaling_factor=cfg.finetuning_method is not None,
            sae_lens_training_version=cfg.sae_lens_training_version,
            context_size=cfg.context_size,
            dataset_path=cfg.dataset_path,
            prepend_bos=cfg.prepend_bos,
            seqpos_slice=tuple(x for x in cfg.seqpos_slice if x is not None) if cfg.seqpos_slice is not None else None,
            # Training cfg
            l1_coefficient=cfg.l1_coefficient,
            lp_norm=cfg.lp_norm,
            use_ghost_grads=cfg.use_ghost_grads,
            normalize_sae_decoder=cfg.normalize_sae_decoder,
            noise_scale=cfg.noise_scale,
            decoder_orthogonal_init=cfg.decoder_orthogonal_init,
            mse_loss_normalization=cfg.mse_loss_normalization,
            decoder_heuristic_init=cfg.decoder_heuristic_init,
            init_encoder_as_decoder_transpose=cfg.init_encoder_as_decoder_transpose,
            scale_sparsity_penalty_by_decoder_norm=cfg.scale_sparsity_penalty_by_decoder_norm,
            normalize_activations=cfg.normalize_activations,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
            model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs or {},
            jumprelu_init_threshold=cfg.jumprelu_init_threshold,
            jumprelu_bandwidth=cfg.jumprelu_bandwidth,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAEConfig":
        # remove any keys that are not in the dataclass
        # since we sometimes enhance the config with the whole LM runner config
        valid_field_names = {field.name for field in fields(cls)}
        valid_config_dict = {
            key: val for key, val in config_dict.items() if key in valid_field_names
        }

        # ensure seqpos slice is tuple
        # ensure that seqpos slices is a tuple
        # Ensure seqpos_slice is a tuple
        if "seqpos_slice" in valid_config_dict:
            if isinstance(valid_config_dict["seqpos_slice"], list):
                valid_config_dict["seqpos_slice"] = tuple(
                    valid_config_dict["seqpos_slice"]
                )
            elif not isinstance(valid_config_dict["seqpos_slice"], tuple):
                valid_config_dict["seqpos_slice"] = (valid_config_dict["seqpos_slice"],)

        return TrainingSAEConfig(**valid_config_dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "l1_coefficient": self.l1_coefficient,
            "lp_norm": self.lp_norm,
            "use_ghost_grads": self.use_ghost_grads,
            "normalize_sae_decoder": self.normalize_sae_decoder,
            "noise_scale": self.noise_scale,
            "decoder_orthogonal_init": self.decoder_orthogonal_init,
            "init_encoder_as_decoder_transpose": self.init_encoder_as_decoder_transpose,
            "mse_loss_normalization": self.mse_loss_normalization,
            "decoder_heuristic_init": self.decoder_heuristic_init,
            "scale_sparsity_penalty_by_decoder_norm": self.scale_sparsity_penalty_by_decoder_norm,
            "normalize_activations": self.normalize_activations,
            "jumprelu_init_threshold": self.jumprelu_init_threshold,
            "jumprelu_bandwidth": self.jumprelu_bandwidth,
        }

    # this needs to exist so we can initialize the parent sae cfg without the training specific
    # parameters. Maybe there's a cleaner way to do this
    def get_base_sae_cfg_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "activation_fn": self.activation_fn,
            "activation_fn_kwargs": self.activation_fn_kwargs,
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "dtype": self.dtype,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "hook_head_index": self.hook_head_index,
            "device": self.device,
            "context_size": self.context_size,
            "prepend_bos": self.prepend_bos,
            "finetuning_scaling_factor": self.finetuning_scaling_factor,
            "normalize_activations": self.normalize_activations,
            "dataset_path": self.dataset_path,
            "dataset_trust_remote_code": self.dataset_trust_remote_code,
            "sae_lens_training_version": self.sae_lens_training_version,
            "model_from_pretrained_kwargs": self.model_from_pretrained_kwargs,
            "seqpos_slice": self.seqpos_slice,
            "neuronpedia_id": self.neuronpedia_id,
        }


class BaseTrainingSAE(BaseSAE, ABC):
    """Abstract base class for training versions of SAEs."""
    
    cfg: "TrainingSAEConfig"  # Quote marks make this a forward reference
    
    def __init__(self, cfg: TrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)
        
        # Turn off hook_z reshaping for training mode - the activation store
        # is expected to handle reshaping before passing data to the SAE
        self.turn_off_forward_pass_hook_z_reshaping()
        
        self.mse_loss_fn = self._get_mse_loss_fn()
    
    @abstractmethod
    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        """Encode with access to pre-activation values for training."""
        pass
    
    @abstractmethod
    def calculate_aux_loss(
        self, 
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: Optional[torch.Tensor] = None,
        current_l1_coefficient: Optional[float] = None,
        sae_in: Optional[torch.Tensor] = None,
        sae_out: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
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
        
        # Calculate architecture-specific auxiliary losses
        aux_losses = self.calculate_aux_loss(
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            dead_neuron_mask=dead_neuron_mask,
            current_l1_coefficient=current_l1_coefficient,
            sae_in=sae_in,
            sae_out=sae_out,
        )
        
        # Total loss is MSE plus all auxiliary losses
        total_loss = mse_loss
        
        # Create losses dictionary with mse_loss
        losses = {"mse_loss": mse_loss}
        
        # Add architecture-specific losses to the dictionary
        losses.update(aux_losses)
        
        # Sum all losses for total_loss
        for loss_name, loss_value in aux_losses.items():
            total_loss = total_loss + loss_value
        
        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
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


_blank_hook = nn.Identity()


@contextmanager
def _disable_hooks(sae: BaseSAE):
    """
    Temporarily disable hooks for the SAE. Swaps out all the hooks with a fake modules that does nothing.
    """
    try:
        for hook_name in sae.hook_dict:
            setattr(sae, hook_name, _blank_hook)
        yield
    finally:
        for hook_name, hook in sae.hook_dict.items():
            setattr(sae, hook_name, hook)