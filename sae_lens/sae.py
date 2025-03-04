"""Facade/factory module for SAE loading and compatibility with old code."""

import json
import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Tuple, Union, Iterator, Dict

import torch
from torch import nn
from safetensors.torch import save_file
from transformer_lens.hook_points import HookedRootModule

from sae_lens.loading.pretrained_sae_loaders import (
    NAMED_PRETRAINED_SAE_LOADERS,
    get_conversion_loader_name,
    handle_config_defaulting,
    read_sae_from_disk,
)
from sae_lens.loading.pretrained_saes_directory import (
    get_norm_scaling_factor,
    get_pretrained_saes_directory,
)
from sae_lens.saes.gated_sae import GatedSAE
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.saes.topk_sae import TopKSAE
from sae_lens.saes.sae_base import BaseSAE, SAEConfig
from sae_lens.saes.standard_sae import StandardSAE

SPARSITY_FILENAME = "sparsity.safetensors"
SAE_WEIGHTS_FILENAME = "sae_weights.safetensors"
SAE_CFG_FILENAME = "cfg.json"


# Factory function to create the appropriate SAE class
def create_sae_from_config(cfg: SAEConfig, use_error_term: bool = False) -> BaseSAE:
    """
    Factory function to create the appropriate SAE instance based on architecture.
    
    Args:
        cfg: SAE configuration
        use_error_term: Whether to use the error term in the forward pass
        
    Returns:
        An instance of the appropriate SAE class
    """
    architecture = cfg.architecture.lower()
    
    if architecture == "standard":
        return StandardSAE(cfg, use_error_term)
    elif architecture == "gated":
        return GatedSAE(cfg, use_error_term)
    elif architecture == "jumprelu":
        return JumpReLUSAE(cfg, use_error_term)
    elif architecture == "topk":
        return TopKSAE(cfg, use_error_term)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


class SAE(HookedRootModule):
    """
    Factory/Facade class for Sparse Autoencoders (SAEs).
    Maintains backward compatibility with existing code while
    using the new architecture-specific implementations internally.
    """
    
    # Internal SAE implementation
    _sae: BaseSAE
    
    @property
    def cfg(self) -> SAEConfig:
        return self._sae.cfg
    
    @property
    def dtype(self) -> torch.dtype:
        return self._sae.dtype
        
    @property
    def device(self) -> torch.device:
        return self._sae.device
        
    @property
    def use_error_term(self) -> bool:
        return self._sae.use_error_term
        
    @use_error_term.setter
    def use_error_term(self, value: bool) -> None:
        self._sae.use_error_term = value
    
    def __init__(self, cfg: SAEConfig, use_error_term: bool = False):
        super().__init__()
        # Create the appropriate implementation based on architecture
        self._sae = create_sae_from_config(cfg, use_error_term)
        
        # Forward hooks from the internal implementation
        # self.hook_sae_input = self._sae.hook_sae_input
        # self.hook_sae_acts_pre = self._sae.hook_sae_acts_pre
        # self.hook_sae_acts_post = self._sae.hook_sae_acts_post
        # self.hook_sae_output = self._sae.hook_sae_output
        # self.hook_sae_recons = self._sae.hook_sae_recons
        # self.hook_sae_error = self._sae.hook_sae_error
        
        # Create property handles for parameters 
        # This ensures tensor methods work properly
        self._param_names = []
        for name, param in self._sae.named_parameters():
            self._param_names.append(name)
            # Use property to dynamically access the parameter from _sae
            setattr(self.__class__, name, property(
                lambda self, name=name: getattr(self._sae, name),
                lambda self, value, name=name: setattr(self._sae, name, value)
            ))
            
        self.setup()  # Required for HookedRootModule

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying SAE implementation."""
        if name.startswith('_'):
            return super().__getattr__(name)
        return getattr(self._sae, name)
    
    # Basic delegation methods
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._sae.forward(x)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._sae.encode(x)
    
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return self._sae.decode(feature_acts)
    
    def to(self, *args: Any, **kwargs: Any) -> "SAE":
        # First update the internal SAE
        self._sae.to(*args, **kwargs)
        
        # Now update the facade
        result = super().to(*args, **kwargs)
        
        # Extract device and dtype from args and kwargs
        device_arg = None
        dtype_arg = None
        
        # Check args
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device_arg = arg
            elif isinstance(arg, torch.dtype):
                dtype_arg = arg
            elif isinstance(arg, torch.Tensor):
                device_arg = arg.device
                dtype_arg = arg.dtype
        
        # Check kwargs
        device_arg = kwargs.get("device", device_arg)
        dtype_arg = kwargs.get("dtype", dtype_arg)
        
        # Update device in config if provided
        if device_arg is not None:
            # Convert device to torch.device if it's a string
            device = torch.device(device_arg) if isinstance(device_arg, str) else device_arg
            
            # Update the cfg.device
            self.cfg.device = str(device)
            
            # Update the device property
            self._sae.device = device
        
        # Update dtype in config if provided
        if dtype_arg is not None:
            # Update the cfg.dtype
            self.cfg.dtype = str(dtype_arg)
            
            # Update the dtype property
            self._sae.dtype = dtype_arg
        
        return result
    
    def fold_activation_norm_scaling_factor(self, activation_norm_scaling_factor: float):
        self._sae.fold_activation_norm_scaling_factor(activation_norm_scaling_factor)
    
    def fold_W_dec_norm(self):
        self._sae.fold_W_dec_norm()
    
    # Method that needs to be maintained at the facade level
    def save_model(self, path: Union[str, Path], sparsity: Optional[torch.Tensor] = None):
        """Save model weights, config, and optional sparsity tensor to disk."""
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)

        # Generate the weights
        state_dict = self._sae.state_dict()  # Use internal SAE state dict
        self.process_state_dict_for_saving(state_dict)
        model_weights_path = path / SAE_WEIGHTS_FILENAME
        save_file(state_dict, model_weights_path)

        # Save the config
        config = self.cfg.to_dict()
        cfg_path = path / SAE_CFG_FILENAME
        with open(cfg_path, "w") as f:
            json.dump(config, f)

        if sparsity is not None:
            sparsity_in_dict = {"sparsity": sparsity}
            sparsity_path = path / SPARSITY_FILENAME
            save_file(sparsity_in_dict, sparsity_path)
            return model_weights_path, cfg_path, sparsity_path

        return model_weights_path, cfg_path
    
    # Forward state dict processing to the internal implementation
    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        if hasattr(self._sae, 'process_state_dict_for_saving'):
            method = getattr(self._sae, 'process_state_dict_for_saving')
            if callable(method):
                method(state_dict)
    
    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        if hasattr(self._sae, 'process_state_dict_for_loading'):
            method = getattr(self._sae, 'process_state_dict_for_loading')
            if callable(method):
                method(state_dict)

    # Delegate hook_z reshaping methods to the internal implementation
    def turn_on_forward_pass_hook_z_reshaping(self):
        self._sae.turn_on_forward_pass_hook_z_reshaping()
    
    def turn_off_forward_pass_hook_z_reshaping(self):
        self._sae.turn_off_forward_pass_hook_z_reshaping()
    
    def get_name(self):
        """Generate a name for this SAE."""
        return f"sae_{self.cfg.model_name}_{self.cfg.hook_name}_{self.cfg.d_sae}"

    # Class methods for loading models
    @classmethod
    def load_from_pretrained(
        cls, path: Union[str, Path], device: str = "cpu", dtype: Optional[str] = None
    ) -> "SAE":
        """Load a pretrained SAE from disk."""
        config_path = os.path.join(path, SAE_CFG_FILENAME)
        with open(config_path) as f:
            cfg_dict = json.load(f)
        cfg_dict = handle_config_defaulting(cfg_dict)
        cfg_dict["device"] = device
        if dtype is not None:
            cfg_dict["dtype"] = dtype

        weight_path = os.path.join(path, SAE_WEIGHTS_FILENAME)
        cfg_dict, state_dict = read_sae_from_disk(
            cfg_dict=cfg_dict,
            weight_path=weight_path,
            device=device,
        )

        sae_cfg = SAEConfig.from_dict(cfg_dict)
        sae = cls(sae_cfg)
        sae.process_state_dict_for_loading(state_dict)
        sae._sae.load_state_dict(state_dict)
        return sae

    @classmethod
    def from_pretrained(
        cls,
        release: str,
        sae_id: str,
        device: str = "cpu",
    ) -> Tuple["SAE", dict[str, Any], Optional[torch.Tensor]]:
        """Load a pretrained SAE from the Hugging Face model hub."""
        # Get sae directory
        sae_directory = get_pretrained_saes_directory()

        # Validate release and sae_id
        if release not in sae_directory:
            if "/" not in release:
                raise ValueError(
                    f"Release {release} not found in pretrained SAEs directory, and is not a valid huggingface repo."
                )
        elif sae_id not in sae_directory[release].saes_map:
            # Handle special cases like Gemma Scope
            if (
                "gemma-scope" in release
                and "canonical" not in release
                and f"{release}-canonical" in sae_directory
            ):
                canonical_ids = list(
                    sae_directory[release + "-canonical"].saes_map.keys()
                )
                # Shorten the lengthy string of valid IDs
                if len(canonical_ids) > 5:
                    str_canonical_ids = str(canonical_ids[:5])[:-1] + ", ...]"
                else:
                    str_canonical_ids = str(canonical_ids)
                value_suffix = f" If you don't want to specify an L0 value, consider using release {release}-canonical which has valid IDs {str_canonical_ids}"
            else:
                value_suffix = ""

            valid_ids = list(sae_directory[release].saes_map.keys())
            # Shorten the lengthy string of valid IDs
            if len(valid_ids) > 5:
                str_valid_ids = str(valid_ids[:5])[:-1] + ", ...]"
            else:
                str_valid_ids = str(valid_ids)

            raise ValueError(
                f"ID {sae_id} not found in release {release}. Valid IDs are {str_valid_ids}."
                + value_suffix
            )
            
        # Get loader configuration
        sae_info = sae_directory.get(release, None)
        config_overrides = sae_info.config_overrides if sae_info is not None else None

        conversion_loader_name = get_conversion_loader_name(sae_info)
        conversion_loader = NAMED_PRETRAINED_SAE_LOADERS[conversion_loader_name]

        # Load config and weights
        cfg_dict, state_dict, log_sparsities = conversion_loader(
            release,
            sae_id=sae_id,
            device=device,
            force_download=False,
            cfg_overrides=config_overrides,
        )

        # Rename keys to match SAEConfig field names
        renamed_cfg_dict = {}
        rename_map = {
            "hook_point": "hook_name",
            "hook_point_layer": "hook_layer",
            "hook_point_head_index": "hook_head_index",
            "activation_fn": "activation_fn",
        }
        
        for k, v in cfg_dict.items():
            renamed_cfg_dict[rename_map.get(k, k)] = v
        
        # Set default values for required fields
        renamed_cfg_dict.setdefault("activation_fn_kwargs", {})
        renamed_cfg_dict.setdefault("seqpos_slice", None)
        
        # Create SAE with appropriate architecture
        sae_cfg = SAEConfig.from_dict(renamed_cfg_dict)
        sae = cls(sae_cfg)
        sae.process_state_dict_for_loading(state_dict)
        sae._sae.load_state_dict(state_dict)

        # Apply normalization if needed
        if renamed_cfg_dict.get("normalize_activations") == "expected_average_only_in":
            norm_scaling_factor = get_norm_scaling_factor(release, sae_id)
            if norm_scaling_factor is not None:
                sae.fold_activation_norm_scaling_factor(norm_scaling_factor)
                renamed_cfg_dict["normalize_activations"] = "none"
            else:
                warnings.warn(
                    f"norm_scaling_factor not found for {release} and {sae_id}, but normalize_activations is 'expected_average_only_in'. Skipping normalization folding."
                )

        return sae, renamed_cfg_dict, log_sparsities

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAE":
        """Create an SAE from a config dictionary."""
        return cls(SAEConfig.from_dict(config_dict))
    

_blank_hook = nn.Identity()


@contextmanager
def _disable_hooks(sae: SAE):
    """
    Temporarily disable hooks for the SAE. Swaps out all the hooks with a fake modules that does nothing.
    """
    try:
        for hook_name in sae._sae.hook_dict:
            setattr(sae._sae, hook_name, _blank_hook)
        yield
    finally:
        for hook_name, hook in sae._sae.hook_dict.items():
            setattr(sae._sae, hook_name, hook)
