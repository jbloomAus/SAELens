"""Clean API for working with architecture-specific SAEs."""

from typing import Any, Dict, Optional, Tuple

import torch

from sae_lens.loading.pretrained_sae_loaders import (
    NAMED_PRETRAINED_SAE_LOADERS,
    get_conversion_loader_name,
)
from sae_lens.loading.pretrained_saes_directory import (
    get_norm_scaling_factor,
    get_pretrained_saes_directory,
)
from sae_lens.saes.gated_sae import GatedSAE
from sae_lens.saes.jumprelu_sae import JumpReLUSAE
from sae_lens.saes.sae_base import BaseSAE, SAEConfig
from sae_lens.saes.standard_sae import StandardSAE
from sae_lens.saes.topk_sae import TopKSAE


def create_sae(cfg: SAEConfig, use_error_term: bool = False) -> BaseSAE:
    """
    Create an appropriate SAE instance based on the provided configuration.

    Args:
        cfg: The SAE configuration
        use_error_term: Whether to use an error term in the forward pass

    Returns:
        An instance of the appropriate SAE subclass
    """
    architecture = cfg.architecture.lower()

    if architecture == "standard":
        return StandardSAE(cfg, use_error_term)
    if architecture == "gated":
        return GatedSAE(cfg, use_error_term)
    if architecture == "jumprelu":
        return JumpReLUSAE(cfg, use_error_term)
    if architecture == "topk":
        return TopKSAE(cfg, use_error_term)
    raise ValueError(f"Unsupported architecture: {architecture}")


def load_pretrained_sae(
    release: str,
    sae_id: str,
    device: str = "cpu",
) -> Tuple[BaseSAE, Dict[str, Any], Optional[torch.Tensor]]:
    """
    Load a pretrained SAE with the appropriate architecture.

    Args:
        release: The release name (maps to a HuggingFace repo)
        sae_id: The ID of the SAE within the release
        device: Device to load the SAE on

    Returns:
        Tuple of (SAE instance, config dictionary, optional sparsity tensor)
    """
    # Get the appropriate loader
    sae_directory = get_pretrained_saes_directory()
    sae_info = sae_directory.get(release, None)
    config_overrides = sae_info.config_overrides if sae_info is not None else None

    conversion_loader_name = get_conversion_loader_name(sae_info)
    conversion_loader = NAMED_PRETRAINED_SAE_LOADERS[conversion_loader_name]

    # Load the SAE
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

    # Create the appropriate SAE instance
    sae_cfg = SAEConfig.from_dict(renamed_cfg_dict)
    sae = create_sae(sae_cfg)
    sae.load_state_dict(state_dict)

    # Apply activation normalization if needed
    if renamed_cfg_dict.get("normalize_activations") == "expected_average_only_in":
        norm_scaling_factor = get_norm_scaling_factor(release, sae_id)
        if norm_scaling_factor is not None:
            sae.fold_activation_norm_scaling_factor(norm_scaling_factor)
            renamed_cfg_dict["normalize_activations"] = "none"

    return sae, renamed_cfg_dict, log_sparsities
