import json
import re
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from packaging.version import Version
from safetensors import safe_open
from safetensors.torch import load_file

from sae_lens import logger
from sae_lens.constants import (
    DTYPE_MAP,
    SAE_CFG_FILENAME,
    SAE_WEIGHTS_FILENAME,
    SPARSIFY_WEIGHTS_FILENAME,
    SPARSITY_FILENAME,
)
from sae_lens.loading.pretrained_saes_directory import (
    get_config_overrides,
    get_pretrained_saes_directory,
    get_repo_id_and_folder_name,
)
from sae_lens.registry import get_sae_class
from sae_lens.util import filter_valid_dataclass_fields

LLM_METADATA_KEYS = {
    "model_name",
    "hook_name",
    "model_class_name",
    "hook_head_index",
    "model_from_pretrained_kwargs",
    "prepend_bos",
    "exclude_special_tokens",
    "neuronpedia_id",
    "context_size",
    "seqpos_slice",
    "dataset_path",
    "sae_lens_version",
    "sae_lens_training_version",
}


# loaders take in a release, sae_id, device, and whether to force download, and returns a tuple of config, state_dict, and log sparsity
class PretrainedSaeHuggingfaceLoader(Protocol):
    def __call__(
        self,
        repo_id: str,
        folder_name: str,
        device: str,
        force_download: bool,
        cfg_overrides: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]: ...


class PretrainedSaeConfigHuggingfaceLoader(Protocol):
    def __call__(
        self,
        repo_id: str,
        folder_name: str,
        device: str,
        force_download: bool,
        cfg_overrides: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


class PretrainedSaeDiskLoader(Protocol):
    def __call__(
        self,
        path: str | Path,
        device: str,
        cfg_overrides: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]: ...


class PretrainedSaeConfigDiskLoader(Protocol):
    def __call__(
        self,
        path: str | Path,
        device: str | None,
        cfg_overrides: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


def sae_lens_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """Loads SAEs from Hugging Face"""
    cfg_dict = get_sae_lens_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    revision = cfg_overrides.get("revision", None) if cfg_overrides else None
    weights_filename = f"{folder_name}/{SAE_WEIGHTS_FILENAME}"

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=weights_filename,
        force_download=force_download,
        revision=revision,
    )

    cfg_dict, state_dict = read_sae_components_from_disk(
        cfg_dict, file_path, device=device
    )

    # try to download log sparsity
    log_sparsity_filename = f"{folder_name}/{SPARSITY_FILENAME}"

    try:
        log_sparsity_path = hf_hub_download(
            repo_id=repo_id,
            filename=log_sparsity_filename,
            force_download=force_download,
            revision=revision,
        )
    except EntryNotFoundError:
        logger.info(
            f"Could not find {log_sparsity_filename} in {repo_id}. Skipping log sparsity loading."
        )
        return cfg_dict, state_dict, None

    log_sparsity = torch.load(log_sparsity_path, map_location=device)

    return cfg_dict, state_dict, log_sparsity


def sae_lens_disk_loader(
    path: str | Path,
    device: str = "cpu",
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Loads SAEs from disk"""

    config = get_sae_lens_config_from_disk(path, None, cfg_overrides)
    # TODO: handle loading log sparsity
    weights_path = Path(path).parent / SAE_WEIGHTS_FILENAME
    return read_sae_components_from_disk(config, weights_path, device=device)


def get_sae_lens_config_from_disk(
    path: str | Path,
    device: str | None = None,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = Path(path).parent / SAE_CFG_FILENAME
    cfg_dict = json.load(open(config_path))

    if device is not None:
        cfg_dict["device"] = device

    cfg_dict = {**cfg_dict, **(cfg_overrides or {})}

    return _post_process_cfg_dict(cfg_dict)


def get_sae_lens_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    revision = cfg_overrides.get("revision", None) if cfg_overrides else None
    config_filename = f"{folder_name}/{SAE_CFG_FILENAME}"

    config_path = hf_hub_download(
        repo_id, config_filename, force_download=force_download, revision=revision
    )
    with open(config_path) as config_file:
        cfg_dict = json.load(config_file)

    if device is not None:
        cfg_dict["device"] = device

    cfg_dict = {**cfg_dict, **(cfg_overrides or {})}

    return _post_process_cfg_dict(cfg_dict)


def _post_process_cfg_dict(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    sae_lens_version = cfg_dict.get("sae_lens_version", None)

    if not sae_lens_version or Version(sae_lens_version) < Version("6.0.0-rc.0"):
        cfg_dict = handle_pre_6_0_config(cfg_dict)
    return cfg_dict


def handle_config_defaulting(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    # Set default values for backwards compatibility
    cfg_dict.setdefault("prepend_bos", True)
    cfg_dict.setdefault("dataset_trust_remote_code", True)
    cfg_dict.setdefault("apply_b_dec_to_input", True)
    cfg_dict.setdefault("finetuning_scaling_factor", False)
    cfg_dict.setdefault("sae_lens_training_version", None)
    cfg_dict.setdefault("activation_fn_str", cfg_dict.get("activation_fn", "relu"))
    cfg_dict.setdefault("architecture", "standard")
    cfg_dict.setdefault("neuronpedia_id", None)

    if "normalize_activations" in cfg_dict and isinstance(
        cfg_dict["normalize_activations"], bool
    ):
        # backwards compatibility
        cfg_dict["normalize_activations"] = (
            "none"
            if not cfg_dict["normalize_activations"]
            else "expected_average_only_in"
        )

    cfg_dict.setdefault("normalize_activations", "none")
    cfg_dict.setdefault("device", "cpu")

    return cfg_dict


def handle_pre_6_0_config(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Format a config dictionary for a Sparse Autoencoder (SAE) to be compatible with the new 6.0 format.
    """

    rename_keys_map = {
        "hook_point": "hook_name",
        "hook_point_head_index": "hook_head_index",
        "activation_fn_str": "activation_fn",
    }
    new_cfg = {rename_keys_map.get(k, k): v for k, v in cfg_dict.items()}

    # Set default values for backwards compatibility
    new_cfg.setdefault("prepend_bos", True)
    new_cfg.setdefault("dataset_trust_remote_code", True)
    new_cfg.setdefault("apply_b_dec_to_input", True)
    new_cfg.setdefault("finetuning_scaling_factor", False)
    new_cfg.setdefault("sae_lens_training_version", None)
    new_cfg.setdefault("activation_fn", new_cfg.get("activation_fn", "relu"))
    new_cfg.setdefault("architecture", "standard")
    new_cfg.setdefault("neuronpedia_id", None)
    new_cfg.setdefault(
        "reshape_activations",
        "hook_z" if "hook_z" in new_cfg.get("hook_name", "") else "none",
    )

    if "normalize_activations" in new_cfg and isinstance(
        new_cfg["normalize_activations"], bool
    ):
        # backwards compatibility
        new_cfg["normalize_activations"] = (
            "none"
            if not new_cfg["normalize_activations"]
            else "expected_average_only_in"
        )

    if new_cfg.get("normalize_activations") is None:
        new_cfg["normalize_activations"] = "none"

    new_cfg.setdefault("device", "cpu")

    architecture = new_cfg.get("architecture", "standard")

    config_class = get_sae_class(architecture)[1]

    sae_cfg_dict = filter_valid_dataclass_fields(new_cfg, config_class)
    if architecture == "topk" and "activation_fn_kwargs" in new_cfg:
        sae_cfg_dict["k"] = new_cfg["activation_fn_kwargs"]["k"]

    sae_cfg_dict["metadata"] = {
        k: v for k, v in new_cfg.items() if k in LLM_METADATA_KEYS
    }
    sae_cfg_dict["architecture"] = architecture
    return sae_cfg_dict


def get_connor_rob_hook_z_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = folder_name.split(".pt")[0] + "_cfg.json"
    config_path = hf_hub_download(repo_id, config_path, force_download=force_download)

    with open(config_path) as config_file:
        old_cfg_dict = json.load(config_file)

    return {
        "architecture": "standard",
        "d_in": old_cfg_dict["act_size"],
        "d_sae": old_cfg_dict["dict_size"],
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "model_name": "gpt2-small",
        "hook_name": old_cfg_dict["act_name"],
        "hook_head_index": None,
        "activation_fn": "relu",
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "Skylion007/openwebtext",
        "context_size": 128,
        "normalize_activations": "none",
        "dataset_trust_remote_code": True,
        "reshape_activations": "hook_z",
        **(cfg_overrides or {}),
    }


def connor_rob_hook_z_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], None]:
    cfg_dict = get_connor_rob_hook_z_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    file_path = hf_hub_download(
        repo_id=repo_id, filename=folder_name, force_download=force_download
    )
    weights = torch.load(file_path, map_location=device)

    return cfg_dict, weights, None


def read_sae_components_from_disk(
    cfg_dict: dict[str, Any],
    weight_path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """
    Given a loaded dictionary and a path to a weight file, load the weights and return the state_dict.
    """
    if dtype is None:
        dtype = DTYPE_MAP[cfg_dict["dtype"]]

    state_dict = {}
    with safe_open(weight_path, framework="pt", device=device) as f:  # type: ignore
        for k in f.keys():  # noqa: SIM118
            state_dict[k] = f.get_tensor(k).to(dtype=dtype)

    # if bool and True, then it's the April update method of normalizing activations and hasn't been folded in.
    if "scaling_factor" in state_dict:
        # we were adding it anyway for a period of time but are no longer doing so.
        # so we should delete it if
        if torch.allclose(
            state_dict["scaling_factor"],
            torch.ones_like(state_dict["scaling_factor"]),
        ):
            del state_dict["scaling_factor"]
            cfg_dict["finetuning_scaling_factor"] = False
        else:
            if not cfg_dict["finetuning_scaling_factor"]:
                logger.warning(
                    "We are removing the scaling factor from the state dict, but the config says not to use it. Unusual."
                )
                assert cfg_dict["sae_lens_training_version"] != "4.0.0"
            else:
                del state_dict["scaling_factor"]

    if cfg_dict["normalize_activations"] == "expected_average_only_in":
        # we were adding it anyway for a period of time but are no longer doing so.
        assert "scaling_factor" not in state_dict

    return cfg_dict, state_dict


def get_gemma_2_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    width_map = {
        "width_16k": 16384,
        "width_65k": 65536,
        "width_262k": 262144,
        "width_524k": 524288,
        "width_1m": 1048576,
    }

    # Check if folder_name is in width_map
    if folder_name in width_map:
        d_sae = width_map[folder_name]
    else:
        # Try to extract width from more complex folder names like "gemma-scope-2b-pt-res-canonical/layer_19/width_16k/average_l0_123"
        match = re.search(r"width_(\d+)k", folder_name)
        if match:
            width_value = int(match.group(1))
            d_sae = width_value * 1024
        else:
            raise ValueError(f"Could not extract dictionary size from folder name: {folder_name}")

    # Extract layer if present in folder_name
    layer_match = re.search(r"layer_(\d+)", folder_name)
    if layer_match:
        layer = int(layer_match.group(1))
    else:
        # Check repo_id for layer information
        layer_match = re.search(r"layer_(\d+)", repo_id)
        if layer_match:
            layer = int(layer_match.group(1))
        else:
            raise ValueError("Could not extract layer index from folder_name or repo_id")

    # Determine model and d_model from repo_id
    if "2b-it" in repo_id:
        model_name = "gemma-2-2b-it"
        d_model = 2304
    elif "9b-it" in repo_id:
        model_name = "gemma-2-9b-it"
        d_model = 3584
    else:
        # Add more model configurations as needed
        raise ValueError(f"Could not determine model from repo_id: {repo_id}")

    cfg_dict = {
        "architecture": "jumprelu",
        "d_in": d_model,
        "d_sae": d_sae,
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "model_name": model_name,
        "hook_name": f"blocks.{layer}.hook_resid_post",
        "hook_head_index": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "activation_fn": "relu",
        "normalize_activations": "none",
        **(cfg_overrides or {}),
    }

    return cfg_dict


def gemma_2_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    cfg_dict = get_gemma_2_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the npz file
    revision = cfg_overrides.get("revision", None) if cfg_overrides else None
    params_filename = f"{folder_name}/params.npz"

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=params_filename,
        force_download=force_download,
        revision=revision,
    )

    # Load weights from npz file
    params = np.load(file_path)
    
    # Convert to state dict with proper naming
    state_dict = {}
    for key in params.files:
        tensor = torch.tensor(params[key], dtype=torch.float32, device=device)
        if key.lower() == "w_enc":
            state_dict["W_enc"] = tensor
        elif key.lower() == "w_dec":
            state_dict["W_dec"] = tensor
        elif key.lower() == "b_enc":
            state_dict["b_enc"] = tensor
        elif key.lower() == "b_dec":
            state_dict["b_dec"] = tensor
        elif key.lower() == "threshold":
            state_dict["threshold"] = tensor

    return cfg_dict, state_dict, None


def get_llama_scope_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Config information is typically in a JSON file within the folder
    revision = cfg_overrides.get("revision", None) if cfg_overrides else None
    config_filename = f"{folder_name}/config.json"

    config_path = hf_hub_download(
        repo_id, config_filename, force_download=force_download, revision=revision
    )
    with open(config_path) as config_file:
        raw_config = json.load(config_file)

    # Process the config to match SAELens format
    layer = raw_config.get("layer", 0)
    hook_name = f"blocks.{layer}.attn.hook_attn_out"

    cfg_dict = {
        "architecture": "standard",
        "d_in": raw_config.get("num_inputs", 4096),
        "d_sae": raw_config.get("num_latents", 32768),
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "model_name": "meta-llama/Llama-3.1-8B-Base",
        "hook_name": hook_name,
        "hook_head_index": None,
        "prepend_bos": True,
        "dataset_path": "HuggingFaceFW/fineweb",
        "context_size": 1024,
        "activation_fn": "relu",
        "normalize_activations": "none",
        **(cfg_overrides or {}),
    }

    return cfg_dict


def llama_scope_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    cfg_dict = get_llama_scope_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Load the safetensors weights
    revision = cfg_overrides.get("revision", None) if cfg_overrides else None
    weights_filename = f"{folder_name}/state_dict.pt"

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=weights_filename,
        force_download=force_download,
        revision=revision,
    )

    # Load weights
    state_dict = torch.load(file_path, map_location=device)

    # Convert to SAELens naming convention
    renamed_state_dict = {}
    for key, value in state_dict.items():
        if "encoder.weight" in key:
            renamed_state_dict["W_enc"] = value.T.to(torch.float32)
        elif "decoder.weight" in key:
            renamed_state_dict["W_dec"] = value.T.to(torch.float32)
        elif "encoder.bias" in key:
            renamed_state_dict["b_enc"] = value.to(torch.float32)
        elif "decoder.bias" in key:
            renamed_state_dict["b_dec"] = value.to(torch.float32)

    return cfg_dict, renamed_state_dict, None


def get_llama_scope_r1_distill_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return get_llama_scope_config_from_hf(
        repo_id, folder_name, device, force_download, cfg_overrides
    )


def llama_scope_r1_distill_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    return llama_scope_sae_huggingface_loader(
        repo_id, folder_name, device, force_download, cfg_overrides
    )


def get_dictionary_learning_config_1_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Config information is not provided, so we infer from the repo structure
    # These SAEs are typically trained on GPT-2 small

    cfg_dict = {
        "architecture": "standard",
        "d_in": 768,  # GPT-2 small hidden size
        "d_sae": 32768,  # Typical expansion factor
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "model_name": "openai-community/gpt2",
        "hook_name": "blocks.0.hook_mlp_out",  # Adjust based on actual layer
        "hook_head_index": None,
        "activation_fn": "relu",
        "prepend_bos": True,
        "dataset_path": "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        "context_size": 256,
        "normalize_activations": "none",
        **(cfg_overrides or {}),
    }

    return cfg_dict


def dictionary_learning_sae_huggingface_loader_1(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    cfg_dict = get_dictionary_learning_config_1_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the state dict file
    revision = cfg_overrides.get("revision", None) if cfg_overrides else None
    weights_filename = f"{folder_name}/ae.pt"

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=weights_filename,
        force_download=force_download,
        revision=revision,
    )

    # Load weights
    state_dict_loaded = torch.load(file_path, map_location=device)

    # Extract and rename tensors
    state_dict = {}
    state_dict["W_enc"] = state_dict_loaded["encoder.weight"].T.to(torch.float32)
    state_dict["W_dec"] = state_dict_loaded["decoder.weight"].T.to(torch.float32)
    state_dict["b_enc"] = state_dict_loaded["encoder.bias"].to(torch.float32)
    state_dict["b_dec"] = state_dict_loaded["decoder.bias"].to(torch.float32)

    return cfg_dict, state_dict, None


def get_deepseek_r1_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # DeepSeek R1 configuration
    layer_match = re.search(r"layer_(\d+)", folder_name)
    if layer_match:
        layer = int(layer_match.group(1))
    else:
        layer = 0  # Default to layer 0

    cfg_dict = {
        "architecture": "topk",
        "d_in": 2048,  # DeepSeek R1 hidden size
        "d_sae": 53248,  # Typical expansion factor for DeepSeek R1
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "hook_name": f"blocks.{layer}.hook_resid_post",
        "hook_head_index": None,
        "k": 128,  # TopK value
        "activation_fn": "topk",
        "prepend_bos": True,
        "dataset_path": "",  # Update with actual dataset
        "context_size": 1024,
        "normalize_activations": "none",
        **(cfg_overrides or {}),
    }

    return cfg_dict


def deepseek_r1_sae_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    cfg_dict = get_deepseek_r1_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the safetensors weights
    revision = cfg_overrides.get("revision", None) if cfg_overrides else None
    weights_filename = f"{folder_name}/sae_weights.safetensors"

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=weights_filename,
        force_download=force_download,
        revision=revision,
    )

    # Load weights
    state_dict = load_file(file_path, device=device)

    # Convert to float32
    for key in state_dict:
        state_dict[key] = state_dict[key].to(torch.float32)

    return cfg_dict, state_dict, None


def get_sparsify_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get config for sparsify SAEs"""
    # Config for sparsify SAEs
    layer_match = re.search(r"layer-(\d+)", folder_name)
    if layer_match:
        layer = int(layer_match.group(1))
    else:
        layer = 0  # Default to layer 0

    # Determine model from repo_id
    if "gemma-2b" in repo_id.lower():
        model_name = "google/gemma-2b"
        d_model = 2048
    elif "llama-3.2-1b" in repo_id.lower():
        model_name = "meta-llama/Llama-3.2-1B"
        d_model = 2048
    else:
        # Default configuration
        model_name = "google/gemma-2b"
        d_model = 2048

    cfg_dict = {
        "architecture": "standard",
        "d_in": d_model,
        "d_sae": 16384,  # Typical expansion factor
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "model_name": model_name,
        "hook_name": f"blocks.{layer}.hook_resid_post",
        "hook_head_index": None,
        "activation_fn": "relu",
        "prepend_bos": True,
        "dataset_path": "HuggingFaceFW/fineweb",
        "context_size": 1024,
        "normalize_activations": "none",
        **(cfg_overrides or {}),
    }

    return cfg_dict


def sparsify_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """Load sparsify SAEs from HuggingFace"""
    cfg_dict = get_sparsify_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the weights file
    revision = cfg_overrides.get("revision", None) if cfg_overrides else None
    weights_filename = f"{folder_name}/{SPARSIFY_WEIGHTS_FILENAME}"

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=weights_filename,
        force_download=force_download,
        revision=revision,
    )

    # Load weights
    state_dict_loaded = torch.load(file_path, map_location=device)

    # Handle different formats that might be used
    dtype = DTYPE_MAP[cfg_dict["dtype"]]
    W_enc = (
        state_dict_loaded["W_enc"]
        if "W_enc" in state_dict_loaded
        else state_dict_loaded["encoder.weight"].T
    ).to(dtype)

    if "W_dec" in state_dict_loaded:
        W_dec = state_dict_loaded["W_dec"].T.to(dtype)
    else:
        W_dec = state_dict_loaded["decoder.weight"].T.to(dtype)

    if "b_enc" in state_dict_loaded:
        b_enc = state_dict_loaded["b_enc"].to(dtype)
    elif "encoder.bias" in state_dict_loaded:
        b_enc = state_dict_loaded["encoder.bias"].to(dtype)
    else:
        b_enc = torch.zeros(cfg_dict["d_sae"], dtype=dtype, device=device)

    if "b_dec" in state_dict_loaded:
        b_dec = state_dict_loaded["b_dec"].to(dtype)
    elif "decoder.bias" in state_dict_loaded:
        b_dec = state_dict_loaded["decoder.bias"].to(dtype)
    else:
        b_dec = torch.zeros(cfg_dict["d_in"], dtype=dtype, device=device)

    state_dict = {"W_enc": W_enc, "b_enc": b_enc, "W_dec": W_dec, "b_dec": b_dec}
    return cfg_dict, state_dict


# Transcoder loaders
def get_gemma_2_transcoder_config_from_hf(
    repo_id: str,
    folder_name: str,
    device: str | None = None,
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get config for Gemma-2 transcoders"""
    width_map = {
        "width_4k": 4096,
        "width_16k": 16384,
        "width_65k": 65536,
        "width_262k": 262144,
        "width_524k": 524288,
        "width_1m": 1048576,
    }

    # Extract width from folder name
    d_sae = None
    for width_key, width_value in width_map.items():
        if width_key in folder_name:
            d_sae = width_value
            break
    
    if d_sae is None:
        # Try to extract from pattern like "width_16k"
        match = re.search(r"width_(\d+)k", folder_name)
        if match:
            d_sae = int(match.group(1)) * 1024
        else:
            raise ValueError(f"Could not extract dictionary size from folder name: {folder_name}")

    # Extract layer
    layer_match = re.search(r"layer_(\d+)", folder_name)
    if layer_match:
        layer = int(layer_match.group(1))
    else:
        layer_match = re.search(r"layer_(\d+)", repo_id)
        if layer_match:
            layer = int(layer_match.group(1))
        else:
            raise ValueError("Could not extract layer index")

    # Determine model and dimensions from repo_id
    model_configs = {
        "2b-it": ("gemma-2-2b-it", 2304),
        "2b": ("gemma-2-2b", 2304),
        "9b-it": ("gemma-2-9b-it", 3584),
        "9b": ("gemma-2-9b", 3584),
        "27b-it": ("gemma-2-27b-it", 4608),
        "27b": ("gemma-2-27b", 4608),
    }
    
    model_name = None
    d_model = None
    for model_key, (name, dim) in model_configs.items():
        if model_key in repo_id:
            model_name = name
            d_model = dim
            break
    
    if model_name is None:
        raise ValueError(f"Could not determine model from repo_id: {repo_id}")

    cfg_dict = {
        "architecture": "jumprelu_transcoder",  # Use JumpReLU transcoder architecture
        "d_in": d_model,
        "d_out": d_model,  # Transcoders map to same dimension by default
        "d_sae": d_sae,
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "model_name": model_name,
        "hook_name": f"blocks.{layer}.ln2.hook_normalized",  # Input hook (after pre-MLP RMSNorm)
        "hook_name_out": f"blocks.{layer}.hook_mlp_out",  # Output hook (after MLP)
        "hook_layer_out": layer,
        "hook_head_index": None,
        "hook_head_index_out": None,
        "activation_fn": "jumprelu",
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "normalize_activations": "none",
        **(cfg_overrides or {}),
    }

    return cfg_dict


def gemma_2_transcoder_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """Load Gemma-2 transcoders from HuggingFace"""
    cfg_dict = get_gemma_2_transcoder_config_from_hf(
        repo_id,
        folder_name,
        device,
        force_download,
        cfg_overrides,
    )

    # Download the npz file
    revision = cfg_overrides.get("revision", None) if cfg_overrides else None
    params_filename = f"{folder_name}/params.npz"

    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=params_filename,
        force_download=force_download,
        revision=revision,
    )

    # Load weights from npz file
    params = np.load(file_path)
    
    # Convert to state dict with proper naming
    state_dict = {}
    for key in params.files:
        tensor = torch.tensor(params[key], dtype=torch.float32, device=device)
        # Handle various naming conventions
        key_lower = key.lower()
        if key_lower in ["w_enc", "wenc", "w_e"]:
            state_dict["W_enc"] = tensor
        elif key_lower in ["w_dec", "wdec", "w_d"]:
            state_dict["W_dec"] = tensor
        elif key_lower in ["b_enc", "benc", "b_e"]:
            state_dict["b_enc"] = tensor
        elif key_lower in ["b_dec", "bdec", "b_d"]:
            state_dict["b_dec"] = tensor
        elif key_lower == "threshold":
            state_dict["threshold"] = tensor

    return cfg_dict, state_dict, None


def llama_relu_skip_transcoder_huggingface_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor | None]:
    """Load Llama ReLU skip transcoders from HuggingFace"""
    # Download the safetensors file
    revision = cfg_overrides.get("revision", None) if cfg_overrides else None
    
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=folder_name,  # folder_name is the actual filename for this loader
        force_download=force_download,
        revision=revision,
    )

    # Load weights
    state_dict_loaded = load_file(file_path, device=device)
    
    # Extract dimensions from loaded weights
    d_sae, d_in = state_dict_loaded["W_enc"].shape
    
    # Extract layer from filename
    layer_match = re.search(r"layer_(\d+)", folder_name)
    if layer_match:
        layer = int(layer_match.group(1))
    else:
        raise ValueError(f"Could not extract layer index from filename: {folder_name}")

    # Build config
    cfg_dict = {
        "architecture": "skip_transcoder",
        "d_in": d_in,
        "d_out": d_in,  # For skip transcoder, output dimension matches input
        "d_sae": d_sae,
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "model_name": "meta-llama/Llama-3.2-1B",
        "hook_name": f"blocks.{layer}.hook_resid_mid",  # Input hook
        "hook_name_out": f"blocks.{layer}.hook_mlp_out",  # Output hook
        "hook_layer_out": layer,
        "hook_head_index": None,
        "hook_head_index_out": None,
        "activation_fn": "relu",
        "prepend_bos": True,
        "dataset_path": "",  # Update with actual dataset
        "context_size": 1024,
        "normalize_activations": "none",
        **(cfg_overrides or {}),
    }
    
    # Convert weights to expected format
    state_dict = {}
    state_dict["W_enc"] = state_dict_loaded["W_enc"].T.to(torch.float32)
    state_dict["W_dec"] = state_dict_loaded["W_dec"].T.to(torch.float32)
    state_dict["W_skip"] = state_dict_loaded["W_skip"].to(torch.float32)  # No transpose for skip
    state_dict["b_enc"] = state_dict_loaded["b_enc"].to(torch.float32)
    state_dict["b_dec"] = state_dict_loaded["b_dec"].to(torch.float32)

    return cfg_dict, state_dict, None


NAMED_PRETRAINED_SAE_LOADERS: dict[str, PretrainedSaeHuggingfaceLoader] = {
    "sae_lens": sae_lens_huggingface_loader,
    "connor_rob_hook_z": connor_rob_hook_z_huggingface_loader,
    "gemma_2": gemma_2_sae_huggingface_loader,
    "llama_scope": llama_scope_sae_huggingface_loader,
    "llama_scope_r1_distill": llama_scope_r1_distill_sae_huggingface_loader,
    "dictionary_learning_1": dictionary_learning_sae_huggingface_loader_1,
    "deepseek_r1": deepseek_r1_sae_huggingface_loader,
    "sparsify": sparsify_huggingface_loader,
    "gemma_2_transcoder": gemma_2_transcoder_huggingface_loader,
    "llama_relu_skip_transcoder": llama_relu_skip_transcoder_huggingface_loader,
}


NAMED_PRETRAINED_SAE_CONFIG_GETTERS: dict[str, PretrainedSaeConfigHuggingfaceLoader] = {
    "sae_lens": get_sae_lens_config_from_hf,
    "connor_rob_hook_z": get_connor_rob_hook_z_config_from_hf,
    "gemma_2": get_gemma_2_config_from_hf,
    "llama_scope": get_llama_scope_config_from_hf,
    "llama_scope_r1_distill": get_llama_scope_r1_distill_config_from_hf,
    "dictionary_learning_1": get_dictionary_learning_config_1_from_hf,
    "deepseek_r1": get_deepseek_r1_config_from_hf,
    "sparsify": get_sparsify_config_from_hf,
    "gemma_2_transcoder": get_gemma_2_transcoder_config_from_hf,
}


def get_conversion_loader_name(release: str) -> str:
    saes_directory = get_pretrained_saes_directory()
    sae_info = saes_directory.get(release, None)
    conversion_loader_name = "sae_lens"
    if sae_info is not None and sae_info.conversion_func is not None:
        conversion_loader_name = sae_info.conversion_func
    if conversion_loader_name not in NAMED_PRETRAINED_SAE_LOADERS:
        raise ValueError(
            f"Conversion func '{conversion_loader_name}' not found in NAMED_PRETRAINED_SAE_LOADERS."
        )
    return conversion_loader_name