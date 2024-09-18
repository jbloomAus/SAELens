import json
import re
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from safetensors import safe_open


# loaders take in a repo_id, folder_name, device, and whether to force download, and returns a tuple of config and state_dict
class PretrainedSaeLoader(Protocol):

    def __call__(
        self,
        repo_id: str,
        folder_name: str,
        device: str | torch.device | None = None,
        force_download: bool = False,
        cfg_overrides: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], Optional[torch.Tensor]]: ...


def sae_lens_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Get's SAEs from HF, loads them.
    """
    # Get the config
    cfg_dict = get_sae_config_from_hf(
        repo_id,
        folder_name,
        force_download,
    )
    # Apply overrides if provided
    if cfg_overrides is not None:
        cfg_dict.update(cfg_overrides)
    cfg_dict["device"] = device
    cfg_dict = handle_config_defaulting(cfg_dict)

    weights_filename = f"{folder_name}/sae_weights.safetensors"
    sae_path = hf_hub_download(
        repo_id=repo_id, filename=weights_filename, force_download=force_download
    )

    # TODO: Make this cleaner. I hate try except statements.
    try:
        sparsity_filename = f"{folder_name}/sparsity.safetensors"
        log_sparsity_path = hf_hub_download(
            repo_id=repo_id, filename=sparsity_filename, force_download=force_download
        )
    except EntryNotFoundError:
        log_sparsity_path = None  # no sparsity file

    cfg_dict, state_dict = read_sae_from_disk(
        cfg_dict=cfg_dict,
        weight_path=sae_path,
        device=device,
    )

    # get sparsity tensor if it exists
    if log_sparsity_path is not None:
        with safe_open(log_sparsity_path, framework="pt", device=device) as f:  # type: ignore
            log_sparsity = f.get_tensor("sparsity")
    else:
        log_sparsity = None

    return cfg_dict, state_dict, log_sparsity


def get_sae_config_from_hf(
    repo_id: str,
    folder_name: str,
    force_download: bool = False,
) -> Dict[str, Any]:
    """
    Retrieve the configuration for a Sparse Autoencoder (SAE) from a Hugging Face repository.

    Args:
        repo_id (str): The repository ID on Hugging Face.
        folder_name (str): The folder name within the repository containing the config file.
        force_download (bool, optional): Whether to force download the config file. Defaults to False.
        cfg_overrides (Optional[Dict[str, Any]], optional): Overrides for the config. Defaults to None.

    Returns:
        Dict[str, Any]: The configuration dictionary for the SAE.
    """
    cfg_filename = f"{folder_name}/cfg.json"
    cfg_path = hf_hub_download(
        repo_id=repo_id, filename=cfg_filename, force_download=force_download
    )

    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)

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
    cfg_dict.setdefault("neuronpedia", None)

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


def connor_rob_hook_z_loader(
    repo_id: str,
    folder_name: str,
    device: Optional[str] = None,
    force_download: bool = False,
    cfg_overrides: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], None]:

    file_path = hf_hub_download(
        repo_id=repo_id, filename=folder_name, force_download=force_download
    )
    config_path = folder_name.split(".pt")[0] + "_cfg.json"
    config_path = hf_hub_download(repo_id, config_path)
    old_cfg_dict = json.load(open(config_path, "r"))

    weights = torch.load(file_path, map_location=device)
    # weights_filename = f"{folder_name}/sae_weights.safetensors"
    # sae_path = hf_hub_download(
    #     repo_id=repo_id, filename=weights_filename, force_download=force_download
    # )
    # if device is None:
    #     device = "cuda" if torch.cuda.is_available() else "cpu"

    # return load_pretrained_sae_lens_sae_components(cfg_path, sae_path, device)

    # old_cfg_dict = {
    #     "seed": 49,
    #     "batch_size": 4096,
    #     "buffer_mult": 384,
    #     "lr": 0.0012,
    #     "num_tokens": 2000000000,
    #     "l1_coeff": 1.8,
    #     "beta1": 0.9,
    #     "beta2": 0.99,
    #     "dict_mult": 32,
    #     "seq_len": 128,
    #     "enc_dtype": "fp32",
    #     "model_name": "gpt2-small",
    #     "site": "z",
    #     "layer": 0,
    #     "device": "cuda",
    #     "reinit": "reinit",
    #     "head": "cat",
    #     "concat_heads": True,
    #     "resample_scheme": "anthropic",
    #     "anthropic_neuron_resample_scale": 0.2,
    #     "dead_direction_cutoff": 1e-06,
    #     "re_init_every": 25000,
    #     "anthropic_resample_last": 12500,
    #     "resample_factor": 0.01,
    #     "num_resamples": 4,
    #     "wandb_project_name": "gpt2-L0-20240117",
    #     "wandb_entity": "ckkissane",
    #     "save_state_dict_every": 50000,
    #     "b_dec_init": "zeros",
    #     "sched_type": "cosine_warmup",
    #     "sched_epochs": 1000,
    #     "sched_lr_factor": 0.1,
    #     "sched_warmup_epochs": 1000,
    #     "sched_finish": True,
    #     "anthropic_resample_batches": 100,
    #     "eval_every": 1000,
    #     "model_batch_size": 512,
    #     "buffer_size": 1572864,
    #     "buffer_batches": 12288,
    #     "act_name": "blocks.0.attn.hook_z",
    #     "act_size": 768,
    #     "dict_size": 24576,
    #     "name": "gpt2-small_0_24576_z",
    # }

    cfg_dict = {
        "architecture": "standard",
        "d_in": old_cfg_dict["act_size"],
        "d_sae": old_cfg_dict["dict_size"],
        "dtype": "float32",
        "device": device if device is not None else "cpu",
        "model_name": "gpt2-small",
        "hook_name": old_cfg_dict["act_name"],
        "hook_layer": old_cfg_dict["layer"],
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "Skylion007/openwebtext",
        "context_size": 128,
        "normalize_activations": "none",
        "dataset_trust_remote_code": True,
    }

    return cfg_dict, weights, None


def read_sae_from_disk(
    cfg_dict: dict[str, Any],
    weight_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """
    Given a loaded dictionary and a path to a weight file, load the weights and return the state_dict.
    """

    state_dict = {}
    with safe_open(weight_path, framework="pt", device=device) as f:  # type: ignore
        for k in f.keys():
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
            assert cfg_dict[
                "finetuning_scaling_factor"
            ], "Scaling factor is present but finetuning_scaling_factor is False."
            state_dict["finetuning_scaling_factor"] = state_dict["scaling_factor"]
            del state_dict["scaling_factor"]
    else:
        # it's there and it's not all 1's, we should use it.
        cfg_dict["finetuning_scaling_factor"] = False

    return cfg_dict, state_dict


def get_gemma_2_config(
    repo_id: str,
    folder_name: str,
    d_sae_override: Optional[int] = None,
    layer_override: Optional[int] = None,
) -> Dict[str, Any]:
    # Detect width from folder_name
    width_map = {
        "width_16k": 16384,
        "width_32k": 32768,
        "width_65k": 65536,
        "width_131k": 131072,
        "width_262k": 262144,
        "width_524k": 524288,
        "width_1m": 1048576,
    }
    d_sae = next(
        (width for key, width in width_map.items() if key in folder_name), None
    )
    if d_sae is None:
        if not d_sae_override:
            raise ValueError("Width not found in folder_name and no override provided.")
        d_sae = d_sae_override

    # Detect layer from folder_name
    match = re.search(r"layer_(\d+)", folder_name)
    layer = int(match.group(1)) if match else layer_override
    if layer is None:
        raise ValueError("Layer not found in folder_name and no override provided.")

    # Model specific parameters
    model_params = {
        "2b-it": {"name": "gemma-2-2b-it", "d_in": 2304},
        "9b-it": {"name": "gemma-2-9b-it", "d_in": 3584},
        "27b-it": {"name": "gemma-2-27b-it", "d_in": 4608},
        "2b": {"name": "gemma-2-2b", "d_in": 2304},
        "9b": {"name": "gemma-2-9b", "d_in": 3584},
        "27b": {"name": "gemma-2-27b", "d_in": 4608},
    }
    model_info = next(
        (info for key, info in model_params.items() if key in repo_id), None
    )
    if not model_info:
        raise ValueError("Model name not found in repo_id.")

    model_name, d_in = model_info["name"], model_info["d_in"]

    # Hook specific parameters
    if "res" in repo_id:
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif "mlp" in repo_id:
        hook_name = f"blocks.{layer}.hook_mlp_out"
    elif "att" in repo_id:
        hook_name = f"blocks.{layer}.attn.hook_z"
        d_in = {"2b": 2048, "9b": 4096, "27b": 4608}.get(
            next(key for key in model_params if key in repo_id), d_in
        )
    else:
        raise ValueError("Hook name not found in folder_name.")

    return {
        "architecture": "jumprelu",
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "model_name": model_name,
        "hook_name": hook_name,
        "hook_layer": layer,
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": None,
    }


def gemma_2_sae_loader(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    d_sae_override: Optional[int] = None,
    layer_override: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Custom loader for Gemma 2 SAEs.
    """
    cfg_dict = get_gemma_2_config(repo_id, folder_name, d_sae_override, layer_override)
    cfg_dict["device"] = device

    # Apply overrides if provided
    if cfg_overrides is not None:
        cfg_dict.update(cfg_overrides)

    # Download the SAE weights
    sae_path = hf_hub_download(
        repo_id=repo_id,
        filename="params.npz",
        subfolder=folder_name,
        force_download=force_download,
    )

    # Load and convert the weights
    state_dict = {}
    with np.load(sae_path) as data:
        for key in data.keys():
            state_dict_key = "W_" + key[2:] if key.startswith("w_") else key
            state_dict[state_dict_key] = (
                torch.tensor(data[key]).to(dtype=torch.float32).to(device)
            )

    # Handle scaling factor
    if "scaling_factor" in state_dict:
        if torch.allclose(
            state_dict["scaling_factor"], torch.ones_like(state_dict["scaling_factor"])
        ):
            del state_dict["scaling_factor"]
            cfg_dict["finetuning_scaling_factor"] = False
        else:
            assert cfg_dict[
                "finetuning_scaling_factor"
            ], "Scaling factor is present but finetuning_scaling_factor is False."
            state_dict["finetuning_scaling_factor"] = state_dict.pop("scaling_factor")
    else:
        cfg_dict["finetuning_scaling_factor"] = False

    # No sparsity tensor for Gemma 2 SAEs
    log_sparsity = None

    return cfg_dict, state_dict, log_sparsity


def get_dictionary_learning_config_1(config: dict[str, Any]) -> dict[str, Any]:
    """
    Suitable for SAEs from https://huggingface.co/canrager/lm_sae.
    """
    trainer = config["trainer"]
    buffer = config["buffer"]

    hook_point_name = f"blocks.{trainer['layer']}.hook_resid_post"

    activation_fn_str = "topk" if "topk" in config.get("path", "") else "relu"
    activation_fn_kwargs = {"k": trainer["k"]} if activation_fn_str == "topk" else {}

    return {
        "architecture": (
            "gated" if trainer["dict_class"] == "GatedAutoEncoder" else "standard"
        ),
        "d_in": trainer["activation_dim"],
        "d_sae": trainer["dict_size"],
        "dtype": "float32",
        "device": "cpu",
        "model_name": trainer["lm_name"].split("/")[-1],
        "hook_name": hook_point_name,
        "hook_layer": trainer["layer"],
        "hook_head_index": None,
        "activation_fn_str": activation_fn_str,
        "activation_fn_kwargs": activation_fn_kwargs,
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "dataset_trust_remote_code": False,
        "context_size": buffer["ctx_len"],
        "normalize_activations": "none",
        "neuronpedia_id": None,
    }


def dictionary_learning_sae_loader_1(
    repo_id: str,
    folder_name: str,
    device: str = "cpu",
    force_download: bool = False,
    cfg_overrides: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Suitable for SAEs from https://huggingface.co/canrager/lm_sae.
    """
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{folder_name}/config.json",
        force_download=force_download,
    )
    encoder_path = hf_hub_download(
        repo_id=repo_id, filename=f"{folder_name}/ae.pt", force_download=force_download
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    cfg_dict = get_dictionary_learning_config_1(config)
    if cfg_overrides:
        cfg_dict.update(cfg_overrides)

    encoder = torch.load(encoder_path, map_location="cpu")

    state_dict = {
        "W_enc": encoder["encoder.weight"].T,
        "W_dec": encoder["decoder.weight"].T,
        "b_dec": encoder.get(
            "b_dec", encoder.get("bias", encoder.get("decoder_bias", None))
        ),
    }

    if "encoder.bias" in encoder:
        state_dict["b_enc"] = encoder["encoder.bias"]

    if "mag_bias" in encoder:
        state_dict["b_mag"] = encoder["mag_bias"]
    if "gate_bias" in encoder:
        state_dict["b_gate"] = encoder["gate_bias"]
    if "r_mag" in encoder:
        state_dict["r_mag"] = encoder["r_mag"]

    return cfg_dict, state_dict, None


# Helper function to get dtype from string
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


NAMED_PRETRAINED_SAE_LOADERS: dict[str, PretrainedSaeLoader] = {
    "sae_lens": sae_lens_loader,  # type: ignore
    "connor_rob_hook_z": connor_rob_hook_z_loader,  # type: ignore
    "gemma_2": gemma_2_sae_loader,
    "dictionary_learning_1": dictionary_learning_sae_loader_1,
}
