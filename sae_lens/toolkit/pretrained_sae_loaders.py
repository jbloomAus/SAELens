import json
from typing import Any, Optional, Protocol

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils._errors import EntryNotFoundError
from safetensors import safe_open


# loaders take in a repo_id, folder_name, device, and whether to force download, and returns a tuple of config and state_dict
class PretrainedSaeLoader(Protocol):

    def __call__(
        self,
        repo_id: str,
        folder_name: str,
        device: str | torch.device | None = None,
        force_download: bool = False,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor], Optional[torch.Tensor]]: ...


def sae_lens_loader(
    repo_id: str,
    folder_name: str,
    device: Optional[str] = None,
    force_download: bool = False,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], Optional[torch.Tensor]]:
    cfg_filename = f"{folder_name}/cfg.json"
    cfg_path = hf_hub_download(
        repo_id=repo_id, filename=cfg_filename, force_download=force_download
    )

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

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return load_pretrained_sae_lens_sae_components(
        cfg_path, sae_path, device, log_sparsity_path=log_sparsity_path
    )


def connor_rob_hook_z_loader(
    repo_id: str,
    folder_name: str,
    device: Optional[str] = None,
    force_download: bool = False,
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
        "dataset_path": "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        "context_size": 128,
        "normalize_activations": False,
    }

    return cfg_dict, weights, None


def load_pretrained_sae_lens_sae_components(
    cfg_path: str,
    weight_path: str,
    device: str = "cpu",
    dtype: str = "float32",
    log_sparsity_path: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], Optional[torch.Tensor]]:
    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)

    # filter config for varnames
    cfg_dict["device"] = device
    cfg_dict["dtype"] = dtype

    # # # Removing this since we should add it during instantiation of the SAE, not the SAE config.
    # # TODO: if we change our SAE implementation such that old versions need conversion to be
    # # loaded, we can inspect the original "sae_lens_version" and apply a conversion function here.
    # config["sae_lens_version"] = __version__

    # check that the config is valid
    for key in ["d_sae", "d_in", "dtype"]:
        assert key in cfg_dict, f"config missing key {key}"

    state_dict = {}
    with safe_open(weight_path, framework="pt", device=device) as f:  # type: ignore
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

    # get sparsity tensor if it exists
    if log_sparsity_path is not None:
        with safe_open(log_sparsity_path, framework="pt", device=device) as f:  # type: ignore
            log_sparsity = f.get_tensor("sparsity")
    else:
        log_sparsity = None

    if "prepend_bos" not in cfg_dict:
        # default to True for backwards compatibility
        cfg_dict["prepend_bos"] = True

    if "apply_b_dec_to_input" not in cfg_dict:
        # default to True for backwards compatibility
        cfg_dict["apply_b_dec_to_input"] = True

    if "finetuning_scaling_factor" not in cfg_dict:
        # default to False for backwards compatibility
        cfg_dict["finetuning_scaling_factor"] = False

    if "sae_lens_training_version" not in cfg_dict:
        cfg_dict["sae_lens_training_version"] = None

    if "activation_fn" not in cfg_dict:
        cfg_dict["activation_fn_str"] = "relu"

    if "normalize_activations" not in cfg_dict:
        cfg_dict["normalize_activations"] = False

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

    return cfg_dict, state_dict, log_sparsity


# TODO: add more loaders for other SAEs not trained by us

NAMED_PRETRAINED_SAE_LOADERS: dict[str, PretrainedSaeLoader] = {
    "sae_lens": sae_lens_loader,  # type: ignore
    "connor_rob_hook_z": connor_rob_hook_z_loader,  # type: ignore
}
