import json
from dataclasses import fields
from typing import Any, Protocol

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from typeguard import TypeCheckError, check_type

from sae_lens import __version__
from sae_lens.training.config import LanguageModelSAERunnerConfig


# loaders take in a repo_id, folder_name, device, and whether to force download, and returns a tuple of config and state_dict
class PretrainedSaeLoader(Protocol):
    def __call__(
        self,
        repo_id: str,
        folder_name: str,
        device: str | torch.device | None = None,
        force_download: bool = False,
    ) -> tuple[LanguageModelSAERunnerConfig, dict[str, torch.Tensor]]: ...


def sae_lens_loader(
    repo_id: str,
    folder_name: str,
    device: str | torch.device | None = None,
    force_download: bool = False,
) -> tuple[LanguageModelSAERunnerConfig, dict[str, torch.Tensor]]:
    cfg_filename = f"{folder_name}/cfg.json"
    cfg_path = hf_hub_download(
        repo_id=repo_id, filename=cfg_filename, force_download=force_download
    )

    weights_filename = f"{folder_name}/sae_weights.safetensors"
    sae_path = hf_hub_download(
        repo_id=repo_id, filename=weights_filename, force_download=force_download
    )

    return load_pretrained_sae_lens_sae_components(cfg_path, sae_path, device)


def load_pretrained_sae_lens_sae_components(
    cfg_path: str, weight_path: str, device: str | torch.device | None = None
) -> tuple[LanguageModelSAERunnerConfig, dict[str, torch.Tensor]]:
    with open(cfg_path, "r") as f:
        config = json.load(f)

    config_fields = fields(LanguageModelSAERunnerConfig)
    fields_by_name = {f.name: f for f in config_fields}

    def is_valid_config_field(var_name: str, val: Any) -> bool:
        if var_name not in fields_by_name:
            return False
        try:
            check_type(val, fields_by_name[var_name].type)
            return True
        except TypeCheckError:
            return False

    # filter out any invalid config fields
    config = {k: v for k, v in config.items() if is_valid_config_field(k, v)}
    config["verbose"] = False
    config["device"] = device

    # TODO: if we change our SAE implementation such that old versions need conversion to be
    # loaded, we can inspect the original "sae_lens_version" and apply a conversion function here.
    config["sae_lens_version"] = __version__

    config = LanguageModelSAERunnerConfig(**config)

    tensors = {}
    with safe_open(weight_path, framework="pt", device=device) as f:  # type: ignore
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    # old saves may not have scaling factors.
    if "scaling_factor" not in tensors:
        assert isinstance(config.d_sae, int)
        tensors["scaling_factor"] = torch.ones(
            config.d_sae, dtype=config.dtype, device=config.device
        )

    return config, tensors


# TODO: add more loaders for other SAEs not trained by us

NAMED_PRETRAINED_SAE_LOADERS: dict[str, PretrainedSaeLoader] = {
    "sae_lens": sae_lens_loader,
}
