from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Optional

import yaml


@dataclass
class PretrainedSAELookup:
    release: str
    repo_id: str
    model: str
    conversion_func: str | None
    saes_map: dict[str, str]  # id -> path
    expected_var_explained: dict[str, float]
    expected_l0: dict[str, float]
    neuronpedia_id: dict[str, str]
    config_overrides: dict[str, str] | dict[str, dict[str, str | bool | int]] | None


@cache
def get_pretrained_saes_directory() -> dict[str, PretrainedSAELookup]:
    package = "sae_lens"
    # Access the file within the package using importlib.resources
    directory: dict[str, PretrainedSAELookup] = {}
    with resources.open_text(package, "pretrained_saes.yaml") as file:
        # Load the YAML file content
        data = yaml.safe_load(file)
        for release, value in data.items():
            saes_map: dict[str, str] = {}
            var_explained_map: dict[str, float] = {}
            l0_map: dict[str, float] = {}
            neuronpedia_id_map: dict[str, str] = {}

            assert "saes" in value, f"Missing 'saes' key in {release}"
            for hook_info in value["saes"]:
                saes_map[hook_info["id"]] = hook_info["path"]
                var_explained_map[hook_info["id"]] = hook_info.get(
                    "variance_explained", 1.00
                )
                l0_map[hook_info["id"]] = hook_info.get("l0", 0.00)
                neuronpedia_id_map[hook_info["id"]] = hook_info.get("neuronpedia")
            directory[release] = PretrainedSAELookup(
                release=release,
                repo_id=value["repo_id"],
                model=value["model"],
                conversion_func=value.get("conversion_func"),
                saes_map=saes_map,
                expected_var_explained=var_explained_map,
                expected_l0=l0_map,
                neuronpedia_id=neuronpedia_id_map,
                config_overrides=value.get("config_overrides"),
            )
    return directory


def get_norm_scaling_factor(release: str, sae_id: str) -> Optional[float]:
    """
    Retrieve the norm_scaling_factor for a specific SAE if it exists.

    Args:
        release (str): The release name of the SAE.
        sae_id (str): The ID of the specific SAE.

    Returns:
        Optional[float]: The norm_scaling_factor if it exists, None otherwise.
    """
    package = "sae_lens"
    with resources.open_text(package, "pretrained_saes.yaml") as file:
        data = yaml.safe_load(file)
        if release in data:
            for sae_info in data[release]["saes"]:
                if sae_info["id"] == sae_id:
                    return sae_info.get("norm_scaling_factor")
    return None


def get_repo_id_and_folder_name(release: str, sae_id: str) -> tuple[str, str]:
    saes_directory = get_pretrained_saes_directory()
    sae_info = saes_directory.get(release, None)

    if sae_info is None:
        return release, sae_id

    if sae_id not in sae_info.saes_map:
        raise ValueError(f"SAE ID '{sae_id}' not found in release '{release}'")

    repo_id = sae_info.repo_id
    folder_name = sae_info.saes_map[sae_id]
    return repo_id, folder_name
