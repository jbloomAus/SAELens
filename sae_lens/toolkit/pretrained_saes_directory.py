from dataclasses import dataclass
from functools import cache
from importlib import resources

import yaml


@dataclass
class PretrainedSAELookup:
    release: str
    repo_id: str
    model: str
    conversion_func: str | None
    saes_map: dict[str, str]  # id -> path


@cache
def get_pretrained_saes_directory() -> dict[str, PretrainedSAELookup]:
    package = "sae_lens"
    # Access the file within the package using importlib.resources
    directory: dict[str, PretrainedSAELookup] = {}
    with resources.open_text(package, "pretrained_saes.yaml") as file:
        # Load the YAML file content
        data = yaml.safe_load(file)
        for release, value in data["SAE_LOOKUP"].items():
            saes_map: dict[str, str] = {}
            for hook_info in value["saes"]:
                saes_map[hook_info["id"]] = hook_info["path"]
            directory[release] = PretrainedSAELookup(
                release=release,
                repo_id=value["repo_id"],
                model=value["model"],
                conversion_func=value.get("conversion_func"),
                saes_map=saes_map,
            )
    return directory
