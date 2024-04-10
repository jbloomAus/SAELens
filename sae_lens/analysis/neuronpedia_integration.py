import os
import webbrowser

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from tqdm import tqdm

from sae_lens.training.sparse_autoencoder import SparseAutoencoder


def load_sparsity(path: str) -> torch.Tensor:
    sparsity_path = os.path.join(path, "sparsity.safetensors")
    with safe_open(sparsity_path, framework="pt", device="cpu") as f:  # type: ignore
        sparsity = f.get_tensor("sparsity")
    return sparsity


def get_all_gpt2_small_saes() -> (
    tuple[dict[str, SparseAutoencoder], dict[str, torch.Tensor]]
):

    GPT2_SMALL_RESIDUAL_SAES_REPO_ID = "jbloom/GPT2-Small-SAEs-Reformatted"
    GPT2_SMALL_RESIDUAL_SAES_HOOK_POINTS = [
        f"blocks.{layer}.hook_resid_pre" for layer in range(12)
    ] + ["blocks.11.hook_resid_post"]

    saes = {}
    sparsities = {}
    for hook_point in tqdm(GPT2_SMALL_RESIDUAL_SAES_HOOK_POINTS):
        # download the files required:
        FILENAME = f"{hook_point}/cfg.json"
        hf_hub_download(repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, filename=FILENAME)

        FILENAME = f"{hook_point}/sae_weights.safetensors"
        path = hf_hub_download(
            repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, filename=FILENAME
        )

        FILENAME = f"{hook_point}/sparsity.safetensors"
        path = hf_hub_download(
            repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, filename=FILENAME
        )

        # Then use our function to download the files
        folder_path = os.path.dirname(path)
        sae = SparseAutoencoder.load_from_pretrained(folder_path)
        sparsity = load_sparsity(folder_path)
        saes[hook_point] = sae
        sparsities[hook_point] = sparsity

    return saes, sparsities


def open_neuronpedia(feature_id: int, layer: int = 0):

    path_to_html = f"https://www.neuronpedia.org/gpt2-small/{layer}-res-jb/{feature_id}"

    print(f"Feature {feature_id}")
    webbrowser.open_new_tab(path_to_html)
