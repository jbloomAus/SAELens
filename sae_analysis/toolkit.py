import webbrowser

import torch
from huggingface_hub import hf_hub_download

from sae_training.sparse_autoencoder import SparseAutoencoder


def get_all_gpt2_small_saes() -> (
    tuple[dict[str, SparseAutoencoder], dict[str, torch.Tensor]]
):

    REPO_ID = "jbloom/GPT2-Small-SAEs"
    gpt2_small_sparse_autoencoders = {}
    gpt2_small_saes_log_feature_sparsities = {}
    for layer in range(12):
        FILENAME = f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        sae = SparseAutoencoder.load_from_pretrained(f"{path}")
        sae.cfg.use_ghost_grads = False
        gpt2_small_sparse_autoencoders[sae.cfg.hook_point] = sae

        FILENAME = f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576_log_feature_sparsity.pt"
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        log_feature_density = torch.load(path, map_location=sae.cfg.device)
        gpt2_small_saes_log_feature_sparsities[sae.cfg.hook_point] = log_feature_density

    # get the final one
    layer = 11
    FILENAME = (
        f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_post_24576.pt"
    )
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    sae = SparseAutoencoder.load_from_pretrained(f"{path}")
    sae.cfg.use_ghost_grads = False
    gpt2_small_sparse_autoencoders[sae.cfg.hook_point] = sae

    FILENAME = f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_post_24576_log_feature_sparsity.pt"
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    log_feature_density = torch.load(path, map_location=sae.cfg.device)
    gpt2_small_saes_log_feature_sparsities[sae.cfg.hook_point] = log_feature_density

    return gpt2_small_sparse_autoencoders, gpt2_small_saes_log_feature_sparsities


def open_neuronpedia(feature_id: int, layer: int = 0):

    path_to_html = f"https://www.neuronpedia.org/gpt2-small/{layer}-res-jb/{feature_id}"

    print(f"Feature {feature_id}")
    webbrowser.open_new_tab(path_to_html)
