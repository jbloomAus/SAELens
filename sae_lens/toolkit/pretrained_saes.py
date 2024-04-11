import json
import os

import torch
from huggingface_hub import hf_hub_download, list_files_info
from safetensors import safe_open
from tqdm import tqdm

from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.sparse_autoencoder import SparseAutoencoder


def load_sparsity(path: str) -> torch.Tensor:
    sparsity_path = os.path.join(path, "sparsity.safetensors")
    with safe_open(sparsity_path, framework="pt", device="cpu") as f:  # type: ignore
        sparsity = f.get_tensor("sparsity")
    return sparsity


def get_gpt2_res_jb_saes() -> (
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


def convert_connor_rob_sae_to_our_saelens_format(
    state_dict: dict[str, torch.Tensor],
    config: dict[str, int | str],
    device: str = "cpu",
):
    """

    # can get session like so.
        model, ae_alt, activation_store = LMSparseAutoencoderSessionloader(
            cfg
        ).load_sae_training_group_session()
        next(iter(ae_alt))[1].load_state_dict(state_dict)
        return model, ae_alt, activation_store

    """

    expansion_factor = int(config["dict_size"]) // int(config["act_size"])

    cfg = LanguageModelSAERunnerConfig(
        model_name=config["model_name"],  # type: ignore
        hook_point=config["act_name"],  # type: ignore
        hook_point_layer=config["layer"],  # type: ignore
        # data
        # dataset_path = "/share/data/datasets/pile/the-eye.eu/public/AI/pile/train", # Training set of The Pile
        dataset_path="NeelNanda/openwebtext-tokenized-9b",
        is_dataset_tokenized=True,
        d_in=config["act_size"],  # type: ignore
        expansion_factor=expansion_factor,
        context_size=config["seq_len"],  # type: ignore
        device=device,
        store_batch_size=32,
        n_batches_in_buffer=10,
        prepend_bos=False,
        verbose=False,
        dtype=torch.float32,
    )

    ae_alt = SparseAutoencoder(cfg)
    ae_alt.load_state_dict(state_dict)
    return ae_alt


def get_gpt2_small_ckrk_attn_out_saes() -> dict[str, SparseAutoencoder]:

    REPO_ID = "ckkissane/attn-saes-gpt2-small-all-layers"

    # list all files in repo
    saes_weights = {}
    sae_configs = {}
    repo_files = list_files_info(REPO_ID)
    for i in tqdm(repo_files):
        file_name = i.path
        if file_name.endswith(".pt"):
            # print(f"Downloading {file_name}")
            path = hf_hub_download(REPO_ID, file_name)
            name = path.split("/")[-1].split(".pt")[0]
            saes_weights[name] = torch.load(path, map_location="mps")
        elif file_name.endswith(".json"):
            # print(f"Downloading {file_name}")
            config_path = hf_hub_download(REPO_ID, file_name)
            name = config_path.split("/")[-1].split("_cfg.json")[0]
            sae_configs[name] = json.load(open(config_path, "r"))

    saes = {}
    for name, config in sae_configs.items():
        print(f"Loading {name}")
        saes[name] = convert_connor_rob_sae_to_our_saelens_format(
            saes_weights[name], config
        )

    return saes
