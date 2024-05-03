import json
import os
import pathlib
from typing import Optional, Tuple

import torch
from huggingface_hub import hf_hub_download, list_repo_tree
from safetensors import safe_open
from tqdm import tqdm

from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.sparse_autoencoder import SparseAutoencoder


def load_sparsity(path: str) -> torch.Tensor:
    sparsity_path = os.path.join(path, "sparsity.safetensors")
    with safe_open(sparsity_path, framework="pt", device="cpu") as f:  # type: ignore
        sparsity = f.get_tensor("sparsity")
    return sparsity


def download_sae_from_hf(
    repo_id: str = "jbloom/GPT2-Small-SAEs-Reformatted",
    folder_name: str = "blocks.0.hook_resid_pre",
    force_download: bool = False,
) -> Tuple[str, str, Optional[str]]:

    FILENAME = f"{folder_name}/cfg.json"
    cfg_path = hf_hub_download(
        repo_id=repo_id, filename=FILENAME, force_download=force_download
    )

    FILENAME = f"{folder_name}/sae_weights.safetensors"
    sae_path = hf_hub_download(
        repo_id=repo_id, filename=FILENAME, force_download=force_download
    )

    try:
        FILENAME = f"{folder_name}/sparsity.safetensors"
        sparsity_path = hf_hub_download(
            repo_id=repo_id, filename=FILENAME, force_download=force_download
        )
    except:  # noqa
        sparsity_path = None

    return cfg_path, sae_path, sparsity_path


def load_sae_from_local_path(path: str) -> Tuple[SparseAutoencoder, torch.Tensor]:
    sae = SparseAutoencoder.load_from_pretrained(path)
    sparsity = load_sparsity(path)
    return sae, sparsity


def get_gpt2_res_jb_saes(
    hook_point: Optional[str] = None,
    device: str = "cpu",
) -> tuple[dict[str, SparseAutoencoder], dict[str, torch.Tensor]]:
    """
    Download the sparse autoencoders for the GPT2-Small model with residual connections
    from the repository of jbloom. You can specify a hook_point to download only one
    of the sparse autoencoders if desired.

    """

    GPT2_SMALL_RESIDUAL_SAES_REPO_ID = "jbloom/GPT2-Small-SAEs-Reformatted"
    GPT2_SMALL_RESIDUAL_SAES_HOOK_POINTS = [
        f"blocks.{layer}.hook_resid_pre" for layer in range(12)
    ] + ["blocks.11.hook_resid_post"]

    if hook_point is not None:
        assert hook_point in GPT2_SMALL_RESIDUAL_SAES_HOOK_POINTS, (
            f"hook_point must be one of {GPT2_SMALL_RESIDUAL_SAES_HOOK_POINTS}"
            f"but got {hook_point}"
        )
        GPT2_SMALL_RESIDUAL_SAES_HOOK_POINTS = [hook_point]

    saes = {}
    sparsities = {}
    for hook_point in tqdm(GPT2_SMALL_RESIDUAL_SAES_HOOK_POINTS):

        _, sae_path, _ = download_sae_from_hf(
            repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, folder_name=hook_point
        )

        # Then use our function to download the files
        folder_path = os.path.dirname(sae_path)
        sae = SparseAutoencoder.load_from_pretrained(folder_path, device=device)
        sparsity = load_sparsity(folder_path)
        sparsity = sparsity.to(device)
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


def convert_old_to_modern_saelens_format(
    pytorch_file: str, out_folder: str = "", force: bool = False
):
    """
    Reads a pretrained SAE from the old pickle-style SAELens .pt format, then saves a modern-format SAELens SAE.

    Arguments:
    ----------
    pytorch_file: str
        Path of old format file to open.
    out_folder: str, optional
        Path where new SAE will be stored; if None, out_folder = pytorch_file with the '.pt' removed.
    force: bool, optional
        If out_folder already exists, this function will not save unless force=True.
    """
    file_path = pathlib.Path(pytorch_file)
    if out_folder == "":
        out_f = file_path.parent / file_path.stem
    else:
        out_f = pathlib.Path(out_folder)
    if (not force) and out_f.exists():
        raise FileExistsError(f"{out_folder} already exists and force=False")
    out_f.mkdir(exist_ok=True, parents=True)

    # Load model & save in new format.
    autoencoder = SparseAutoencoder.load_from_pretrained_legacy(str(file_path))
    autoencoder.save_model(str(out_f))


def get_gpt2_small_ckrk_attn_out_saes() -> dict[str, SparseAutoencoder]:

    REPO_ID = "ckkissane/attn-saes-gpt2-small-all-layers"

    # list all files in repo
    saes_weights = {}
    sae_configs = {}
    repo_files = list_repo_tree(REPO_ID)
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
