import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from sae_lens.sae import SAE


def test_SparseAutoencoder_from_pretrained_loads_from_hugginface_using_shorthand():
    sae, original_cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.0.hook_resid_pre",
        device="cpu",
    )

    assert (
        sae.cfg.neuronpedia_id == "gpt2-small/0-res-jb"
    )  # what we expect from the yml

    # it should match what we get when manually loading from hf
    repo_id = "jbloom/GPT2-Small-SAEs-Reformatted"
    hook_point = "blocks.0.hook_resid_pre"
    filename = f"{hook_point}/sae_weights.safetensors"
    weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
    state_dict = {}
    with safe_open(weight_path, framework="pt", device="cpu") as f:  # type: ignore
        for k in f.keys():  # noqa: SIM118
            state_dict[k] = f.get_tensor(k)

    assert isinstance(sae, SAE)
    assert sae.cfg.model_name == "gpt2-small"
    assert sae.cfg.hook_name == "blocks.0.hook_resid_pre"

    assert isinstance(original_cfg_dict, dict)

    assert isinstance(sparsity, torch.Tensor)
    assert sparsity.shape == (sae.cfg.d_sae,)
    assert sparsity.max() < 0.0

    for k in sae.state_dict():
        if k == "finetuning_scaling_factor":
            continue
        assert torch.allclose(sae.state_dict()[k], state_dict[k])


def test_SparseAutoencoder_from_pretrained_can_load_arbitrary_saes_from_hugginface():
    sae, original_cfg_dict, sparsity = SAE.from_pretrained(
        release="jbloom/GPT2-Small-SAEs-Reformatted",
        sae_id="blocks.0.hook_resid_pre",
        device="cpu",
    )

    # it should match what we get when manually loading from hf
    repo_id = "jbloom/GPT2-Small-SAEs-Reformatted"
    hook_point = "blocks.0.hook_resid_pre"
    filename = f"{hook_point}/sae_weights.safetensors"
    weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
    state_dict = {}
    with safe_open(weight_path, framework="pt", device="cpu") as f:  # type: ignore
        for k in f.keys():  # noqa: SIM118
            state_dict[k] = f.get_tensor(k)

    assert isinstance(sae, SAE)
    assert sae.cfg.model_name == "gpt2-small"
    assert sae.cfg.hook_name == "blocks.0.hook_resid_pre"

    assert isinstance(original_cfg_dict, dict)

    assert isinstance(sparsity, torch.Tensor)
    assert sparsity.shape == (sae.cfg.d_sae,)
    assert sparsity.max() < 0.0

    for k in sae.state_dict():
        if k == "finetuning_scaling_factor":
            continue
        assert torch.allclose(sae.state_dict()[k], state_dict[k])


def test_SparseAutoencoder_from_pretrained_errors_for_invalid_releases():
    with pytest.raises(ValueError):
        SAE.from_pretrained(
            release="wrong",
            sae_id="blocks.0.hook_resid_pre",
            device="cpu",
        )


def test_SparseAutoencoder_from_pretrained_errors_for_invalid_sae_ids():
    with pytest.raises(ValueError):
        SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="wrong",
            device="cpu",
        )
