import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from sae_lens.training.sparse_autoencoder import SparseAutoencoderBase


def test_SparseAutoencoder_from_pretrained_loads_from_hugginface_using_shorthand():
    sae, _ = SparseAutoencoderBase.from_pretrained(
        release="gpt2-small-res-jb",
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
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

    assert isinstance(sae, SparseAutoencoderBase)
    assert sae.cfg.model_name == "gpt2-small"
    assert sae.cfg.hook_point == "blocks.0.hook_resid_pre"

    for k in sae.state_dict().keys():
        if k == "scaling_factor":
            continue
        assert torch.allclose(sae.state_dict()[k], state_dict[k])


def test_SparseAutoencoder_from_pretrained_errors_for_invalid_releases():
    with pytest.raises(ValueError):
        SparseAutoencoderBase.from_pretrained(
            release="wrong",
            sae_id="blocks.0.hook_resid_pre",
            device="cpu",
        )


def test_SparseAutoencoder_from_pretrained_errors_for_invalid_sae_ids():
    with pytest.raises(ValueError):
        SparseAutoencoderBase.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="wrong",
            device="cpu",
        )
