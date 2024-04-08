import torch
from huggingface_hub import hf_hub_download

from sae_lens.training.sae_group import SAEGroup


def test_SAEGroup_load_from_pretrained_can_load_old_autoencoders_from_huggingface():
    layer = 8  # pick a layer you want.
    REPO_ID = "jbloom/GPT2-Small-SAEs"
    FILENAME = (
        f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"
    )
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    sae_group = SAEGroup.load_from_pretrained(path=path)
    assert isinstance(sae_group, SAEGroup)
    assert len(sae_group) == 1
    assert sae_group.cfg.hook_point_layer == layer
    assert sae_group.cfg.model_name == "gpt2-small"

    reloaded_sae_group = SAEGroup.load_from_pretrained(path=path)
    assert reloaded_sae_group.cfg == sae_group.cfg

    orig_sae_state_dict = sae_group.autoencoders[0].state_dict()
    reloaded_sae_state_dict = reloaded_sae_group.autoencoders[0].state_dict()
    for key in orig_sae_state_dict.keys():
        assert torch.allclose(
            orig_sae_state_dict[key],
            reloaded_sae_state_dict[key],
        )
