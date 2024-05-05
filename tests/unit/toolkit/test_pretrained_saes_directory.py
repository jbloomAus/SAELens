from sae_lens.toolkit.pretrained_saes_directory import (
    PretrainedSAELookup,
    get_pretrained_saes_directory,
)


def test_get_pretrained_saes_directory():
    sae_directory = get_pretrained_saes_directory()
    assert isinstance(sae_directory, dict)
    assert sae_directory["gpt2-small-res-jb"] == PretrainedSAELookup(
        release="gpt2-small-res-jb",
        repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
        model="gpt2-small",
        conversion_func=None,
        saes_map={
            "blocks.0.hook_resid_pre": "blocks.0.hook_resid_pre",
            "blocks.1.hook_resid_pre": "blocks.1.hook_resid_pre",
            "blocks.2.hook_resid_pre": "blocks.2.hook_resid_pre",
            "blocks.3.hook_resid_pre": "blocks.3.hook_resid_pre",
            "blocks.4.hook_resid_pre": "blocks.4.hook_resid_pre",
            "blocks.5.hook_resid_pre": "blocks.5.hook_resid_pre",
            "blocks.6.hook_resid_pre": "blocks.6.hook_resid_pre",
            "blocks.7.hook_resid_pre": "blocks.7.hook_resid_pre",
            "blocks.8.hook_resid_pre": "blocks.8.hook_resid_pre",
            "blocks.9.hook_resid_pre": "blocks.9.hook_resid_pre",
            "blocks.10.hook_resid_pre": "blocks.10.hook_resid_pre",
            "blocks.11.hook_resid_pre": "blocks.11.hook_resid_pre",
            "blocks.11.hook_resid_post": "blocks.11.hook_resid_post",
        },
    )
