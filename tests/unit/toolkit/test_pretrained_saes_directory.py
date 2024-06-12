from sae_lens.toolkit.pretrained_saes_directory import (
    PretrainedSAELookup,
    get_pretrained_saes_directory,
)


def test_get_pretrained_saes_directory():
    sae_directory = get_pretrained_saes_directory()
    assert isinstance(sae_directory, dict)
    expected_result = PretrainedSAELookup(
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
        expected_var_explained={
            "blocks.0.hook_resid_pre": 0.999,
            "blocks.1.hook_resid_pre": 0.999,
            "blocks.2.hook_resid_pre": 0.999,
            "blocks.3.hook_resid_pre": 0.999,
            "blocks.4.hook_resid_pre": 0.9,
            "blocks.5.hook_resid_pre": 0.9,
            "blocks.6.hook_resid_pre": 0.9,
            "blocks.7.hook_resid_pre": 0.9,
            "blocks.8.hook_resid_pre": 0.9,
            "blocks.9.hook_resid_pre": 0.77,
            "blocks.10.hook_resid_pre": 0.77,
            "blocks.11.hook_resid_pre": 0.77,
            "blocks.11.hook_resid_post": 0.77,
        },
        expected_l0={
            "blocks.0.hook_resid_pre": 10.0,
            "blocks.1.hook_resid_pre": 10.0,
            "blocks.2.hook_resid_pre": 18.0,
            "blocks.3.hook_resid_pre": 23.0,
            "blocks.4.hook_resid_pre": 31.0,
            "blocks.5.hook_resid_pre": 41.0,
            "blocks.6.hook_resid_pre": 51.0,
            "blocks.7.hook_resid_pre": 54.0,
            "blocks.8.hook_resid_pre": 60.0,
            "blocks.9.hook_resid_pre": 70.0,
            "blocks.10.hook_resid_pre": 52.0,
            "blocks.11.hook_resid_pre": 56.0,
            "blocks.11.hook_resid_post": 70.0,
        },
    )

    assert sae_directory["gpt2-small-res-jb"] == expected_result
