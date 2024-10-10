from typing import Optional

import pytest

from sae_lens import __version__
from sae_lens.config import CacheActivationsRunnerConfig, LanguageModelSAERunnerConfig

TINYSTORIES_MODEL = "tiny-stories-1M"
TINYSTORIES_DATASET = "roneneldan/TinyStories"


def test_sae_training_runner_config_runs_with_defaults():
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    _ = LanguageModelSAERunnerConfig()

    assert True


def test_sae_training_runner_config_total_training_tokens():
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    cfg = LanguageModelSAERunnerConfig()

    assert cfg.total_training_tokens == 2000000


def test_sae_training_runner_config_total_training_steps():
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    cfg = LanguageModelSAERunnerConfig()

    assert cfg.total_training_steps == 488


def test_sae_training_runner_config_get_sae_base_parameters():
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    cfg = LanguageModelSAERunnerConfig()

    expected_config = {
        "architecture": "standard",
        "d_in": 512,
        "d_sae": 2048,
        "activation_fn_str": "relu",
        "activation_fn_kwargs": {},
        "apply_b_dec_to_input": True,
        "dtype": "float32",
        "model_name": "gelu-2l",
        "hook_name": "blocks.0.hook_mlp_out",
        "hook_layer": 0,
        "hook_head_index": None,
        "device": "cpu",
        "context_size": 128,
        "prepend_bos": True,
        "finetuning_scaling_factor": False,
        "dataset_path": "",
        "dataset_trust_remote_code": True,
        "sae_lens_training_version": str(__version__),
        "normalize_activations": "none",
        "model_from_pretrained_kwargs": {
            "center_writing_weights": False,
        },
        "seqpos_slice": (None,),
    }
    assert expected_config == cfg.get_base_sae_cfg_dict()


def test_sae_training_runner_config_raises_error_if_resume_true():
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    with pytest.raises(ValueError):
        _ = LanguageModelSAERunnerConfig(resume=True)
    assert True


def test_sae_training_runner_config_raises_error_if_d_sae_and_expansion_factor_not_none():
    with pytest.raises(ValueError):
        _ = LanguageModelSAERunnerConfig(d_sae=128, expansion_factor=4)
    assert True


def test_sae_training_runner_config_expansion_factor():
    cfg = LanguageModelSAERunnerConfig()

    assert cfg.expansion_factor == 4


test_cases_for_seqpos = [
    ((None, 10, -1), AssertionError),
    ((None, 10, 0), AssertionError),
    ((5, 5, None), AssertionError),
    ((6, 3, None), AssertionError),
]


@pytest.mark.parametrize("seqpos_slice, expected_error", test_cases_for_seqpos)
def test_sae_training_runner_config_seqpos(
    seqpos_slice: tuple[int, int], expected_error: Optional[AssertionError]
):
    context_size = 10
    if expected_error is AssertionError:
        with pytest.raises(expected_error):
            LanguageModelSAERunnerConfig(
                seqpos_slice=seqpos_slice,
                context_size=context_size,
            )
    else:
        LanguageModelSAERunnerConfig(
            seqpos_slice=seqpos_slice,
            context_size=context_size,
        )


@pytest.mark.parametrize("seqpos_slice, expected_error", test_cases_for_seqpos)
def test_cache_activations_runner_config_seqpos(
    seqpos_slice: tuple[int, int], expected_error: Optional[AssertionError]
):
    context_size = 10
    if expected_error is AssertionError:
        with pytest.raises(expected_error):
            CacheActivationsRunnerConfig(
                seqpos_slice=seqpos_slice,
                context_size=context_size,
            )
    else:
        CacheActivationsRunnerConfig(
            seqpos_slice=seqpos_slice,
            context_size=context_size,
        )
