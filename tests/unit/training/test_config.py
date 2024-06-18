import pytest

from sae_lens import __version__
from sae_lens.config import LanguageModelSAERunnerConfig

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
        "d_in": 512,
        "d_sae": 2048,
        "activation_fn_str": "relu",
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
        "dataset_path": "NeelNanda/c4-tokenized-2b",
        "dataset_trust_remote_code": True,
        "sae_lens_training_version": str(__version__),
        "normalize_activations": "none",
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
