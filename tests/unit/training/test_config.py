import pytest
import torch

from sae_lens.training.config import LanguageModelSAERunnerConfig

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
        "activation_fn": "relu",
        "apply_b_dec_to_input": True,
        "d_in": 512,
        "d_sae": 2048,
        "dtype": torch.float32,
        "hook_point": "blocks.0.hook_mlp_out",
        "hook_point_head_index": None,
        "hook_point_layer": 0,
        "model_name": "gelu-2l",
        "device": torch.device("cpu"),
    }
    assert expected_config == cfg.get_sae_base_parameters()


def test_sae_training_runner_config_raises_error_if_resume_true():
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    with pytest.raises(ValueError):
        _ = LanguageModelSAERunnerConfig(resume=True)
    assert True
