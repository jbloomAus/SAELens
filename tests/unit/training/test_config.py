from dataclasses import fields
from typing import Optional

import pytest

from sae_lens import __version__
from sae_lens.config import CacheActivationsRunnerConfig, LanguageModelSAERunnerConfig
from sae_lens.sae import SAEConfig
from sae_lens.training.training_sae import TrainingSAEConfig

TINYSTORIES_MODEL = "tiny-stories-1M"
TINYSTORIES_DATASET = "roneneldan/TinyStories"


def test_get_training_sae_cfg_dict_passes_scale_sparsity_penalty_by_decoder_norm():
    cfg = LanguageModelSAERunnerConfig(
        scale_sparsity_penalty_by_decoder_norm=True, normalize_sae_decoder=False
    )
    assert cfg.get_training_sae_cfg_dict()["scale_sparsity_penalty_by_decoder_norm"]
    cfg = LanguageModelSAERunnerConfig(
        scale_sparsity_penalty_by_decoder_norm=False, normalize_sae_decoder=False
    )
    assert not cfg.get_training_sae_cfg_dict()["scale_sparsity_penalty_by_decoder_norm"]


def test_get_training_sae_cfg_dict_has_all_relevant_options():
    cfg = LanguageModelSAERunnerConfig()
    cfg_dict = cfg.get_training_sae_cfg_dict()
    training_sae_opts = fields(TrainingSAEConfig)
    allowed_missing_fields = {"neuronpedia_id"}
    training_sae_field_names = {opt.name for opt in training_sae_opts}
    missing_fields = training_sae_field_names - allowed_missing_fields - cfg_dict.keys()
    assert missing_fields == set()


def test_get_base_sae_cfg_dict_has_all_relevant_options():
    cfg = LanguageModelSAERunnerConfig()
    cfg_dict = cfg.get_base_sae_cfg_dict()
    sae_opts = fields(SAEConfig)
    allowed_missing_fields = {"neuronpedia_id"}
    sae_field_names = {opt.name for opt in sae_opts}
    missing_fields = sae_field_names - allowed_missing_fields - cfg_dict.keys()
    assert missing_fields == set()


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
    if expected_error is AssertionError:
        with pytest.raises(expected_error):
            CacheActivationsRunnerConfig(
                dataset_path="",
                model_name="",
                model_batch_size=1,
                hook_name="",
                hook_layer=0,
                d_in=1,
                training_tokens=100,
                context_size=10,
                seqpos_slice=seqpos_slice,
            )
    else:
        CacheActivationsRunnerConfig(
            dataset_path="",
            model_name="",
            model_batch_size=1,
            hook_name="",
            hook_layer=0,
            d_in=1,
            training_tokens=100,
            context_size=10,
            seqpos_slice=seqpos_slice,
        )


def test_topk_architecture_requires_topk_activation():
    with pytest.raises(
        ValueError, match="If using topk architecture, activation_fn must be topk."
    ):
        LanguageModelSAERunnerConfig(architecture="topk", activation_fn="relu")


def test_topk_architecture_requires_k_parameter():
    with pytest.raises(
        ValueError,
        match="activation_fn_kwargs.k must be provided for topk architecture.",
    ):
        LanguageModelSAERunnerConfig(
            architecture="topk", activation_fn="topk", activation_fn_kwargs={}
        )


def test_topk_architecture_sets_topk_defaults():
    cfg = LanguageModelSAERunnerConfig(architecture="topk")
    assert cfg.activation_fn == "topk"
    assert cfg.activation_fn_kwargs == {"k": 100}
