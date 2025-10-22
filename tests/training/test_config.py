from typing import Type

import pytest

from sae_lens.config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
    _default_cached_activations_path,
)
from sae_lens.saes.jumprelu_sae import JumpReLUTrainingSAEConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig

test_cases_for_seqpos = [
    ((None, 10, -1), ValueError),
    ((None, 10, 0), ValueError),
    ((5, 5, None), ValueError),
    ((6, 3, None), ValueError),
]


@pytest.mark.parametrize("seqpos_slice, expected_error", test_cases_for_seqpos)
def test_sae_training_runner_config_seqpos(
    seqpos_slice: tuple[int, int], expected_error: Type[BaseException]
):
    context_size = 10
    with pytest.raises(expected_error):
        LanguageModelSAERunnerConfig(
            sae=StandardTrainingSAEConfig(d_in=10, d_sae=10),
            seqpos_slice=seqpos_slice,
            context_size=context_size,
        )


def test_LanguageModelSAERunnerConfig_hook_eval_deprecated_usage():
    with pytest.warns(
        DeprecationWarning,
        match="The 'hook_eval' field is deprecated and will be removed in v7.0.0. ",
    ):
        LanguageModelSAERunnerConfig(
            sae=StandardTrainingSAEConfig(d_in=10, d_sae=10),
            hook_eval="blocks.0.hook_output",
        )


@pytest.mark.parametrize("seqpos_slice, expected_error", test_cases_for_seqpos)
def test_cache_activations_runner_config_seqpos(
    seqpos_slice: tuple[int, int],
    expected_error: Type[BaseException],
):
    with pytest.raises(expected_error):
        CacheActivationsRunnerConfig(
            dataset_path="",
            model_name="",
            model_batch_size=1,
            hook_name="",
            d_in=1,
            training_tokens=100,
            context_size=10,
            seqpos_slice=seqpos_slice,
        )


def test_default_cached_activations_path():
    assert (
        _default_cached_activations_path(
            dataset_path="ds_path",
            model_name="model_name",
            hook_name="hook_name",
            hook_head_index=None,
        )
        == "activations/ds_path/model_name/hook_name"
    )


def test_LanguageModelSAERunnerConfig_to_dict_and_from_dict():
    cfg = LanguageModelSAERunnerConfig(
        sae=JumpReLUTrainingSAEConfig(
            d_in=5,
            d_sae=10,
            jumprelu_init_threshold=0.1,
            jumprelu_bandwidth=0.1,
            jumprelu_sparsity_loss_mode="tanh",
        ),
        seqpos_slice=(0, 10),
        context_size=10,
    )
    cfg_dict = cfg.to_dict()
    assert cfg_dict == cfg.to_dict()
    assert cfg == LanguageModelSAERunnerConfig.from_dict(cfg_dict)


def test_LanguageModelSAERunnerConfig_errors_when_loading_from_dict_with_missing_fields():
    cfg = LanguageModelSAERunnerConfig(
        sae=StandardTrainingSAEConfig(d_in=5, d_sae=10),
        seqpos_slice=(0, 10),
        context_size=10,
    )
    with pytest.raises(
        ValueError, match="sae field is required in the config dictionary"
    ):
        test_dict = cfg.to_dict()
        del test_dict["sae"]
        LanguageModelSAERunnerConfig.from_dict(test_dict)
    with pytest.raises(
        ValueError, match="architecture field is required in the sae dictionary"
    ):
        test_dict = cfg.to_dict()
        del test_dict["sae"]["architecture"]
        LanguageModelSAERunnerConfig.from_dict(test_dict)
    with pytest.raises(
        ValueError, match="logger field is required in the config dictionary"
    ):
        test_dict = cfg.to_dict()
        del test_dict["logger"]
        LanguageModelSAERunnerConfig.from_dict(test_dict)
