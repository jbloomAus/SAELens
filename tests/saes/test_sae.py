import pytest

from sae_lens.registry import get_sae_class, get_sae_training_class
from sae_lens.saes.sae import SAEConfig, TrainingSAEConfig
from tests.helpers import (
    ALL_ARCHITECTURES,
    build_sae_training_cfg_for_arch,
)


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
def test_TrainingSAEConfig_to_and_from_dict_all_architectures(architecture: str):
    cfg = build_sae_training_cfg_for_arch(architecture=architecture)
    reloaded_cfg = TrainingSAEConfig.from_dict(cfg.to_dict())
    assert reloaded_cfg.to_dict() == cfg.to_dict()
    assert reloaded_cfg.__class__ == cfg.__class__
    assert reloaded_cfg.__class__ == get_sae_training_class(architecture)[1]


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
def test_SAEConfig_to_and_from_dict_all_architectures(architecture: str):
    cfg_dict = build_sae_training_cfg_for_arch(architecture).get_base_sae_cfg_dict()
    reloaded_cfg = SAEConfig.from_dict(cfg_dict)
    assert reloaded_cfg.architecture() == architecture
    assert reloaded_cfg.to_dict() == cfg_dict
    assert reloaded_cfg.__class__ == get_sae_class(architecture)[1]
