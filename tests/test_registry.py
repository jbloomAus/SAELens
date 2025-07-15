import pytest

from sae_lens import (
    StandardSAE,
    StandardSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)
from sae_lens.registry import register_sae_class, register_sae_training_class


def test_register_sae_class_errors_if_arch_is_already_registered():
    with pytest.raises(
        ValueError, match="SAE class for architecture standard already registered."
    ):
        register_sae_class("standard", StandardSAE, StandardSAEConfig)


def test_register_sae_training_class_errors_if_arch_is_already_registered():
    with pytest.raises(
        ValueError,
        match="SAE training class for architecture standard already registered.",
    ):
        register_sae_training_class(
            "standard", StandardTrainingSAE, StandardTrainingSAEConfig
        )
