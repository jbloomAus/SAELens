from typing import TYPE_CHECKING

# avoid circular imports
if TYPE_CHECKING:
    from sae_lens.saes.sae import SAE, TrainingSAE

SAE_CLASS_REGISTRY: dict[str, "type[SAE]"] = {}
SAE_TRAINING_CLASS_REGISTRY: dict[str, "type[TrainingSAE]"] = {}


def register_sae_class(architecture: str, sae_class: "type[SAE]") -> None:
    if architecture in SAE_CLASS_REGISTRY:
        raise ValueError(
            f"SAE class for architecture {architecture} already registered."
        )
    SAE_CLASS_REGISTRY[architecture] = sae_class


def register_sae_training_class(
    architecture: str, sae_training_class: "type[TrainingSAE]"
) -> None:
    if architecture in SAE_TRAINING_CLASS_REGISTRY:
        raise ValueError(
            f"SAE training class for architecture {architecture} already registered."
        )
    SAE_TRAINING_CLASS_REGISTRY[architecture] = sae_training_class


def get_sae_class(architecture: str) -> "type[SAE]":
    return SAE_CLASS_REGISTRY[architecture]


def get_sae_training_class(architecture: str) -> "type[TrainingSAE]":
    return SAE_TRAINING_CLASS_REGISTRY[architecture]
