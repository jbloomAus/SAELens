from typing import TYPE_CHECKING, Any

# avoid circular imports
if TYPE_CHECKING:
    from sae_lens.saes.sae import SAE, SAEConfig, TrainingSAE, TrainingSAEConfig

SAE_CLASS_REGISTRY: dict[str, tuple["type[SAE[Any]]", "type[SAEConfig]"]] = {}
SAE_TRAINING_CLASS_REGISTRY: dict[
    str, tuple["type[TrainingSAE[Any]]", "type[TrainingSAEConfig]"]
] = {}


def register_sae_class(
    architecture: str,
    sae_class: "type[SAE[Any]]",
    sae_config_class: "type[SAEConfig]",
) -> None:
    if architecture in SAE_CLASS_REGISTRY:
        raise ValueError(
            f"SAE class for architecture {architecture} already registered."
        )
    SAE_CLASS_REGISTRY[architecture] = (sae_class, sae_config_class)


def register_sae_training_class(
    architecture: str,
    sae_training_class: "type[TrainingSAE[Any]]",
    sae_training_config_class: "type[TrainingSAEConfig]",
) -> None:
    if architecture in SAE_TRAINING_CLASS_REGISTRY:
        raise ValueError(
            f"SAE training class for architecture {architecture} already registered."
        )
    SAE_TRAINING_CLASS_REGISTRY[architecture] = (
        sae_training_class,
        sae_training_config_class,
    )


def get_sae_class(
    architecture: str,
) -> tuple["type[SAE[Any]]", "type[SAEConfig]"]:
    return SAE_CLASS_REGISTRY[architecture]


def get_sae_training_class(
    architecture: str,
) -> tuple["type[TrainingSAE[Any]]", "type[TrainingSAEConfig]"]:
    return SAE_TRAINING_CLASS_REGISTRY[architecture]
