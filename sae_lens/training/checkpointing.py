import os

import wandb
from safetensors.torch import save_file

from sae_lens.training.sparse_autoencoder import (
    SAE_CFG_PATH,
    SAE_WEIGHTS_PATH,
    SPARSITY_PATH,
)


def save_checkpoint(
    trainer,  # type: ignore
    checkpoint_name: int | str,
    wandb_aliases: list[str] | None = None,
) -> str:

    sae = trainer.sae
    checkpoint_path = f"{sae.cfg.checkpoint_path}/{checkpoint_name}"

    os.makedirs(checkpoint_path, exist_ok=True)

    path = f"{checkpoint_path}"
    os.makedirs(path, exist_ok=True)

    if sae.normalize_sae_decoder:
        sae.set_decoder_norm_to_unit_norm()
    sae.save_model(path)
    log_feature_sparsities = {"sparsity": trainer.log_feature_sparsity}

    log_feature_sparsity_path = f"{path}/{SPARSITY_PATH}"
    save_file(log_feature_sparsities, log_feature_sparsity_path)

    if sae.cfg.log_to_wandb and os.path.exists(log_feature_sparsity_path):
        model_artifact = wandb.Artifact(
            f"{sae.get_name()}",
            type="model",
            metadata=dict(sae.cfg.__dict__),
        )
        model_artifact.add_file(f"{path}/{SAE_WEIGHTS_PATH}")
        model_artifact.add_file(f"{path}/{SAE_CFG_PATH}")

        wandb.log_artifact(model_artifact, aliases=wandb_aliases)

        sparsity_artifact = wandb.Artifact(
            f"{sae.get_name()}_log_feature_sparsity",
            type="log_feature_sparsity",
            metadata=dict(sae.cfg.__dict__),
        )
        sparsity_artifact.add_file(log_feature_sparsity_path)
        wandb.log_artifact(sparsity_artifact)

    return checkpoint_path
