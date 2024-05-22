from typing import Any, cast

import einops
import torch
import wandb

from sae_lens.training.config import ToyModelSAERunnerConfig
from sae_lens.training.sparse_autoencoder import SparseAutoencoderBase
from sae_lens.training.toy_models import ReluOutputModel as ToyModel
from sae_lens.training.toy_models import ToyConfig
from sae_lens.training.train_sae_on_toy_model import train_toy_sae


def toy_model_sae_runner(cfg: ToyModelSAERunnerConfig):
    """
    A runner for training an SAE on a toy model.
    """
    # Toy Model Config
    toy_model_cfg = ToyConfig(
        n_features=cfg.n_features,
        n_hidden=cfg.n_hidden,
        n_correlated_pairs=cfg.n_correlated_pairs,
        n_anticorrelated_pairs=cfg.n_anticorrelated_pairs,
        feature_probability=cfg.feature_probability,
    )

    # Initialize Toy Model
    model = ToyModel(
        cfg=toy_model_cfg,
        device=torch.device(cfg.device),
    )

    # Train the Toy Model
    model.optimize(steps=cfg.model_training_steps)

    # Generate Training Data
    batch = model.generate_batch(cfg.total_training_tokens)
    hidden = einops.einsum(
        batch,
        model.W,
        "batch_size features, hidden features -> batch_size hidden",
    )

    sparse_autoencoder = SparseAutoencoderBase(
        cast(Any, cfg)  # TODO: the types are broken here
    )  # config has the hyperparameters for the SAE

    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, config=cast(Any, cfg))

    sparse_autoencoder = train_toy_sae(
        sparse_autoencoder,
        activation_store=hidden.detach().squeeze(),
        batch_size=cfg.train_batch_size,
        feature_sampling_window=cfg.feature_sampling_window,
        dead_feature_threshold=cfg.dead_feature_threshold,
        use_wandb=cfg.log_to_wandb,
        wandb_log_frequency=cfg.wandb_log_frequency,
    )

    if cfg.log_to_wandb:
        wandb.finish()

    return sparse_autoencoder
