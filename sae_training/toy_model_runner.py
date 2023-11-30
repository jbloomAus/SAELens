from dataclasses import dataclass

import einops
import torch
from transformer_lens import HookedTransformer

import wandb
from sae_training.SAE import SAE
from sae_training.toy_models import Config as ToyConfig
from sae_training.toy_models import Model as ToyModel
from sae_training.train_sae_on_toy_model import train_toy_sae


@dataclass
class SAEToyModelRunnerConfig:
    # ReLu Model Parameters
    n_features: int = 5
    n_hidden: int = 2
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0
    feature_probability: float = 0.025
    # Relu Model Training Parameters
    model_training_steps: int = 10_000
    # SAE Parameters
    d_sae: int = 5
    # Training Parameters
    n_sae_training_tokens: int = 25_000
    l1_coefficient: float = 1e-3
    lr: float = 3e-4
    train_batch_size: int = 1024 # Shouldn't be as big as the batch size for language models
    train_epochs: int = 10
    feature_sampling_window: int = 100
    feature_reinit_scale: float = 0.2
    dead_feature_threshold: float = 1e-8
    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "mats_sae_training_toy_model"
    wandb_entity: str = None
    wandb_log_frequency: int = 50
    # Misc
    device: str = "cpu"
    seed: int = 42
    checkpoint_path: str = "checkpoints"
    dtype: torch.dtype = (
        torch.float32
    )  # TODO: Make this a string (have a dictionary to map)

    def __post_init__(self):
        self.d_in = self.n_hidden  # hidden for the ReLu model is the input for the SAE

def toy_model_sae_runner(cfg):
    '''
    A runner for training an SAE on a toy model.
    '''
    # Toy Model Config
    toy_model_cfg = ToyConfig(
        n_instances=1,  # Not set up to train > 1 SAE so shouldn't do > 1 model.
        n_features=cfg.n_features,
        n_hidden=cfg.n_hidden,
        n_correlated_pairs=cfg.n_correlated_pairs,
        n_anticorrelated_pairs=cfg.n_anticorrelated_pairs,
    )

    # Initialize Toy Model
    model = ToyModel(
        cfg=toy_model_cfg,
        device="cpu",
        feature_probability=cfg.feature_probability,
    )

    # Train the Toy Model
    model.optimize(steps=cfg.model_training_steps)

    # Generate Training Data
    batch = model.generate_batch(cfg.n_sae_training_tokens)
    hidden = einops.einsum(
        batch,
        model.W,
        "batch_size instances features, instances hidden features -> batch_size instances hidden",
    )

    sae = SAE(cfg)  # config has the hyperparameters for the SAE

    if cfg.log_to_wandb:
        wandb.init(project="sae-training-test", config=cfg)

    sae = train_toy_sae(
        model, # need model so we can do evals for neuron resampling
        sae,
        hidden.detach().squeeze(),
        use_wandb=cfg.log_to_wandb,
        batch_size=cfg.train_batch_size,
        n_epochs=cfg.train_epochs,
        feature_sampling_window=cfg.feature_sampling_window,
        feature_reinit_scale=cfg.feature_reinit_scale,
        dead_feature_threshold=cfg.dead_feature_threshold,
        wandb_log_frequency=cfg.wandb_log_frequency,
    )

    if cfg.log_to_wandb:
        wandb.finish()

    return sae
