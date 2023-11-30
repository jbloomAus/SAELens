import einops
import pytest
import torch

import wandb
from sae_training.SAE import SAE
from sae_training.toy_model_runner import SAEToyModelRunnerConfig, toy_model_sae_runner


def test_toy_model_sae_runner():
    
    cfg = SAEToyModelRunnerConfig(
        n_features = 5,
        n_hidden = 2,
        n_correlated_pairs = 0,
        n_anticorrelated_pairs = 0,
        feature_probability = 0.025,
        model_training_steps = 10_000,
        n_sae_training_tokens = 50_000,
        log_to_wandb = True,
    )

    trained_sae = toy_model_sae_runner(cfg)

    assert trained_sae is not None