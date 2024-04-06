from typing import Any

import torch

from sae_training.config import LanguageModelSAERunnerConfig

TINYSTORIES_MODEL = "tiny-stories-1M"
TINYSTORIES_DATASET = "roneneldan/TinyStories"


def build_sae_cfg(**kwargs: Any) -> LanguageModelSAERunnerConfig:
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    mock_config = LanguageModelSAERunnerConfig(
        model_name=TINYSTORIES_MODEL,
        hook_point="blocks.0.hook_mlp_out",
        hook_point_layer=0,
        hook_point_head_index=None,
        dataset_path=TINYSTORIES_DATASET,
        is_dataset_tokenized=False,
        use_cached_activations=False,
        d_in=64,
        expansion_factor=2,
        l1_coefficient=2e-3,
        lp_norm=1,
        lr=2e-4,
        train_batch_size=2048,
        context_size=64,
        feature_sampling_window=50,
        dead_feature_threshold=1e-7,
        n_batches_in_buffer=10,
        total_training_tokens=1_000_000,
        store_batch_size=32,
        log_to_wandb=False,
        wandb_project="test_project",
        wandb_entity="test_entity",
        wandb_log_frequency=10,
        device=torch.device("cpu"),
        seed=24,
        checkpoint_path="test/checkpoints",
        dtype=torch.float32,
        prepend_bos=True,
    )

    for key, val in kwargs.items():
        setattr(mock_config, key, val)

    return mock_config
