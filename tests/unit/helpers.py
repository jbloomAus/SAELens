import copy
from typing import Any

from transformer_lens import HookedTransformer

from sae_lens.config import LanguageModelSAERunnerConfig

TINYSTORIES_MODEL = "tiny-stories-1M"
TINYSTORIES_DATASET = "roneneldan/TinyStories"


def build_sae_cfg(**kwargs: Any) -> LanguageModelSAERunnerConfig:
    """
    Helper to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    mock_config = LanguageModelSAERunnerConfig(
        model_name=TINYSTORIES_MODEL,
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        hook_head_index=None,
        dataset_path=TINYSTORIES_DATASET,
        dataset_trust_remote_code=True,
        is_dataset_tokenized=False,
        use_cached_activations=False,
        d_in=64,
        expansion_factor=2,
        l1_coefficient=2e-3,
        lp_norm=1,
        lr=2e-4,
        train_batch_size_tokens=4,
        context_size=6,
        feature_sampling_window=50,
        dead_feature_threshold=1e-7,
        dead_feature_window=1000,
        n_batches_in_buffer=2,
        training_tokens=1_000_000,
        store_batch_size_prompts=4,
        log_to_wandb=False,
        wandb_project="test_project",
        wandb_entity="test_entity",
        wandb_log_frequency=10,
        device="cpu",
        seed=24,
        checkpoint_path="test/checkpoints",
        dtype="float32",
        prepend_bos=True,
    )

    for key, val in kwargs.items():
        setattr(mock_config, key, val)

    # Call the post-init method to set any derived attributes
    # useful for checking the correctness of the configuration
    # in the tests.
    mock_config.__post_init__()

    # reset checkpoint path (as we add an id to each each time)
    mock_config.checkpoint_path = (
        "test/checkpoints"
        if "checkpoint_path" not in kwargs
        else kwargs["checkpoint_path"]
    )

    return mock_config


MODEL_CACHE: dict[str, HookedTransformer] = {}


def load_model_cached(model_name: str) -> HookedTransformer:
    """
    helper to avoid unnecessarily loading the same model multiple times.
    NOTE: if the model gets modified in tests this will not work.
    """
    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = HookedTransformer.from_pretrained(
            model_name, device="cpu"
        )
    # we copy here to prevent sharing state across tests
    return copy.deepcopy(MODEL_CACHE[model_name])
