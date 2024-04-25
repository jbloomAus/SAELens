import torch

from sae_lens import LanguageModelSAERunnerConfig


def get_lm_sae_runner_config():
    return LanguageModelSAERunnerConfig(
        # Data Generating Function (Model + Training Distibuion)
        model_name="gpt2-small",
        sae_class_name="GatedSparseAutoencoder",
        hook_point="blocks.2.hook_resid_pre",
        hook_point_layer=2,
        d_in=768,
        dataset_path="Skylion007/openwebtext",
        is_dataset_tokenized=False,
        # SAE Parameters
        expansion_factor=64,
        b_dec_init_method="geometric_median",
        # Training Parameters
        lr=0.0004,
        l1_coefficient=0.00008,
        lr_scheduler_name="constant",
        train_batch_size=1024,
        context_size=128,
        lr_warm_up_steps=5000,
        # Activation Store Parameters
        n_batches_in_buffer=128,
        training_tokens=1_000_000 * 300,
        store_batch_size=32,
        # Dead Neurons and Sparsity
        use_ghost_grads=True,
        feature_sampling_window=1000,
        dead_feature_window=5000,
        dead_feature_threshold=1e-6,
        # WANDB
        log_to_wandb=True,
        wandb_project="gpt2",
        wandb_entity=None,
        wandb_log_frequency=100,
        # Misc
        device="cuda",
        seed=42,
        n_checkpoints=10,
        checkpoint_path="checkpoints",
        dtype=torch.float32,
    )
