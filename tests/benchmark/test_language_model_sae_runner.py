import torch

from sae_training.lm_runner import (
    LanguageModelSAERunnerConfig,
    language_model_sae_runner,
)


def test_language_model_sae_runner():
    
    
    cfg = LanguageModelSAERunnerConfig(

        # Data Generating Function (Model + Training Distibuion)
        model_name = "gelu-2l",
        hook_point = "blocks.0.hook_mlp_out",
        hook_point_layer = 0,
        d_in = 512,
        dataset_path = "NeelNanda/c4-tokenized-2b",
        
        # SAE Parameters
        expansion_factor = 8, # determines the dimension of the SAE.
        
        # Training Parameters
        lr = 1e-4,
        l1_coefficient = 3e-3,
        train_batch_size = 4096,
        context_size = 128,
        
        # Activation Store Parameters
        n_batches_in_buffer = 2,
        total_training_tokens = 20_000_000,
        store_batch_size = 32,
        
        # WANDB
        log_to_wandb = True,
        wandb_project= "mats_sae_training_language_models",
        wandb_entity = None,
        
        # Misc
        device = "cuda",
        seed = 42,
        checkpoint_path = "checkpoints",
        dtype = torch.float32,
        )

    trained_sae = language_model_sae_runner(cfg)

    assert trained_sae is not None



