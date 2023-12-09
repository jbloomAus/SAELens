import torch

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner


def test_language_model_sae_runner_mlp_out():
    
    
    cfg = LanguageModelSAERunnerConfig(

        # Data Generating Function (Model + Training Distibuion)
        model_name = "gelu-2l",
        hook_point = "blocks.0.hook_mlp_out",
        hook_point_layer = 0,
        d_in = 512,
        dataset_path = "NeelNanda/c4-tokenized-2b",
        is_dataset_tokenized=True,
        
        # SAE Parameters
        expansion_factor = 64,
        
        # Training Parameters
        lr = 1e-4,
         l1_coefficient = 3e-4,
        train_batch_size = 4096,
        context_size = 128,
        
        # Activation Store Parameters
        n_batches_in_buffer = 24,
        total_training_tokens = 5_000_00 * 100,
        store_batch_size = 32,
        
        # Resampling protocol
        feature_sampling_method = 'l2',
        feature_sampling_window = 2500,
        feature_reinit_scale = 0.2,
        dead_feature_window=1250,
        dead_feature_threshold = 1e-8,
        
        # WANDB
        log_to_wandb = True,
        wandb_project= "mats_sae_training_language_models",
        wandb_entity = None,
        
        # Misc
        device = "mps",
        seed = 42,
        n_checkpoints = 5,
        checkpoint_path = "checkpoints",
        dtype = torch.float32,
        )

    sparse_autoencoder = language_model_sae_runner(cfg)

    assert sparse_autoencoder is not None



def test_language_model_sae_runner_resid_pre():
     
    cfg = LanguageModelSAERunnerConfig(

        # Data Generating Function (Model + Training Distibuion)
        model_name = "gelu-2l",
        hook_point = "blocks.0.hook_resid_mid",
        hook_point_layer = 0,
        d_in = 512,
        dataset_path = "NeelNanda/c4-tokenized-2b",
        is_dataset_tokenized=True,
        
        # SAE Parameters
        expansion_factor = 64, 
        
        # Training Parameters
        lr = 1e-4,
        l1_coefficient = 1e-4,
        train_batch_size = 4096,
        context_size = 128,
        
        # Activation Store Parameters
        n_batches_in_buffer = 24,
        total_training_tokens = 5_000_00 * 100, 
        store_batch_size = 32,
        
        # Resampling protocol
        feature_sampling_method = 'l2',
        feature_sampling_window = 1000, 
        feature_reinit_scale = 0.2,
        dead_feature_threshold = 1e-8,
        
        # WANDB
        log_to_wandb = True,
        wandb_project= "mats_sae_training_language_models",
        wandb_entity = None,
        
        # Misc
        device = "cuda",
        seed = 42,
        n_checkpoints = 5,
        checkpoint_path = "checkpoints",
        dtype = torch.float32,
        )

    trained_sae = language_model_sae_runner(cfg)

    assert trained_sae is not None


def test_language_model_sae_runner_not_tokenized():
    
    cfg = LanguageModelSAERunnerConfig(

        # Data Generating Function (Model + Training Distibuion)
        model_name = "gelu-2l",
        hook_point = "blocks.1.hook_mlp_out",
        hook_point_layer = 0,
        d_in = 512,
        dataset_path = "roneneldan/TinyStories",
        is_dataset_tokenized=False,
        
        # SAE Parameters
        expansion_factor = 64, # determines the dimension of the SAE.
        
        # Training Parameters
        lr = 1e-4,
        l1_coefficient = 1e-4,
        train_batch_size = 4096,
        context_size = 128,
        
        # Activation Store Parameters
        n_batches_in_buffer = 8,
        total_training_tokens = 25_000_00 * 60,
        store_batch_size = 32,
        
        # Resampling protocol
        feature_sampling_window = 1000,
        feature_reinit_scale = 0.2,
        dead_feature_threshold = 1e-8,
        
        # WANDB
        log_to_wandb = True,
        wandb_project= "mats_sae_training_language_models",
        wandb_entity = None,
        
        # Misc
        device = "mps",
        seed = 42,
        checkpoint_path = "checkpoints",
        dtype = torch.float32,
        )

    trained_sae = language_model_sae_runner(cfg)

    assert trained_sae is not None
