import torch
import os 
import sys 

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner

cfg = LanguageModelSAERunnerConfig(

    # Data Generating Function (Model + Training Distibuion)
    model_name = "gpt2-small",
    hook_point = "blocks.2.hook_resid_pre",
    hook_point_layer = 2,
    d_in = 768,
    dataset_path = "Skylion007/openwebtext",
    is_dataset_tokenized=False,
    
    # SAE Parameters
    expansion_factor = 64,
    b_dec_init_method = "geometric_median",
    
    # Training Parameters
    lr = 4e-4,
    l1_coefficient = 8e-5,
    lr_scheduler_name="constantwithwarmup",
    train_batch_size = 4096,
    context_size = 128,
    lr_warm_up_steps=5000,
    
    # Activation Store Parameters
    n_batches_in_buffer = 128,
    total_training_tokens = 1_000_000 * 300,
    store_batch_size = 32,
    
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_window = 1000,
    dead_feature_window=5000,
    dead_feature_threshold = 1e-6,
    
    # WANDB
    log_to_wandb = True,
    wandb_project= "mats_sae_training_gpt2",
    wandb_entity = None,
    wandb_log_frequency=100,
    
    # Misc
    device = "cuda",
    seed = 42,
    n_checkpoints = 10,
    checkpoint_path = "checkpoints",
    dtype = torch.float32,
    use_cached_activations = True,
)

sparse_autoencoder = language_model_sae_runner(cfg)