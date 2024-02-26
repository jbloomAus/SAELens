import os
import sys

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae_training.cache_activations_runner import cache_activations_runner
from sae_training.config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
)
from sae_training.lm_runner import language_model_sae_runner

cfg = CacheActivationsRunnerConfig(

    # Data Generating Function (Model + Training Distibuion)
    model_name = "gpt2-small",
    hook_point = f"blocks.{3}.hook_resid_pre",
    hook_point_layer = 3,
    d_in = 768,
    dataset_path = "Skylion007/openwebtext",
    is_dataset_tokenized=True,
    cached_activations_path="activations/",
    
    # Activation Store Parameters
    n_batches_in_buffer = 16,
    total_training_tokens = 300_000_000, 
    store_batch_size = 64,

    # Activation caching shuffle parameters
    n_shuffles_final = 16,
    
    # Misc
    device = "cuda",
    seed = 42,
    dtype = torch.bfloat16,
    )

cache_activations_runner(cfg)