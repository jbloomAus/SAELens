import torch
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

dataset_path = 'taufeeque/othellogpt'
model_name = 'othello-gpt'
device = "cuda" if torch.cuda.is_available() else "cpu"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner

import neel.utils as nutils

hyper_params = {
    "l1_coefficient": 0.0002,
    "exp_factor": 1.0,
    "num_million_tokens": 100,
}
hyper_params = nutils.arg_parse_update_cfg(hyper_params)

config = LanguageModelSAERunnerConfig(
    model_name=model_name,
    hook_point="blocks.6.hook_resid_pre",
    hook_point_layer=6,
    dataset_path=dataset_path,
    context_size=59,
    d_in=512,
    n_batches_in_buffer=32,
    # total_training_tokens=1*(1e6), # prev: 10*(1e6)
    total_training_tokens=hyper_params["num_million_tokens"]*(1e6), # prev: 10*(1e6)
    store_batch_size=32,
    device=device,
    seed=42,
    dtype=torch.float32,
    b_dec_init_method="geometric_median", # todo: geometric_median
    expansion_factor=hyper_params["exp_factor"], # todo: adjust
    l1_coefficient=hyper_params["l1_coefficient"], # prev: 0.001, 0.0001, 0.0002
    lr=0.00003, # prev: 0.0003
    lr_scheduler_name="constantwithwarmup",
    lr_warm_up_steps=5000,
    train_batch_size=4096,
    use_ghost_grads=True,
    feature_sampling_window=500,
    dead_feature_window=1e6,
    log_to_wandb=True,
    wandb_project="othello_gpt_sae",
    wandb_log_frequency=30,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    start_pos_offset=5, # exclude first seq position
    end_pos_offset=-5
)

sparse_autoencoder = language_model_sae_runner(config)
    # import time
    # rand_string = time.time()