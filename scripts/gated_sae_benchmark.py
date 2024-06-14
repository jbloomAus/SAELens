# type: ignore
# TDD: Let's make a config and work back from it.
import os

import torch

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

total_training_steps = 300_000
batch_size = 4096
total_training_tokens = total_training_steps * batch_size
print(f"Total Training Tokens: {total_training_tokens}")

# change these configs
model_name = "gelu-1l"

log_to_wandb = True
use_anthropic_method = True
if use_anthropic_method:
    print("Using Anthropic Method")
    apply_b_dec_to_input = False
    l1_warm_up_steps = total_training_steps // 20
    lr_warm_up_steps = 0
    decoder_heuristic_init = True
    normalize_sae_decoder = False
    adam_b1 = 0.0
    lr = 0.0003
else:
    print("Not using Anthropic Method")
    apply_b_dec_to_input = True
    l1_warm_up_steps = 0
    lr_warm_up_steps = total_training_steps // 10
    decoder_heuristic_init = False
    normalize_sae_decoder = True
    adam_b1 = 0.9
    lr = 0.003


l1_warm_up_steps = 0 if not use_anthropic_method else total_training_steps // 20
lr_warm_up_steps = total_training_steps // 10 if not use_anthropic_method else 0
decoder_heuristic_init = False if not use_anthropic_method else True

cfg = LanguageModelSAERunnerConfig(
    # Pick a tiny model to make this easier.
    model_name="gelu-1l",
    ## MLP Layer 0 ##
    hook_name="blocks.0.hook_mlp_out",
    hook_layer=0,
    d_in=512,
    dataset_path="NeelNanda/c4-tokenized-2b",
    streaming=False,
    context_size=128,
    is_dataset_tokenized=True,
    architecture="gated",
    # How big do we want our SAE to be?
    expansion_factor=64,
    use_cached_activations=False,
    training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
    train_batch_size_tokens=batch_size,
    l1_coefficient=0.5,
    lr_scheduler_name="constant",  # we set this independently of warmup and decay steps.
    lr_warm_up_steps=lr_warm_up_steps,
    l1_warm_up_steps=l1_warm_up_steps,
    lr_decay_steps=total_training_steps // 5,
    apply_b_dec_to_input=apply_b_dec_to_input,
    b_dec_init_method="zeros",
    normalize_sae_decoder=normalize_sae_decoder,
    decoder_heuristic_init=decoder_heuristic_init,
    init_encoder_as_decoder_transpose=True,
    lr=lr,
    adam_beta1=adam_b1,
    adam_beta2=0.999,
    n_batches_in_buffer=64,
    store_batch_size_prompts=16,
    normalize_activations="none",
    # Feature Store
    feature_sampling_window=1000,
    dead_feature_window=1000,
    dead_feature_threshold=1e-4,
    # WANDB
    log_to_wandb=log_to_wandb,  # always use wandb unless you are just testing code.
    wandb_project="gated_sae_benchmark",
    wandb_log_frequency=50,
    eval_every_n_wandb_logs=10,
    # Misc
    autocast=True,
    autocast_lm=True,
    device="mps",
    act_store_device="with_model",
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32",
)

# look at the next cell to see some instruction for what to do while this is running.
runner = SAETrainingRunner(cfg)

sae = runner.run()

print("=" * 50)
