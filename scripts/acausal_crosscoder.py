import os
import sys

import torch

sys.path.append("..")

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.training.training_crosscoder_sae import (
    TrainingCrosscoderSAE,
    TrainingCrosscoderSAEConfig
)
from sae_lens.sae_training_runner import SAETrainingRunner

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# total_training_steps = 200_000
# total_training_steps = 60_000
total_training_steps = 10_000
batch_size = 4092
# batch_size = 256
total_training_tokens = total_training_steps * batch_size
print(f"Total Training Tokens: {total_training_tokens}")

hook_name_template = "blocks.{layer}.hook_mlp_out"
layers = list(range(2))

model_name = "gpt2-small"
dataset_path = "apollo-research/SkyLion007-openwebtext-tokenizer-gpt2"
new_cached_activations_path = (
    f"./cached_activations/{model_name}/{dataset_path}/{total_training_steps}"
)

lr_warm_up_steps = total_training_steps // 40
print(f"lr_warm_up_steps: {lr_warm_up_steps}")
lr_decay_steps = total_training_steps // 5  # 20% of training steps.
print(f"lr_decay_steps: {lr_decay_steps}")
l1_warmup_steps = total_training_steps // 20
print(f"l1_warmup_steps: {l1_warmup_steps}")
log_to_wandb = True
if not log_to_wandb:
    print("NOT LOGGING TO WANDB")

d_in = 768
expansion_factor = 32
d_sae = d_in * expansion_factor
learning_rate = 2e-5
l1_coefficient = 1
hook_name = hook_name_template.format(
    layer=f"{min(layers)}_through_{max(layers)}"
)
hook_names = [hook_name_template.format(layer=layer) for layer in layers]

cfg = LanguageModelSAERunnerConfig(
    model_name=model_name,
    hook_name=hook_name,
    hook_names=hook_names,
    hook_layer=max(layers),
    d_in=d_in,
    dataset_path=dataset_path,
    streaming=True,
    context_size=512,
    is_dataset_tokenized=True,
    prepend_bos=True,
    expansion_factor=expansion_factor,
    use_cached_activations=False,
    training_tokens=total_training_tokens,
    train_batch_size_tokens=batch_size,
    # Loss Function
    mse_loss_normalization=None,
    l1_coefficient=l1_coefficient,
    lp_norm=1.0,
    scale_sparsity_penalty_by_decoder_norm=True,
    # TODO(mkbehr): plumb this through config
    # sparsity_penalty_decoder_norm_lp_norm=1.0,
    # Learning Rate
    lr_scheduler_name="constant",  # we set this independently of warmup and decay steps.
    l1_warm_up_steps=l1_warmup_steps,
    lr_warm_up_steps=lr_warm_up_steps,
    lr_decay_steps=lr_warm_up_steps,
    use_ghost_grads=False,
    # Initialization / Architecture
    apply_b_dec_to_input=False,
    b_dec_init_method="zeros",
    normalize_sae_decoder=False,
    decoder_heuristic_init=True,
    decoder_heuristic_init_norm=0.1,
    init_encoder_as_decoder_transpose=True,
    # Optimizer
    lr=learning_rate,
    ## adam optimizer has no weight decay by default so worry about this.
    adam_beta1=0.9,
    adam_beta2=0.999,
    # Buffer details won't matter in we cache / shuffle our activations ahead of time.
    n_batches_in_buffer=32,
    store_batch_size_prompts=16,
    normalize_activations="expected_average_only_in",
    # Feature Store
    feature_sampling_window=1000,
    dead_feature_window=1000,
    dead_feature_threshold=1e-4,
    # WANDB
    log_to_wandb=log_to_wandb,  # always use wandb unless you are just testing code.
    wandb_project="crosscoder-acausal-gpt2-small",
    wandb_log_frequency=50,
    eval_every_n_wandb_logs=10,
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype="float32",
)

sae = SAETrainingRunner(
    cfg,
    override_sae = TrainingCrosscoderSAE(
        TrainingCrosscoderSAEConfig.from_sae_runner_config(cfg),
        use_error_term=True,
    )).run()

print("=" * 50)

