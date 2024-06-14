import os
import sys

import torch

sys.path.append("..")

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

total_training_steps = 20_000
batch_size = 4096
total_training_tokens = total_training_steps * batch_size
print(f"Total Training Tokens: {total_training_tokens}")

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training steps.
print(f"lr_decay_steps: {lr_decay_steps}")
l1_warmup_steps = total_training_steps // 20  # 5% of training steps.
print(f"l1_warmup_steps: {l1_warmup_steps}")
log_to_wandb = True

for l1_coefficient in [3, 4, 5, 6, 7]:
    for block in [1, 3, 5, 6]:
        cfg = LanguageModelSAERunnerConfig(
            # Pick a tiny model to make this easier.
            model_name="gpt2",
            ## MLP ##
            hook_name=f"blocks.{block}.hook_mlp_out",
            hook_layer=block,
            d_in=768,
            dataset_path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
            streaming=True,
            context_size=512,
            is_dataset_tokenized=True,
            prepend_bos=True,
            # How big do we want our SAE to be?
            expansion_factor=64,
            # Dataset / Activation Store
            use_cached_activations=False,
            training_tokens=total_training_tokens,
            train_batch_size_tokens=4096,
            # Loss Function
            ## Reconstruction Coefficient.
            mse_loss_normalization=None,  # MSE Loss Normalization is not mentioned (so we use stanrd MSE Loss). But not we take an average over the batch.
            ## Anthropic does not mention using an Lp norm other than L1.
            l1_coefficient=l1_coefficient,
            lp_norm=1.0,
            # Instead, they multiply the L1 loss contribution
            # from each feature of the activations by the decoder norm of the corresponding feature.
            scale_sparsity_penalty_by_decoder_norm=True,
            # Learning Rate
            lr_scheduler_name="constant",  # we set this independently of warmup and decay steps.
            l1_warm_up_steps=l1_warmup_steps,
            lr_warm_up_steps=lr_warm_up_steps,
            lr_decay_steps=lr_warm_up_steps,
            ## No ghost grad term.
            use_ghost_grads=False,
            # Initialization / Architecture
            apply_b_dec_to_input=False,
            # encoder bias zero's. (I'm not sure what it is by default now)
            # decoder bias zero's.
            b_dec_init_method="zeros",
            normalize_sae_decoder=False,
            decoder_heuristic_init=True,
            init_encoder_as_decoder_transpose=True,
            # Optimizer
            lr=1e-4,
            ## adam optimizer has no weight decay by default so worry about this.
            adam_beta1=0.9,
            adam_beta2=0.999,
            # Unsure if this is enough
            n_batches_in_buffer=64,
            store_batch_size_prompts=32,
            normalize_activations="none",
            # Feature Store
            feature_sampling_window=1000,
            dead_feature_window=1000,
            dead_feature_threshold=1e-4,
            # WANDB
            log_to_wandb=log_to_wandb,
            wandb_project="gpt-2-sweep-15may24-try-normalisation",
            wandb_log_frequency=50,
            eval_every_n_wandb_logs=10,
            # Misc
            device=device,
            seed=42,
            n_checkpoints=0,
            checkpoint_path="checkpoints",
            dtype="float32",
            eval_batch_size_prompts=2,
            n_eval_batches=40,
            autocast=True,
            compile_llm=True,
            compile_sae=True,
        )

        SAETrainingRunner(cfg).run()

        print("=" * 50)
