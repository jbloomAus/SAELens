# install from https://github.com/Phylliida/MambaLens
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# run this as python3 tutorials/mamba_train_example.py
# i.e. from the root directory
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    model_name="state-spaces/mamba-370m",
    model_class_name="HookedMamba",
    hook_name="blocks.39.hook_ssm_input",
    hook_layer=39,
    hook_eval="blocks.39.hook_ssm_output",  # we compare this when replace hook_point activations with autoencode.decode(autoencoder.encode( hook_point activations))
    d_in=2048,
    dataset_path="NeelNanda/openwebtext-tokenized-9b",
    is_dataset_tokenized=True,
    # SAE Parameters
    expansion_factor=64,
    b_dec_init_method="geometric_median",
    # Training Parameters
    lr=0.0004,
    l1_coefficient=0.00006 * 0.2,
    lr_scheduler_name="cosineannealingwarmrestarts",
    train_batch_size_tokens=4096,
    context_size=128,
    lr_warm_up_steps=5000,
    # Activation Store Parameters
    n_batches_in_buffer=128,
    training_tokens=1_000_000 * 300,
    store_batch_size_prompts=32,
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_window=1000,
    dead_feature_window=5000,
    dead_feature_threshold=1e-6,
    # WANDB
    log_to_wandb=True,
    wandb_project="sae_training_mamba",
    wandb_entity=None,
    wandb_log_frequency=100,
    # Misc
    device="cuda",
    seed=42,
    checkpoint_path="checkpoints",
    dtype="float32",
    model_kwargs={
        "fast_ssm": True,
        "fast_conv": True,
    },
)

SAETrainingRunner(cfg).run()
