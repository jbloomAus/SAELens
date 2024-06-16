import os
import shutil

import torch

from sae_lens.cache_activations_runner import CacheActivationsRunner
from sae_lens.config import CacheActivationsRunnerConfig

os.environ["WANDB_MODE"] = "offline"  # turn this off if you want to see the output


# The way to run this with this command:
# poetry run py.test tests/benchmark/test_cache_activations_runner.py --profile-svg -s
def test_cache_activations_runner():

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    total_training_steps = 500
    batch_size = 4096
    total_training_tokens = total_training_steps * batch_size
    print(f"Total Training Tokens: {total_training_tokens}")

    new_cached_activations_path = (
        os.path.dirname(os.path.realpath(__file__))
        + "/fixtures/test_activations/gelu_1l"
    )

    # If the directory exists, delete it.
    if os.path.exists(new_cached_activations_path):
        shutil.rmtree(new_cached_activations_path)

    torch.mps.empty_cache()

    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=new_cached_activations_path,
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        # model_name="gpt2-xl",
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        d_in=512,
        # d_in=1600,
        dataset_path="NeelNanda/c4-tokenized-2b",
        streaming=False,
        context_size=1024,
        is_dataset_tokenized=True,
        prepend_bos=True,
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=4096,
        # buffer details
        n_batches_in_buffer=32,
        store_batch_size_prompts=16,
        normalize_activations="none",
        #
        shuffle_every_n_buffers=8,
        n_shuffles_with_last_section=1,
        n_shuffles_in_entire_dir=1,
        n_shuffles_final=1,
        # Misc
        device=device,
        seed=42,
        dtype="float32",
    )

    # look at the next cell to see some instruction for what to do while this is running.
    CacheActivationsRunner(cfg).run()
