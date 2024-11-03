import os
import shutil
import time

import torch

from sae_lens.cache_activations_runner import CacheActivationsRunner

# from pathlib import Path
from sae_lens.config import CacheActivationsRunnerConfig

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

# change these configs
model_name = "gelu-1l"
dataset_path = "NeelNanda/c4-tokenized-2b"
new_cached_activations_path = (
    f"./cached_activations/{model_name}/{dataset_path}/{total_training_steps}"
)

# check how much data is in the directory
if os.path.exists(new_cached_activations_path):
    print("Directory exists. Checking how much data is in the directory.")
    total_files = sum(
        os.path.getsize(os.path.join(new_cached_activations_path, f))
        for f in os.listdir(new_cached_activations_path)
        if os.path.isfile(os.path.join(new_cached_activations_path, f))
    )
    print(f"Total size of directory: {total_files / 1e9:.2f} GB")

# If the directory exists, delete it.
if input("Delete the directory? (y/n): ") == "y" and os.path.exists(
    new_cached_activations_path
):
    if os.path.exists(new_cached_activations_path):
        shutil.rmtree(new_cached_activations_path)

if device == "cuda":
    torch.cuda.empty_cache()
elif device == "mps":
    torch.mps.empty_cache()

cfg = CacheActivationsRunnerConfig(
    new_cached_activations_path=new_cached_activations_path,
    # Pick a tiny model to make this easier.
    model_name=model_name,
    ## MLP Layer 0 ##
    hook_name="blocks.0.hook_mlp_out",
    hook_layer=0,
    d_in=512,
    dataset_path=dataset_path,
    context_size=1024,
    is_dataset_tokenized=True,
    prepend_bos=True,
    training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
    train_batch_size_tokens=4096,
    # buffer details
    n_batches_in_buffer=4,
    store_batch_size_prompts=128,
    normalize_activations="none",
    #
    # Misc
    device=device,
    seed=42,
    dtype="float16",
)
# look at the next cell to see some instruction for what to do while this is running.

start_time = time.time()


runner = CacheActivationsRunner(cfg)

print("-" * 50)
print(runner.__str__())
print("-" * 50)
runner.run()


end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
print(
    f"{total_training_tokens / ((end_time - start_time)*10**6):.2f} Million Tokens / Second"
)
