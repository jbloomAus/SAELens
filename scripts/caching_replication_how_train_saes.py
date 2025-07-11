import os
import time

import torch

from sae_lens.cache_activations_runner import CacheActivationsRunner
from sae_lens.config import CacheActivationsRunnerConfig

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# change these configs
model_name = "gelu-1l"
model_batch_size = 16

dataset_path = "NeelNanda/c4-tokenized-2b"
total_training_tokens = 100_000

if device == "cuda":
    torch.cuda.empty_cache()
elif device == "mps":
    torch.mps.empty_cache()

cfg = CacheActivationsRunnerConfig(
    # Pick a tiny model to make this easier.
    model_name=model_name,
    dataset_path=dataset_path,
    ## MLP Layer 0 ##
    hook_name="blocks.0.hook_mlp_out",
    d_in=512,
    prepend_bos=True,
    training_tokens=total_training_tokens,
    model_batch_size=model_batch_size,
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
    f"{cfg.training_tokens / ((end_time - start_time)*10**6):.2f} Million Tokens / Second"
)
