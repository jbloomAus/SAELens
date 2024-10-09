import os
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import trange

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
        # Misc
        device=device,
        seed=42,
        dtype="float32",
    )

    # look at the next cell to see some instruction for what to do while this is running.
    CacheActivationsRunner(cfg).run()


def test_hf_dataset_save_vs_safetensors(tmp_path: Path):
    niters = 10

    ###

    d_in = 512
    context_size = 32
    n_batches_in_buffer = 32
    batch_size = 8
    num_buffers = 4 * niters
    num_tokens = batch_size * context_size * n_batches_in_buffer * num_buffers

    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path),
        d_in=d_in,
        context_size=context_size,
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        dataset_path="NeelNanda/c4-tokenized-2b",
        training_tokens=num_tokens,
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=batch_size,
        normalize_activations="none",
        device="cpu",
        seed=42,
        dtype="float32",
    )
    runner = CacheActivationsRunner(cfg)
    store = runner.activations_store

    ###

    safetensors_path = tmp_path / "saftensors"
    hf_path = tmp_path / "hf"
    safetensors_path.mkdir()
    hf_path.mkdir()

    print("Warmup")

    for i in trange(niters // 2, leave=False):
        buffer = store.get_buffer(n_batches_in_buffer)

    start_time = time.perf_counter()
    for i in trange(niters, leave=False):
        buffer = store.get_buffer(n_batches_in_buffer)
    end_time = time.perf_counter()

    print(f"No saving took: {end_time - start_time:.4f}")

    start_time = time.perf_counter()
    for i in trange(niters, leave=False):
        buffer = store.get_buffer(n_batches_in_buffer)
        shard = runner._create_shard(buffer)
        shard.save_to_disk(hf_path / str(i), num_shards=1)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"HF Dataset took: {elapsed_time:.4f}", flush=True)
    hf_size = sum(f.stat().st_size for f in hf_path.glob("**/*") if f.is_file())
    print(f"HF Dataset size: {hf_size / (1024 * 1024):.2f} MB")

    start_time = time.perf_counter()
    for i in trange(niters, leave=False):
        buffer = store.get_buffer(n_batches_in_buffer)
        save_file({"activations": buffer}, safetensors_path / f"{i}.safetensors")
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Safetensors took: {elapsed_time:.4f}")
    safetensors_size = sum(
        f.stat().st_size for f in safetensors_path.glob("**/*") if f.is_file()
    )
    print(f"Safetensors size: {safetensors_size / (1024 * 1024):.2f} MB")
