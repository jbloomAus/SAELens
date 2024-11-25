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

    activations_save_path = (
        os.path.dirname(os.path.realpath(__file__))
        + "/fixtures/test_activations/gelu_1l"
    )

    # If the directory exists, delete it.
    if os.path.exists(activations_save_path):
        shutil.rmtree(activations_save_path)

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    cfg = CacheActivationsRunnerConfig(
        activation_save_path=activations_save_path,
        total_training_tokens=16_000,
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        model_batch_size=16,
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        final_hook_layer=0,
        d_in=512,
        ## Dataset ##
        hf_dataset_path="NeelNanda/c4-tokenized-2b",
        context_size=1024,
        ## Misc ##
        device=device,
        seed=42,
        dtype="float32",
    )

    start_time = time.perf_counter()
    CacheActivationsRunner(cfg).run()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Caching activations took: {elapsed_time:.4f}")


def test_hf_dataset_save_vs_safetensors(tmp_path: Path):
    context_size = 32

    dataset_num_rows = 10_000
    total_training_tokens = dataset_num_rows * context_size
    model_batch_size = 8

    ###

    d_in = 512
    dtype = "float32"
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    safetensors_path = tmp_path / "saftensors"
    hf_path = tmp_path / "hf"
    safetensors_path.mkdir()
    hf_path.mkdir()

    cfg = CacheActivationsRunnerConfig(
        activation_save_path=str(hf_path),
        hf_dataset_path="NeelNanda/c4-tokenized-2b",
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        final_hook_layer=0,
        d_in=d_in,
        context_size=context_size,
        total_training_tokens=total_training_tokens,
        model_batch_size=model_batch_size,
        prepend_bos=False,
        shuffle=False,
        device=device,
        seed=42,
        dtype=dtype,
    )
    runner = CacheActivationsRunner(cfg)
    store = runner.activations_store

    print("Warmup")

    for i in trange(10 // 2, leave=False):
        buffer = store.get_buffer(cfg.batches_in_buffer)

    start_time = time.perf_counter()
    for i in trange(cfg.n_buffers, leave=False):
        buffer = store.get_buffer(cfg.batches_in_buffer)
    end_time = time.perf_counter()

    print(f"No saving took: {end_time - start_time:.4f}")

    start_time = time.perf_counter()
    runner.run()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f"HF Dataset took: {elapsed_time:.4f}", flush=True)
    hf_size = sum(f.stat().st_size for f in hf_path.glob("**/*") if f.is_file())
    print(f"HF Dataset size: {hf_size / (1024 * 1024):.2f} MB")

    start_time = time.perf_counter()
    for i in trange(cfg.n_buffers, leave=False):
        buffer = store.get_buffer(cfg.batches_in_buffer)
        save_file({"activations": buffer}, safetensors_path / f"{i}.safetensors")
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Safetensors took: {elapsed_time:.4f}")
    safetensors_size = sum(
        f.stat().st_size for f in safetensors_path.glob("**/*") if f.is_file()
    )
    print(f"Safetensors size: {safetensors_size / (1024 * 1024):.2f} MB")
