import os

# import pytest
# import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from transformer_lens import HookedTransformer

from sae_lens.cache_activations_runner import CacheActivationsRunner
from sae_lens.config import CacheActivationsRunnerConfig, LanguageModelSAERunnerConfig
from sae_lens.training.activations_store import ActivationsStore


# The way to run this with this command:
# poetry run py.test tests/unit/test_cache_activations_runner.py --profile-svg -s
def test_cache_activations_runner(tmp_path: Path):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # total_training_steps = 20_000
    context_size = 32
    print(f"n tokens per context: {context_size}")
    n_batches_in_buffer = 32
    print(f"n batches in buffer: {n_batches_in_buffer}")
    store_batch_size = 1
    print(f"store_batch_size: {store_batch_size}")
    n_buffers = 3
    print(f"n_buffers: {n_buffers}")

    tokens_in_buffer = n_batches_in_buffer * store_batch_size * context_size
    total_training_tokens = n_buffers * tokens_in_buffer
    print(f"Total Training Tokens: {total_training_tokens}")

    # better if we can look at the files (change tmp_path to a real path to look at the files)
    # tmp_path = os.path.join(os.path.dirname(__file__), "tmp")
    # tmp_path = Path("/Volumes/T7 Shield/activations/gelu_1l")
    # if os.path.exists(tmp_path):
    #     shutil.rmtree(tmp_path)

    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path),
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        d_in=512,
        dataset_path="NeelNanda/c4-tokenized-2b",
        context_size=context_size,  # Speed things up.
        is_dataset_tokenized=True,
        prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=4096,
        # Loss Function
        ## Reconstruction Coefficient.
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=store_batch_size,
        normalize_activations="none",
        #
        shuffle_every_n_buffers=2,
        n_shuffles_with_last_section=1,
        n_shuffles_in_entire_dir=1,
        n_shuffles_final=1,
        # Misc
        device=device,
        seed=42,
        dtype="float16",
    )

    # look at the next cell to see some instruction for what to do while this is running.
    CacheActivationsRunner(cfg).run()

    assert os.path.exists(tmp_path)

    # assert that there are n_buffer files in the directory.
    assert len(os.listdir(tmp_path)) == n_buffers

    for _, buffer_file in enumerate(os.listdir(tmp_path)):
        path_to_file = Path(tmp_path) / buffer_file
        with safe_open(path_to_file, framework="pt", device=str(device)) as f:  # type: ignore
            buffer = f.get_tensor("activations")
            assert buffer.shape == (
                tokens_in_buffer,
                1,
                cfg.d_in,
            )


def test_load_cached_activations():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # total_training_steps = 20_000
    context_size = 256
    print(f"n tokens per context: {context_size}")
    n_batches_in_buffer = 32
    print(f"n batches in buffer: {n_batches_in_buffer}")
    store_batch_size = 1
    print(f"store_batch_size: {store_batch_size}")
    n_buffers = 3
    print(f"n_buffers: {n_buffers}")

    tokens_in_buffer = n_batches_in_buffer * store_batch_size * context_size
    total_training_tokens = n_buffers * tokens_in_buffer
    print(f"Total Training Tokens: {total_training_tokens}")

    # better if we can look at the files
    cached_activations_fixture_path = os.path.join(
        os.path.dirname(__file__), "fixtures", "cached_activations"
    )

    cfg = LanguageModelSAERunnerConfig(
        cached_activations_path=cached_activations_fixture_path,
        use_cached_activations=True,
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        d_in=512,
        dataset_path="NeelNanda/c4-tokenized-2b",
        context_size=context_size,
        is_dataset_tokenized=True,
        prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=4096,
        # Loss Function
        ## Reconstruction Coefficient.
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=store_batch_size,
        normalize_activations="none",
        # shuffle_every_n_buffers=2,
        # n_shuffles_with_last_section=1,
        # n_shuffles_in_entire_dir=1,
        # n_shuffles_final=1,
        # Misc
        device=device,
        seed=42,
        dtype="float16",
    )

    model = HookedTransformer.from_pretrained(cfg.model_name)
    activations_store = ActivationsStore.from_config(model, cfg)

    for _ in range(n_buffers):
        buffer = activations_store.get_buffer(32)
        assert buffer.shape == (tokens_in_buffer, 1, cfg.d_in)

    # assert sparse_autoencoder_dictionary is not None
    # know whether or not this works by looking at the dashboard!
