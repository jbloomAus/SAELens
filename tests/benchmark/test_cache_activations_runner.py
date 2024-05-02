import os

import pytest
import torch
from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.cache_activations_runner import cache_activations_runner
from sae_lens.training.config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
)


# The way to run this with this command:
# poetry run py.test tests/benchmark/test_cache_activations_runner.py --profile-svg -s
def test_cache_activations_runner(tmp_path):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # total_training_steps = 20_000
    total_training_tokens = 256 * 32 * 4
    print(f"Total Training Tokens: {total_training_tokens}")

    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=tmp_path,
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        ## MLP Layer 0 ##
        hook_point="blocks.0.hook_mlp_out",
        hook_point_layer=0,
        d_in=512,
        dataset_path="NeelNanda/c4-tokenized-2b",
        context_size=1024,
        is_dataset_tokenized=True,
        prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size=4096,
        # Loss Function
        ## Reconstruction Coefficient.
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=32,
        store_batch_size=1,
        normalize_activations=False,
        #
        shuffle_every_n_buffers=2,
        n_shuffles_with_last_section=1,
        n_shuffles_in_entire_dir=1,
        n_shuffles_final=1,
        # Misc
        device=device,
        seed=42,
        dtype=torch.float32,
    )

    # look at the next cell to see some instruction for what to do while this is running.
    cache_activations_runner(cfg)

    assert os.path.exists(tmp_path)
    path_to_first_file = tmp_path / os.listdir(tmp_path)[0]
    assert os.path.exists(path_to_first_file)
    assert torch.load(path_to_first_file).shape == (
        total_training_tokens,
        1,
        cfg.d_in,
    )

    cfg = LanguageModelSAERunnerConfig(
        cached_activations_path=tmp_path,
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        ## MLP Layer 0 ##
        hook_point="blocks.0.hook_mlp_out",
        hook_point_layer=0,
        d_in=512,
        dataset_path="NeelNanda/c4-tokenized-2b",
        context_size=1024,
        is_dataset_tokenized=True,
        prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size=4096,
        # Loss Function
        ## Reconstruction Coefficient.
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=32,
        store_batch_size=1,
        normalize_activations=False,
        #
        # shuffle_every_n_buffers=2,
        # n_shuffles_with_last_section=1,
        # n_shuffles_in_entire_dir=1,
        # n_shuffles_final=1,
        # Misc
        device=device,
        seed=42,
        dtype=torch.float32,
    )

    model = HookedTransformer.from_pretrained(cfg.model_name)
    activations_store = ActivationsStore.from_config(model, cfg)
    buffer = activations_store.get_buffer(32)
    assert buffer.shape == (total_training_tokens, 1, cfg.d_in)

    # assert sparse_autoencoder_dictionary is not None
    # know whether or not this works by looking at the dashboard!
