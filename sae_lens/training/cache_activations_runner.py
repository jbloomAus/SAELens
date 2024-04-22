import math
import os

import torch
from tqdm import tqdm

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.config import CacheActivationsRunnerConfig
from sae_lens.training.load_model import load_model
from sae_lens.training.utils import shuffle_activations_pairwise


def cache_activations_runner(cfg: CacheActivationsRunnerConfig):
    model = load_model(
        model_class_name=cfg.model_class_name,
        model_name=cfg.model_name,
        device=cfg.device,
    )
    activations_store = ActivationsStore.from_config(
        model,
        cfg,
    )

    # if the activations directory exists and has files in it, raise an exception
    assert activations_store.cached_activations_path is not None
    if os.path.exists(activations_store.cached_activations_path):
        if len(os.listdir(activations_store.cached_activations_path)) > 0:
            raise Exception(
                f"Activations directory ({activations_store.cached_activations_path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
            )
    else:
        os.makedirs(activations_store.cached_activations_path)

    print(f"Started caching {cfg.training_tokens} activations")
    tokens_per_buffer = (
        cfg.store_batch_size * cfg.context_size * cfg.n_batches_in_buffer
    )
    n_buffers = math.ceil(cfg.training_tokens / tokens_per_buffer)
    # for i in tqdm(range(n_buffers), desc="Caching activations"):
    for i in range(n_buffers):
        buffer = activations_store.get_buffer(cfg.n_batches_in_buffer)
        torch.save(buffer, f"{activations_store.cached_activations_path}/{i}.pt")
        del buffer

        if i % cfg.shuffle_every_n_buffers == 0 and i > 0:
            # Shuffle the buffers on disk

            # Do random pairwise shuffling between the last shuffle_every_n_buffers buffers
            for _ in range(cfg.n_shuffles_with_last_section):
                shuffle_activations_pairwise(
                    activations_store.cached_activations_path,
                    buffer_idx_range=(i - cfg.shuffle_every_n_buffers, i),
                )

            # Do more random pairwise shuffling between all the buffers
            for _ in range(cfg.n_shuffles_in_entire_dir):
                shuffle_activations_pairwise(
                    activations_store.cached_activations_path,
                    buffer_idx_range=(0, i),
                )

    # More final shuffling (mostly in case we didn't end on an i divisible by shuffle_every_n_buffers)
    if n_buffers > 1:
        for _ in tqdm(range(cfg.n_shuffles_final), desc="Final shuffling"):
            shuffle_activations_pairwise(
                activations_store.cached_activations_path,
                buffer_idx_range=(0, n_buffers),
            )
