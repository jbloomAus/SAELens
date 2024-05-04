import math
import os
from typing import Tuple

import torch
from tqdm import tqdm

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.config import DTYPE_MAP, CacheActivationsRunnerConfig
from sae_lens.training.load_model import load_model


class CacheActivationsRunner:

    def __init__(self, cfg: CacheActivationsRunnerConfig):
        self.cfg = cfg
        self.model = load_model(
            model_class_name=cfg.model_class_name,
            model_name=cfg.model_name,
            device=cfg.device,
        )
        self.activations_store = ActivationsStore.from_config(
            self.model,
            cfg,
        )

        self.file_extension = "safetensors"

    def __str__(self):
        """
        Print the number of tokens to be cached.
        Print the number of buffers, and the number of tokens per buffer.
        Print the disk space required to store the activations.

        """

        bytes_per_token = (
            self.cfg.d_in * self.cfg.dtype.itemsize
            if isinstance(self.cfg.dtype, torch.dtype)
            else DTYPE_MAP[self.cfg.dtype].itemsize
        )
        tokens_in_buffer = (
            self.cfg.n_batches_in_buffer
            * self.cfg.store_batch_size
            * self.cfg.context_size
        )
        total_training_tokens = self.cfg.training_tokens
        total_disk_space_gb = total_training_tokens * bytes_per_token / 10**9

        return (
            f"Activation Cache Runner:\n"
            f"Total training tokens: {total_training_tokens}\n"
            f"Number of buffers: {math.ceil(total_training_tokens / tokens_in_buffer)}\n"
            f"Tokens per buffer: {tokens_in_buffer}\n"
            f"Disk space required: {total_disk_space_gb:.2f} GB\n"
            f"Configuration:\n"
            f"{self.cfg}"
        )

    @torch.no_grad()
    def run(self):

        new_cached_activations_path = self.cfg.new_cached_activations_path

        # if the activations directory exists and has files in it, raise an exception
        assert new_cached_activations_path is not None
        if os.path.exists(new_cached_activations_path):
            if len(os.listdir(new_cached_activations_path)) > 0:
                raise Exception(
                    f"Activations directory ({new_cached_activations_path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
                )
        else:
            os.makedirs(new_cached_activations_path)

        print(f"Started caching {self.cfg.training_tokens} activations")
        tokens_per_buffer = (
            self.cfg.store_batch_size
            * self.cfg.context_size
            * self.cfg.n_batches_in_buffer
        )

        n_buffers = math.ceil(self.cfg.training_tokens / tokens_per_buffer)

        for i in tqdm(range(n_buffers), desc="Caching activations"):
            buffer = self.activations_store.get_buffer(self.cfg.n_batches_in_buffer)
            self.activations_store.save_buffer(
                buffer, f"{new_cached_activations_path}/{i}.safetensors"
            )

            del buffer

            if i % self.cfg.shuffle_every_n_buffers == 0 and i > 0:
                # Shuffle the buffers on disk

                # Do random pairwise shuffling between the last shuffle_every_n_buffers buffers
                for _ in range(self.cfg.n_shuffles_with_last_section):
                    self.shuffle_activations_pairwise(
                        new_cached_activations_path,
                        buffer_idx_range=(i - self.cfg.shuffle_every_n_buffers, i),
                    )

                # Do more random pairwise shuffling between all the buffers
                for _ in range(self.cfg.n_shuffles_in_entire_dir):
                    self.shuffle_activations_pairwise(
                        new_cached_activations_path,
                        buffer_idx_range=(0, i),
                    )

        # More final shuffling (mostly in case we didn't end on an i divisible by shuffle_every_n_buffers)
        if n_buffers > 1:
            for _ in tqdm(range(self.cfg.n_shuffles_final), desc="Final shuffling"):
                self.shuffle_activations_pairwise(
                    new_cached_activations_path,
                    buffer_idx_range=(0, n_buffers),
                )

    @torch.no_grad()
    def shuffle_activations_pairwise(
        self, datapath: str, buffer_idx_range: Tuple[int, int]
    ):
        """
        Shuffles two buffers on disk.
        """
        assert (
            buffer_idx_range[0] < buffer_idx_range[1] - 1
        ), "buffer_idx_range[0] must be smaller than buffer_idx_range[1] by at least 1"

        buffer_idx1 = torch.randint(
            buffer_idx_range[0], buffer_idx_range[1], (1,)
        ).item()
        buffer_idx2 = torch.randint(
            buffer_idx_range[0], buffer_idx_range[1], (1,)
        ).item()
        while buffer_idx1 == buffer_idx2:  # Make sure they're not the same
            buffer_idx2 = torch.randint(
                buffer_idx_range[0], buffer_idx_range[1], (1,)
            ).item()

        buffer1 = self.activations_store.load_buffer(
            f"{datapath}/{buffer_idx1}.{self.file_extension}"
        )
        buffer2 = self.activations_store.load_buffer(
            f"{datapath}/{buffer_idx2}.{self.file_extension}"
        )
        joint_buffer = torch.cat([buffer1, buffer2])

        # Shuffle them
        joint_buffer = joint_buffer[torch.randperm(joint_buffer.shape[0])]
        shuffled_buffer1 = joint_buffer[: buffer1.shape[0]]
        shuffled_buffer2 = joint_buffer[buffer1.shape[0] :]

        # Save them back
        self.activations_store.save_buffer(
            shuffled_buffer1, f"{datapath}/{buffer_idx1}.{self.file_extension}"
        )
        self.activations_store.save_buffer(
            shuffled_buffer2, f"{datapath}/{buffer_idx2}.{self.file_extension}"
        )
