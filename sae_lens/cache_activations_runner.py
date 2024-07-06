import math
import os

import numpy as np
import torch
from tqdm import tqdm

from sae_lens.config import DTYPE_MAP, CacheActivationsRunnerConfig
from sae_lens.load_model import load_model
from sae_lens.training.activations_store import ActivationsStore


class CacheActivationsRunner:

    def __init__(self, cfg: CacheActivationsRunnerConfig):
        self.cfg = cfg
        self.model = load_model(
            model_class_name=cfg.model_class_name,
            model_name=cfg.model_name,
            device=cfg.device,
            model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
        )
        self.activations_store = ActivationsStore.from_config(
            self.model,
            cfg,
        )

        self.file_extension = "dat"

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
        total_training_tokens = self.cfg.training_tokens
        total_disk_space_gb = total_training_tokens * bytes_per_token / 10**9

        return (
            f"Activation Cache Runner:\n"
            f"Total training tokens: {total_training_tokens}\n"
            f"Number of buffers: {self.n_buffers}\n"
            f"Tokens per buffer: {self.tokens_in_buffer}\n"
            f"Disk space required: {total_disk_space_gb:.2f} GB\n"
            f"Configuration:\n"
            f"{self.cfg}"
        )
    
    @property
    def tokens_in_buffer(self):
        return (
            self.cfg.n_batches_in_buffer
            * self.cfg.store_batch_size_prompts
            * self.cfg.context_size
        )

    @property
    def n_buffers(self):
        return math.ceil(self.cfg.training_tokens / self.tokens_in_buffer)

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


        for i in tqdm(range(self.n_buffers), desc="Caching activations"):
            buffer = self.activations_store.get_buffer()
            buffer_path = f"{new_cached_activations_path}/{i}.{self.file_extension}"
            self.activations_store.save_buffer(buffer, buffer_path)

            del buffer

            if i > 0 and i % self.cfg.shuffle_every_n_buffers == 0:
                # Shuffle the buffers on disk

                # Do random pairwise shuffling between the last shuffle_every_n_buffers buffers
                for _ in range(self.cfg.n_shuffles_with_last_section):
                    self.shuffle_activations_pairwise(
                        new_cached_activations_path,
                        start_idx = i - self.cfg.shuffle_every_n_buffers,
                        end_idx = i,
                    )

                # Do more random pairwise shuffling between all the buffers
                for _ in range(self.cfg.n_shuffles_in_entire_dir):
                    self.shuffle_activations_pairwise(
                        new_cached_activations_path,
                        start_idx=0,
                        end_idx=i
                    )

        # More final shuffling (mostly in case we didn't end on an i divisible by shuffle_every_n_buffers)
        if self.n_buffers > 1:
            for _ in tqdm(range(self.cfg.n_shuffles_final), desc="Final shuffling"):
                self.shuffle_activations_pairwise(
                    new_cached_activations_path,
                    start_idx = 0,
                    end_idx = self.n_buffers,
                )

    @torch.no_grad()
    def shuffle_activations_pairwise(
        self, datapath: str, start_idx: int, end_idx: int
    ):
        """
        Shuffles two buffers on disk.
        """
        assert (
            start_idx < end_idx - 1
        ), "buffer_idx_range[0] must be smaller than buffer_idx_range[1] by at least 1"

        buffer_idx1 = torch.randint(
            start_idx, end_idx, (1,)
        ).item()
        buffer_idx2 = torch.randint(
            start_idx, end_idx, (1,)
        ).item()
        while buffer_idx1 == buffer_idx2:  # Make sure they're not the same
            buffer_idx2 = torch.randint(
                start_idx, end_idx, (1,)
            ).item()

        path1 = f"{datapath}/{buffer_idx1}.{self.file_extension}"
        path2 = f"{datapath}/{buffer_idx2}.{self.file_extension}"

        buffer1 = self.activations_store.load_buffer(path1)
        buffer2 = self.activations_store.load_buffer(path2)

        # Get total size and create a joint buffer
        total_size = buffer1.shape[0] + buffer2.shape[0]
        joint_buffer = np.memmap(
            f"{datapath}/temp_joint_buffer",
            dtype=buffer1.dtype,
            mode="w+",
            shape=(total_size,) + buffer1.shape[1:],
        )

        # Copy data to joint buffer
        joint_buffer[: buffer1.shape[0]] = buffer1
        joint_buffer[buffer1.shape[0] :] = buffer2

        # Generate random permutation
        permutation = np.random.permutation(total_size)

        # Create shuffled buffers
        shuffled_buffer1 = np.memmap(
            f"{datapath}/temp_shuffled_1",
            dtype=buffer1.dtype,
            mode="w+",
            shape=buffer1.shape,
        )
        shuffled_buffer2 = np.memmap(
            f"{datapath}/temp_shuffled_2",
            dtype=buffer2.dtype,
            mode="w+",
            shape=buffer2.shape,
        )

        # Apply permutation
        shuffled_buffer1[:] = joint_buffer[permutation[: buffer1.shape[0]]]
        shuffled_buffer2[:] = joint_buffer[permutation[buffer1.shape[0] :]]

        # Save shuffled buffers back to original files
        self.activations_store.save_buffer(shuffled_buffer1, path1)
        self.activations_store.save_buffer(shuffled_buffer2, path2)

        # Clean up temporary files
        import os

        os.remove(f"{datapath}/temp_joint_buffer")
        os.remove(f"{datapath}/temp_shuffled_1")
        os.remove(f"{datapath}/temp_shuffled_2")
