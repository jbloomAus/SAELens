import io
import json
import math
import os
import shutil
from dataclasses import asdict

import einops
import torch
from datasets import Array2D, Dataset, Features, concatenate_datasets, load_from_disk
from huggingface_hub import HfApi
from jaxtyping import Float
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
        ctx_size = _get_sliced_context_size(self.cfg)
        self.features = Features(
            {
                f"{self.cfg.hook_name}": Array2D(
                    shape=(ctx_size, self.cfg.d_in), dtype=self.cfg.dtype
                )
            }
        )
        self.tokens_in_buffer = (
            self.cfg.n_batches_in_buffer * self.cfg.store_batch_size_prompts * ctx_size
        )
        self.n_buffers = math.ceil(self.cfg.training_tokens / self.tokens_in_buffer)

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

    @torch.no_grad()
    def _create_shard(
        self,
        buffer: Float[torch.Tensor, "(bs context_size) num_layers d_in"],
    ) -> Dataset:
        hook_names = [self.cfg.hook_name]  # allow multiple hooks in future

        buffer = einops.rearrange(
            buffer,
            "(bs context_size) num_layers d_in -> num_layers bs context_size d_in",
            bs=self.cfg.n_batches_in_buffer * self.cfg.store_batch_size_prompts,
            context_size=_get_sliced_context_size(self.cfg),
            d_in=self.cfg.d_in,
            num_layers=len(hook_names),
        )
        shard = Dataset.from_dict(
            {hook_name: act for hook_name, act in zip(hook_names, buffer)},
            features=self.features,
        )
        return shard

    @torch.no_grad()
    def run(self) -> Dataset:
        new_cached_activations_path = self.cfg.new_cached_activations_path
        assert new_cached_activations_path is not None

        ### Paths setup

        # if the activations directory exists and has files in it, raise an exception
        if os.path.exists(new_cached_activations_path):
            if len(os.listdir(new_cached_activations_path)) > 0:
                raise Exception(
                    f"Activations directory ({new_cached_activations_path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
                )
        else:
            os.makedirs(new_cached_activations_path)

        ### Create temporary sharded datasets

        print(f"Started caching {self.cfg.training_tokens} activations")

        for i in tqdm(range(self.n_buffers), desc="Caching activations"):
            try:
                # num activations in a single shard: n_batches_in_buffer * store_batch_size_prompts
                buffer = self.activations_store.get_buffer(
                    self.cfg.n_batches_in_buffer, shuffle=self.cfg.shuffle
                )
                shard = self._create_shard(buffer)
                shard.save_to_disk(
                    f"{new_cached_activations_path}/shard_{i}", num_shards=1
                )
                del buffer, shard

            except StopIteration:
                print(
                    f"Warning: Ran out of samples while filling the buffer at batch {i} before reaching {self.n_buffers} batches. No more caching will occur."
                )
                break

        ### Concat sharded datasets together, shuffle and push to hub

        # mem mapped
        dataset_shard_paths = [
            f"{new_cached_activations_path}/shard_{i}" for i in range(self.n_buffers)
        ]
        dataset_shards = [
            Dataset.load_from_disk(shard_path) for shard_path in dataset_shard_paths
        ]

        print("Concatenating shards...")
        dataset = concatenate_datasets(dataset_shards)

        if self.cfg.shuffle:
            print("Shuffling...")
            dataset = dataset.shuffle(seed=self.cfg.seed)

        dataset.save_to_disk(new_cached_activations_path)

        for shard_path in dataset_shard_paths:
            shutil.rmtree(shard_path)

        dataset = load_from_disk(new_cached_activations_path)
        assert isinstance(dataset, Dataset)

        if self.cfg.hf_repo_id:
            print("Pushing to hub...")
            dataset.push_to_hub(
                repo_id=self.cfg.hf_repo_id,
                num_shards=self.cfg.hf_num_shards or self.n_buffers,
                private=self.cfg.hf_is_private_repo,
                revision=self.cfg.hf_revision,
            )

            meta_io = io.BytesIO()
            meta_contents = json.dumps(
                asdict(self.cfg), indent=2, ensure_ascii=False
            ).encode("utf-8")
            meta_io.write(meta_contents)
            meta_io.seek(0)

            api = HfApi()
            api.upload_file(
                path_or_fileobj=meta_io,
                path_in_repo="cache_activations_runner_cfg.json",
                repo_id=self.cfg.hf_repo_id,
                repo_type="dataset",
                commit_message="Add cache_activations_runner metadata",
            )

        return dataset


def _get_sliced_context_size(cfg: CacheActivationsRunnerConfig) -> int:
    context_size = cfg.context_size
    if cfg.seqpos_slice:
        context_size = len(range(context_size)[slice(*cfg.seqpos_slice)])
    return context_size
