from __future__ import annotations

import contextlib
import os
import tempfile
from typing import Any, Generator, Iterator, Literal, cast

import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule

from sae_lens.config import (
    DTYPE_MAP,
    CacheActivationsRunnerConfig,
    HfDataset,
    LanguageModelSAERunnerConfig,
)
from sae_lens.sae import SAE
from sae_lens.tokenization_and_batching import concat_and_batch_sequences

torch_to_numpy_dtype_dict: dict[torch.dtype, Any] = {
    torch.float32: np.float32,
    torch.float: np.float32,
    torch.float64: np.float64,
    torch.double: np.float64,
    torch.float16: np.float16,
    torch.half: np.float16,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.short: np.int16,
    torch.int32: np.int32,
    torch.int: np.int32,
    torch.int64: np.int64,
    torch.long: np.int64,
    torch.bool: np.bool_,
}

FILE_EXTENSION = "dat"


def torch_dtype_to_numpy_dtype(torch_dtype: torch.dtype) -> np.dtype[Any] | None:
    return torch_to_numpy_dtype_dict.get(torch_dtype, None)


# TODO: Make an activation store config class to be consistent with the rest of the code.
class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    model: HookedRootModule
    dataset: HfDataset
    cached_activations_path: str | None
    tokens_column: Literal["tokens", "input_ids", "text"]
    hook_name: str
    hook_layer: int
    hook_head_index: int | None
    _dataloader: Iterator[Any] | None = None
    _storage_buffer: np.ndarray[Any, np.dtype[Any]] | None = None
    device: torch.device

    @classmethod
    def from_config(
        cls,
        model: HookedRootModule,
        cfg: LanguageModelSAERunnerConfig | CacheActivationsRunnerConfig,
        override_dataset: HfDataset | None = None,
    ) -> "ActivationsStore":
        cached_activations_path = cfg.cached_activations_path
        # set cached_activations_path to None if we're not using cached activations
        if (
            isinstance(cfg, LanguageModelSAERunnerConfig)
            and not cfg.use_cached_activations
        ):
            cached_activations_path = None

        if override_dataset is None and cfg.dataset_path == "":
            raise ValueError(
                "You must either pass in a dataset or specify a dataset_path in your configutation."
            )

        return cls(
            model=model,
            dataset=override_dataset or cfg.dataset_path,
            streaming=cfg.streaming,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            hook_head_index=cfg.hook_head_index,
            context_size=cfg.context_size,
            d_in=cfg.d_in,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.store_batch_size_prompts,
            train_batch_size_tokens=cfg.train_batch_size_tokens,
            prepend_bos=cfg.prepend_bos,
            normalize_activations=cfg.normalize_activations,
            device=torch.device(cfg.act_store_device),
            dtype=cfg.dtype,
            cached_activations_path=cached_activations_path,
            model_kwargs=cfg.model_kwargs,
            autocast_lm=cfg.autocast_lm,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
        )

    @classmethod
    def from_sae(
        cls,
        model: HookedRootModule,
        sae: SAE,
        streaming: bool = True,
        store_batch_size_prompts: int = 8,
        n_batches_in_buffer: int = 8,
        train_batch_size_tokens: int = 4096,
        total_tokens: int = 10**9,
        device: str = "cpu",
    ) -> "ActivationsStore":

        return cls(
            model=model,
            dataset=sae.cfg.dataset_path,
            d_in=sae.cfg.d_in,
            hook_name=sae.cfg.hook_name,
            hook_layer=sae.cfg.hook_layer,
            hook_head_index=sae.cfg.hook_head_index,
            context_size=sae.cfg.context_size,
            prepend_bos=sae.cfg.prepend_bos,
            streaming=streaming,
            store_batch_size_prompts=store_batch_size_prompts,
            train_batch_size_tokens=train_batch_size_tokens,
            n_batches_in_buffer=n_batches_in_buffer,
            total_training_tokens=total_tokens,
            normalize_activations=sae.cfg.normalize_activations,
            dataset_trust_remote_code=sae.cfg.dataset_trust_remote_code,
            dtype=sae.cfg.dtype,
            device=torch.device(device),
        )

    def __init__(
        self,
        model: HookedRootModule,
        dataset: HfDataset | str,
        streaming: bool,
        hook_name: str,
        hook_layer: int,
        hook_head_index: int | None,
        context_size: int,
        d_in: int,
        n_batches_in_buffer: int,
        total_training_tokens: int,
        store_batch_size_prompts: int,
        train_batch_size_tokens: int,
        prepend_bos: bool,
        normalize_activations: str,
        device: torch.device,
        dtype: str,
        cached_activations_path: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        autocast_lm: bool = False,
        dataset_trust_remote_code: bool | None = None,
    ):
        self.model = model
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs
        self.dataset = (
            load_dataset(
                dataset,
                split="train",
                streaming=streaming,
                trust_remote_code=dataset_trust_remote_code,  # type: ignore
            )
            if isinstance(dataset, str)
            else dataset
        )
        self.hook_name = hook_name
        self.hook_layer = hook_layer
        self.hook_head_index = hook_head_index
        self.context_size = context_size
        self.d_in = d_in
        self.n_batches_in_buffer = n_batches_in_buffer
        self.total_training_tokens = total_training_tokens
        self.store_batch_size_prompts = store_batch_size_prompts
        self.train_batch_size_tokens = train_batch_size_tokens
        self.prepend_bos = prepend_bos
        self.normalize_activations = normalize_activations
        self.device = torch.device(device)
        self.dtype = DTYPE_MAP[dtype]
        self.numpy_dtype = torch_dtype_to_numpy_dtype(self.dtype)
        self.cached_activations_path = cached_activations_path
        self.autocast_lm = autocast_lm

        self.n_dataset_processed = 0

        self.estimated_norm_scaling_factor = 1.0

        # Check if dataset is tokenized
        dataset_sample = next(iter(self.dataset))

        # check if it's tokenized
        if "tokens" in dataset_sample.keys():
            self.is_dataset_tokenized = True
            self.tokens_column = "tokens"
        elif "input_ids" in dataset_sample.keys():
            self.is_dataset_tokenized = True
            self.tokens_column = "input_ids"
        elif "text" in dataset_sample.keys():
            self.is_dataset_tokenized = False
            self.tokens_column = "text"
        else:
            raise ValueError(
                "Dataset must have a 'tokens', 'input_ids', or 'text' column."
            )
        if self.is_dataset_tokenized:
            ds_context_size = len(dataset_sample[self.tokens_column])
            if ds_context_size != self.context_size:
                raise ValueError(
                    f"pretokenized dataset has context_size {ds_context_size}, but the provided context_size is {self.context_size}."
                )
            # TODO: investigate if this can work for iterable datasets, or if this is even worthwhile as a perf improvement
            if hasattr(self.dataset, "set_format"):
                self.dataset.set_format(type="torch", columns=[self.tokens_column])  # type: ignore
        else:
            print(
                "Warning: Dataset is not tokenized. Pre-tokenizing will improve performance and allows for more control over special tokens. See https://jbloomaus.github.io/SAELens/training_saes/#pretokenizing-datasets for more info."
            )

        self.iterable_sequences = self._iterate_tokenized_sequences()

        self.check_cached_activations_against_config()

        # TODO add support for "mixed loading" (ie use cache until you run out, then switch over to streaming from HF)

    def _iterate_raw_dataset(
        self,
    ) -> Generator[torch.Tensor | list[int] | str, None, None]:
        """
        Helper to iterate over the dataset while incrementing n_dataset_processed
        """
        for row in self.dataset:
            # typing datasets is difficult
            yield row[self.tokens_column]  # type: ignore
            self.n_dataset_processed += 1

    def _iterate_raw_dataset_tokens(self) -> Generator[torch.Tensor, None, None]:
        """
        Helper to create an iterator which tokenizes raw text from the dataset on the fly
        """
        for row in self._iterate_raw_dataset():
            tokens = (
                self.model.to_tokens(
                    row,
                    truncate=False,
                    move_to_device=True,
                    prepend_bos=False,
                )
                .squeeze(0)
                .to(self.device)
            )
            assert (
                len(tokens.shape) == 1
            ), f"tokens.shape should be 1D but was {tokens.shape}"
            yield tokens

    def _iterate_tokenized_sequences(self) -> Generator[torch.Tensor, None, None]:
        """
        Generator which iterates over full sequence of context_size tokens
        """
        # If the datset is pretokenized, we can just return each row as a tensor, no further processing is needed.
        # We assume that all necessary BOS/EOS/SEP tokens have been added during pretokenization.
        if self.is_dataset_tokenized:
            for row in self._iterate_raw_dataset():
                yield torch.tensor(
                    row,
                    dtype=torch.long,
                    device=self.device,
                    requires_grad=False,
                )
        # If the dataset isn't tokenized, we'll tokenize, concat, and batch on the fly
        else:
            tokenizer = getattr(self.model, "tokenizer", None)
            bos_token_id = None if tokenizer is None else tokenizer.bos_token_id
            yield from concat_and_batch_sequences(
                tokens_iterator=self._iterate_raw_dataset_tokens(),
                context_size=self.context_size,
                begin_batch_token_id=(bos_token_id if self.prepend_bos else None),
                begin_sequence_token_id=None,
                sequence_separator_token_id=(
                    bos_token_id if self.prepend_bos else None
                ),
            )

    def check_cached_activations_against_config(self):
        if self.cached_activations_path is not None:
            assert os.path.exists(
                self.cached_activations_path
            ), f"Cache directory {self.cached_activations_path} does not exist. Consider double-checking your dataset, model, and hook names."

            self.next_cache_idx = 0
            self.next_idx_within_buffer = 0

            # Check that we have enough data on disk
            first_buffer = self.load_buffer(
                f"{self.cached_activations_path}/0.{FILE_EXTENSION}"
            )

            buffer_size_on_disk = first_buffer.shape[0]
            n_buffers_on_disk = len(
                [
                    f
                    for f in os.listdir(self.cached_activations_path)
                    if f.endswith(FILE_EXTENSION)
                ]
            )

            n_activations_on_disk = buffer_size_on_disk * n_buffers_on_disk
            assert (
                n_activations_on_disk >= self.total_training_tokens
            ), f"Only {n_activations_on_disk/1e6:.1f}M activations on disk, but total_training_tokens is {self.total_training_tokens/1e6:.1f}M."

    def apply_norm_scaling_factor(
        self,
        activations: torch.Tensor,
    ) -> torch.Tensor:
        return activations * self.estimated_norm_scaling_factor

    def unscale(self, activations: torch.Tensor) -> torch.Tensor:
        return activations / self.estimated_norm_scaling_factor

    def get_norm_scaling_factor(self, activations: torch.Tensor) -> torch.Tensor:
        return (self.d_in**0.5) / activations.norm(dim=-1).mean()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, n_batches_for_norm_estimate: int = int(1e3)):

        norms_per_batch = []
        for _ in tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            acts = self.next_batch()
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(self.d_in) / mean_norm

        return scaling_factor

    @property
    def storage_buffer(self) -> np.ndarray[Any, np.dtype[Any]]:
        if self._storage_buffer is None:
            self._storage_buffer = self.get_buffer()
        return self._storage_buffer

    @property
    def dataloader(self) -> Iterator[Any]:
        if self._dataloader is None:
            self._dataloader = self.get_data_loader()
        return self._dataloader

    def get_batch_tokens(self, batch_size: int | None = None):
        """
        Streams a batch of tokens from a dataset.
        """
        if not batch_size:
            batch_size = self.store_batch_size_prompts
        sequences = []
        # the sequences iterator yields fully formed tokens of size context_size, so we just need to cat these into a batch
        for _ in range(batch_size):
            sequences.append(next(self.iterable_sequences))
        return torch.stack(sequences, dim=0).to(self.model.W_E.device)

    @torch.no_grad()
    def get_activations(self, batch_tokens: torch.Tensor):
        """
        Returns activations of shape (batches, context, num_layers, d_in)

        d_in may result from a concatenated head dimension.
        """

        # Setup autocast if using
        if self.autocast_lm:
            autocast_if_enabled = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.autocast_lm,
            )
        else:
            autocast_if_enabled = contextlib.nullcontext()

        with autocast_if_enabled:
            layerwise_activations = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_name],
                stop_at_layer=self.hook_layer + 1,
                prepend_bos=False,
                **self.model_kwargs,
            )[1]

        n_batches, n_context = batch_tokens.shape

        stacked_activations = torch.zeros((n_batches, n_context, 1, self.d_in))

        if self.hook_head_index is not None:
            stacked_activations[:, :, 0] = layerwise_activations[self.hook_name][
                :, :, self.hook_head_index
            ]
        elif (
            layerwise_activations[self.hook_name].ndim > 3
        ):  # if we have a head dimension
            stacked_activations[:, :, 0] = layerwise_activations[self.hook_name].view(
                n_batches, n_context, -1
            )
        else:
            stacked_activations[:, :, 0] = layerwise_activations[self.hook_name]

        return stacked_activations

    @torch.no_grad()
    def get_buffer(self) -> np.memmap[Any, np.dtype[Any]]:
        context_size = self.context_size
        batch_size = self.store_batch_size_prompts
        d_in = self.d_in
        total_size = batch_size * self.n_batches_in_buffer
        num_layers = 1

        if self.cached_activations_path is not None:
            # Load the activations from disk (this part remains similar to before)
            # buffer_size = total_size * context_size * 2
            next_buffer_path = (
                f"{self.cached_activations_path}/{self.next_cache_idx}.{FILE_EXTENSION}"
            )
            new_buffer = self.load_buffer(next_buffer_path, num_layers)

        else:
            # Generate the buffer directly
            refill_iterator = range(
                0, batch_size * self.n_batches_in_buffer, batch_size
            )

            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{FILE_EXTENSION}"
            ) as tmp:
                temp_file_path = tmp.name

            # Initialize empty numpy memmap of the maximum required size
            new_buffer = np.memmap(
                temp_file_path,
                dtype=self.numpy_dtype,
                mode="w+",
                shape=(total_size, context_size, num_layers, d_in),
            )

            for refill_batch_idx_start in refill_iterator:
                # move batch toks to gpu for model
                refill_batch_tokens = self.get_batch_tokens().to(self.model.cfg.device)
                refill_activations = self.get_activations(refill_batch_tokens)
                # move acts back to cpu and convert to numpy
                refill_activations = refill_activations.cpu().numpy()
                new_buffer[
                    refill_batch_idx_start : refill_batch_idx_start + batch_size, ...
                ] = refill_activations

            # Reshape and shuffle
            new_buffer = new_buffer.reshape(-1, num_layers, d_in)
            np.random.shuffle(new_buffer)
            new_buffer = cast(np.memmap[Any, np.dtype[Any]], new_buffer)

        # # Normalize if needed
        # if self.normalize_activations == "expected_average_only_in":
        #     new_buffer = self.apply_norm_scaling_factor(new_buffer)

        return new_buffer

    @classmethod
    def _save_buffer(cls, buffer: np.memmap[Any, np.dtype[Any]], path: str):
        # If buffer is already a memmap, we just need to flush it
        if buffer.filename != path:
            # If the paths are different, we need to create a new memmap
            new_buffer = np.memmap(
                path, dtype=buffer.dtype, mode="w+", shape=buffer.shape
            )
            new_buffer[:] = buffer[:]
            new_buffer.flush()
        else:
            # If the paths are the same, we just need to flush the existing buffer
            buffer.flush()

    def save_buffer(self, buffer: np.memmap[Any, np.dtype[Any]], path: str):
        self._save_buffer(buffer, path)

    @classmethod
    def _load_buffer(cls, path: str, num_layers:int, dtype: np.dtype[Any] | None, d_in: int) -> np.memmap[Any, np.dtype[Any]]:
        # Load the memory-mapped array
        memmap_file = np.memmap(path, dtype=dtype, mode="r")
        memmap_file = cast(np.memmap[Any, np.dtype[Any]], memmap_file.reshape(-1, num_layers, d_in))
        return memmap_file

    def load_buffer(self, path: str, num_layers:int=1) -> np.memmap[Any, np.dtype[Any]]:
        return self._load_buffer(path, num_layers, self.numpy_dtype, self.d_in)

    def get_data_loader(self) -> Iterator[Any]:
        batch_size = self.train_batch_size_tokens

        # Create new buffer by mixing stored and new buffer
        mixing_buffer = np.concatenate([self.get_buffer(), self.storage_buffer])

        # Shuffle the mixing buffer
        np.random.shuffle(mixing_buffer)

        # Put 50% in storage
        self._storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # Create a dataset from the other 50%
        # if self.normalize_activations == "expected_average_only_in":
        #     # Apply normalization when creating the dataset
        #     normalized_buffer = self.apply_norm_scaling_factor(
        #         torch.from_numpy(mixing_buffer[mixing_buffer.shape[0] // 2 :])
        #     )
        #     dataset = torch.utils.data.TensorDataset(normalized_buffer)
        # else:
        dataset = torch.utils.data.TensorDataset( # type: ignore
            torch.from_numpy(mixing_buffer[mixing_buffer.shape[0] // 2 :])
        )

        # Create and return the dataloader
        return iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))

    def next_batch(self) -> torch.Tensor:
        """
        Get the next batch from the current DataLoader.
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            x = next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self._dataloader = self.get_data_loader()
            x = next(self.dataloader)

        x = x[0].to(self.device)
        if self.normalize_activations == "expected_average_only_in":
            x = self.apply_norm_scaling_factor(x)

        return x

    def state_dict(self) -> dict[str, torch.Tensor]:
        result = {
            "n_dataset_processed": torch.tensor(self.n_dataset_processed),
        }
        if self._storage_buffer is not None:  # first time might be None
            result["storage_buffer"] = self._storage_buffer # type: ignore
        return result

    def save(self, file_path: str):
        save_file(self.state_dict(), file_path)
