from __future__ import annotations

import os
from typing import Any, Iterator, Literal, TypeVar, cast

import torch
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)
from torch.utils.data import DataLoader
from transformer_lens.hook_points import HookedRootModule

from sae_lens.training.config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
)

HfDataset = DatasetDict | Dataset | IterableDatasetDict | IterableDataset


class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    model: HookedRootModule
    dataset: HfDataset
    cached_activations_path: str | None
    tokens_column: Literal["tokens", "input_ids", "text"]
    hook_point_head_index: int | None
    _dataloader: Iterator[Any] | None = None
    _storage_buffer: torch.Tensor | None = None

    @classmethod
    def from_config(
        cls,
        model: HookedRootModule,
        cfg: LanguageModelSAERunnerConfig | CacheActivationsRunnerConfig,
        dataset: HfDataset | None = None,
    ) -> "ActivationsStore":
        cached_activations_path = cfg.cached_activations_path
        # set cached_activations_path to None if we're not using cached activations
        if (
            isinstance(cfg, LanguageModelSAERunnerConfig)
            and not cfg.use_cached_activations
        ):
            cached_activations_path = None
        return cls(
            model=model,
            dataset=dataset or cfg.dataset_path,
            hook_point=cfg.hook_point,
            hook_point_layers=listify(cfg.hook_point_layer),
            hook_point_head_index=cfg.hook_point_head_index,
            context_size=cfg.context_size,
            d_in=cfg.d_in,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size=cfg.store_batch_size,
            train_batch_size=cfg.train_batch_size,
            prepend_bos=cfg.prepend_bos,
            device=cfg.device,
            dtype=cfg.dtype,
            cached_activations_path=cached_activations_path,
            model_kwargs=cfg.model_kwargs,
        )

    def __init__(
        self,
        model: HookedRootModule,
        dataset: HfDataset | str,
        hook_point: str,
        hook_point_layers: list[int],
        hook_point_head_index: int | None,
        context_size: int,
        d_in: int,
        n_batches_in_buffer: int,
        total_training_tokens: int,
        store_batch_size: int,
        train_batch_size: int,
        prepend_bos: bool,
        device: str | torch.device,
        dtype: str | torch.dtype,
        cached_activations_path: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        self.model = model
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs
        self.dataset = (
            load_dataset(dataset, split="train", streaming=True)
            if isinstance(dataset, str)
            else dataset
        )
        self.hook_point = hook_point
        self.hook_point_layers = hook_point_layers
        self.hook_point_head_index = hook_point_head_index
        self.context_size = context_size
        self.d_in = d_in
        self.n_batches_in_buffer = n_batches_in_buffer
        self.total_training_tokens = total_training_tokens
        self.store_batch_size = store_batch_size
        self.train_batch_size = train_batch_size
        self.prepend_bos = prepend_bos
        self.device = device
        self.dtype = dtype
        self.cached_activations_path = cached_activations_path

        self.iterable_dataset = iter(self.dataset)

        # Check if dataset is tokenized
        dataset_sample = next(self.iterable_dataset)

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
        self.iterable_dataset = iter(self.dataset)  # Reset iterator after checking

        if cached_activations_path is not None:  # EDIT: load from multi-layer acts
            assert self.cached_activations_path is not None  # keep pyright happy
            # Sanity check: does the cache directory exist?
            assert os.path.exists(
                self.cached_activations_path
            ), f"Cache directory {self.cached_activations_path} does not exist. Consider double-checking your dataset, model, and hook names."

            self.next_cache_idx = 0  # which file to open next
            self.next_idx_within_buffer = 0  # where to start reading from in that file

            # Check that we have enough data on disk
            first_buffer = torch.load(f"{self.cached_activations_path}/0.pt")
            buffer_size_on_disk = first_buffer.shape[0]
            n_buffers_on_disk = len(os.listdir(self.cached_activations_path))
            # Note: we're assuming all files have the same number of tokens
            # (which seems reasonable imo since that's what our script does)
            n_activations_on_disk = buffer_size_on_disk * n_buffers_on_disk
            assert (
                n_activations_on_disk > self.total_training_tokens
            ), f"Only {n_activations_on_disk/1e6:.1f}M activations on disk, but total_training_tokens is {self.total_training_tokens/1e6:.1f}M."

            # TODO add support for "mixed loading" (ie use cache until you run out, then switch over to streaming from HF)

    @property
    def storage_buffer(self) -> torch.Tensor:
        if self._storage_buffer is None:
            self._storage_buffer = self.get_buffer(self.n_batches_in_buffer // 2)
        return self._storage_buffer

    @property
    def dataloader(self) -> Iterator[Any]:
        if self._dataloader is None:
            self._dataloader = self.get_data_loader()
        return self._dataloader

    def get_batch_tokens(self):
        """
        Streams a batch of tokens from a dataset.
        """

        batch_size = self.store_batch_size
        context_size = self.context_size
        device = self.device

        batch_tokens = torch.zeros(
            size=(0, context_size), device=device, dtype=torch.long, requires_grad=False
        )

        current_batch = []
        current_length = 0

        # pbar = tqdm(total=batch_size, desc="Filling batches")
        while batch_tokens.shape[0] < batch_size:
            tokens = self._get_next_dataset_tokens()
            token_len = tokens.shape[0]

            # TODO: Fix this so that we are limiting how many tokens we get from the same context.
            assert self.model.tokenizer is not None  # keep pyright happy
            while token_len > 0 and batch_tokens.shape[0] < batch_size:
                # Space left in the current batch
                space_left = context_size - current_length

                # If the current tokens fit entirely into the remaining space
                if token_len <= space_left:
                    current_batch.append(tokens[:token_len])
                    current_length += token_len
                    break

                else:
                    # Take as much as will fit
                    current_batch.append(tokens[:space_left])

                    # Remove used part, add BOS
                    tokens = tokens[space_left:]
                    token_len -= space_left

                    # only add BOS if it's not already the first token
                    if self.prepend_bos:
                        bos_token_id_tensor = torch.tensor(
                            [self.model.tokenizer.bos_token_id],
                            device=tokens.device,
                            dtype=torch.long,
                        )
                        if tokens[0] != bos_token_id_tensor:
                            tokens = torch.cat(
                                (
                                    bos_token_id_tensor,
                                    tokens,
                                ),
                                dim=0,
                            )
                            token_len += 1
                    current_length = context_size

                # If a batch is full, concatenate and move to next batch
                if current_length == context_size:
                    full_batch = torch.cat(current_batch, dim=0)
                    batch_tokens = torch.cat(
                        (batch_tokens, full_batch.unsqueeze(0)), dim=0
                    )
                    current_batch = []
                    current_length = 0

            # pbar.n = batch_tokens.shape[0]
            # pbar.refresh()
        return batch_tokens[:batch_size]

    def get_activations(self, batch_tokens: torch.Tensor):
        """
        Returns activations of shape (batches, context, num_layers, d_in)

        d_in may result from a concatenated head dimension.
        """
        layers = self.hook_point_layers
        act_names = [self.hook_point.format(layer=layer) for layer in layers]
        hook_point_max_layer = max(layers)
        layerwise_activations = self.model.run_with_cache(
            batch_tokens,
            names_filter=act_names,
            stop_at_layer=hook_point_max_layer + 1,
            prepend_bos=self.prepend_bos,
            **self.model_kwargs,
        )[1]
        activations_list = [layerwise_activations[act_name] for act_name in act_names]
        if self.hook_point_head_index is not None:
            activations_list = [
                act[:, :, self.hook_point_head_index] for act in activations_list
            ]
        elif activations_list[0].ndim > 3:  # if we have a head dimension
            # flatten the head dimension
            activations_list = [
                act.view(act.shape[0], act.shape[1], -1) for act in activations_list
            ]

        # Stack along a new dimension to keep separate layers distinct
        stacked_activations = torch.stack(activations_list, dim=2)

        return stacked_activations

    def get_buffer(self, n_batches_in_buffer: int) -> torch.Tensor:
        context_size = self.context_size
        batch_size = self.store_batch_size
        d_in = self.d_in
        total_size = batch_size * n_batches_in_buffer
        num_layers = len(self.hook_point_layers)  # Number of hook points or layers

        if self.cached_activations_path is not None:
            # Load the activations from disk
            buffer_size = total_size * context_size
            # Initialize an empty tensor with an additional dimension for layers
            new_buffer = torch.zeros(
                (buffer_size, num_layers, d_in),
                dtype=self.dtype,  # type: ignore
                device=self.device,
            )
            n_tokens_filled = 0

            # Assume activations for different layers are stored separately and need to be combined
            while n_tokens_filled < buffer_size:
                if not os.path.exists(
                    f"{self.cached_activations_path}/{self.next_cache_idx}.pt"
                ):
                    print(
                        "\n\nWarning: Ran out of cached activation files earlier than expected."
                    )
                    print(
                        f"Expected to have {buffer_size} activations, but only found {n_tokens_filled}."
                    )
                    if buffer_size % self.total_training_tokens != 0:
                        print(
                            "This might just be a rounding error — your batch_size * n_batches_in_buffer * context_size is not divisible by your total_training_tokens"
                        )
                    print(f"Returning a buffer of size {n_tokens_filled} instead.")
                    print("\n\n")
                    new_buffer = new_buffer[:n_tokens_filled, ...]
                    return new_buffer

                activations = torch.load(
                    f"{self.cached_activations_path}/{self.next_cache_idx}.pt"
                )
                taking_subset_of_file = False
                if n_tokens_filled + activations.shape[0] > buffer_size:
                    activations = activations[: buffer_size - n_tokens_filled, ...]
                    taking_subset_of_file = True

                new_buffer[
                    n_tokens_filled : n_tokens_filled + activations.shape[0], ...
                ] = activations

                if taking_subset_of_file:
                    self.next_idx_within_buffer = activations.shape[0]
                else:
                    self.next_cache_idx += 1
                    self.next_idx_within_buffer = 0

                n_tokens_filled += activations.shape[0]

            return new_buffer

        refill_iterator = range(0, batch_size * n_batches_in_buffer, batch_size)
        # Initialize empty tensor buffer of the maximum required size with an additional dimension for layers
        new_buffer = torch.zeros(
            (total_size, context_size, num_layers, d_in),
            dtype=self.dtype,  # type: ignore
            device=self.device,
        )

        for refill_batch_idx_start in refill_iterator:
            refill_batch_tokens = self.get_batch_tokens()
            refill_activations = self.get_activations(refill_batch_tokens)
            new_buffer[
                refill_batch_idx_start : refill_batch_idx_start + batch_size, ...
            ] = refill_activations

            # pbar.update(1)

        new_buffer = new_buffer.reshape(-1, num_layers, d_in)
        new_buffer = new_buffer[torch.randperm(new_buffer.shape[0])]

        return new_buffer

    def get_data_loader(
        self,
    ) -> Iterator[Any]:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """

        batch_size = self.train_batch_size

        # 1. # create new buffer by mixing stored and new buffer
        mixing_buffer = torch.cat(
            [self.get_buffer(self.n_batches_in_buffer // 2), self.storage_buffer],
            dim=0,
        )

        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        # 2.  put 50 % in storage
        self._storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # 3. put other 50 % in a dataloader
        dataloader = iter(
            DataLoader(
                # TODO: seems like a typing bug?
                cast(Any, mixing_buffer[mixing_buffer.shape[0] // 2 :]),
                batch_size=batch_size,
                shuffle=True,
            )
        )

        return dataloader

    def next_batch(self):
        """
        Get the next batch from the current DataLoader.
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self._dataloader = self.get_data_loader()
            return next(self.dataloader)

    def _get_next_dataset_tokens(self) -> torch.Tensor:
        device = self.device
        if not self.is_dataset_tokenized:
            s = next(self.iterable_dataset)[self.tokens_column]
            tokens = self.model.to_tokens(
                s,
                truncate=True,
                move_to_device=True,
                prepend_bos=self.prepend_bos,
            ).squeeze(0)
            assert (
                len(tokens.shape) == 1
            ), f"tokens.shape should be 1D but was {tokens.shape}"
        else:
            tokens = torch.tensor(
                next(self.iterable_dataset)[self.tokens_column],
                dtype=torch.long,
                device=device,
                requires_grad=False,
            )
            if (
                not self.prepend_bos
                and tokens[0] == self.model.tokenizer.bos_token_id  # type: ignore
            ):
                tokens = tokens[1:]
        return tokens


T = TypeVar("T")


def listify(x: T | list[T]) -> list[T]:
    if isinstance(x, list):
        return x
    return [x]
