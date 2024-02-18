import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer


class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    def __init__(
        self,
        cfg,
        model: HookedTransformer,
        create_dataloader: bool = True,
    ):
        self.cfg = cfg
        self.model = model
        self.dataset = load_dataset(cfg.dataset_path, split="train", streaming=False)
        self.iterable_dataset = iter(self.dataset)

        # check if it's tokenized
        if "tokens" in next(self.iterable_dataset).keys():
            self.cfg.is_dataset_tokenized = True
            print("Dataset is tokenized! Updating config.")
        elif "text" in next(self.iterable_dataset).keys():
            self.cfg.is_dataset_tokenized = False
            print("Dataset is not tokenized! Updating config.")

        if self.cfg.use_cached_activations:
            # Sanity check: does the cache directory exist?
            assert os.path.exists(
                self.cfg.cached_activations_path
            ), f"Cache directory {self.cfg.cached_activations_path} does not exist. Consider double-checking your dataset, model, and hook names."

            self.next_cache_idx = 0  # which file to open next
            self.next_idx_within_buffer = 0  # where to start reading from in that file

            # Check that we have enough data on disk
            first_buffer = torch.load(f"{self.cfg.cached_activations_path}/0.pt")
            buffer_size_on_disk = first_buffer.shape[0]
            n_buffers_on_disk = len(os.listdir(self.cfg.cached_activations_path))
            # Note: we're assuming all files have the same number of tokens
            # (which seems reasonable imo since that's what our script does)
            n_activations_on_disk = buffer_size_on_disk * n_buffers_on_disk
            assert (
                n_activations_on_disk > self.cfg.total_training_tokens
            ), f"Only {n_activations_on_disk/1e6:.1f}M activations on disk, but cfg.total_training_tokens is {self.cfg.total_training_tokens/1e6:.1f}M."

            # TODO add support for "mixed loading" (ie use cache until you run out, then switch over to streaming from HF)

        if create_dataloader:
            # fill buffer half a buffer, so we can mix it with a new buffer
            self.storage_buffer = self.get_buffer(self.cfg.n_batches_in_buffer // 2)
            self.dataloader = self.get_data_loader()

    def get_batch_tokens(self, batch_size=None):
        """
        Overriding this method to not care about NLP specific stuff.
        """

        if batch_size is None:
            batch_size = self.cfg.store_batch_size

        batch_tokens = torch.zeros(
            (batch_size, self.cfg.context_size),
            dtype=torch.long,
            device=self.cfg.device,
        )

        for i in range(batch_size):
            # truncate the tokens if they are longer than the model's context size
            # TODO: investigate why n_ctx of OthelloGPT is 59 and not 60
            batch_tokens[i] = torch.tensor(
                next(self.iterable_dataset)["tokens"],
                dtype=torch.long,
                device=self.cfg.device,
            )[:self.cfg.context_size]

        return batch_tokens

    # def get_batch_tokens(self):
    #     """
    #     Streams a batch of tokens from a dataset.
    #     """

    #     batch_size = self.cfg.store_batch_size
    #     context_size = self.cfg.context_size
    #     device = self.cfg.device

    #     batch_tokens = torch.zeros(
    #         size=(0, context_size), device=device, dtype=torch.long, requires_grad=False
    #     )

    #     current_batch = []
    #     current_length = 0

    #     # pbar = tqdm(total=batch_size, desc="Filling batches")
    #     while batch_tokens.shape[0] < batch_size:
    #         if not self.cfg.is_dataset_tokenized:
    #             s = next(self.iterable_dataset)["text"]
    #             tokens = self.model.to_tokens(
    #                 s,
    #                 truncate=True,
    #                 move_to_device=True,
    #             ).squeeze(0)
    #             assert (
    #                 len(tokens.shape) == 1
    #             ), f"tokens.shape should be 1D but was {tokens.shape}"
    #         else:
    #             tokens = torch.tensor(
    #                 next(self.iterable_dataset)["tokens"],
    #                 dtype=torch.long,
    #                 device=device,
    #                 requires_grad=False,
    #             )
    #         token_len = tokens.shape[0]

    #         # TODO: Fix this so that we are limiting how many tokens we get from the same context.

    #         bos_token_id_tensor = torch.tensor(
    #             [self.model.tokenizer.bos_token_id],
    #             device=tokens.device,
    #             dtype=torch.long,
    #         )
    #         while token_len > 0 and batch_tokens.shape[0] < batch_size:
    #             # Space left in the current batch
    #             space_left = context_size - current_length

    #             # If the current tokens fit entirely into the remaining space
    #             if token_len <= space_left:
    #                 current_batch.append(tokens[:token_len])
    #                 current_length += token_len
    #                 break

    #             else:
    #                 # Take as much as will fit
    #                 current_batch.append(tokens[:space_left])

    #                 # Remove used part, add BOS
    #                 tokens = tokens[space_left:]
    #                 tokens = torch.cat(
    #                     (
    #                         bos_token_id_tensor,
    #                         tokens,
    #                     ),
    #                     dim=0,
    #                 )

    #                 token_len -= space_left
    #                 token_len += 1
    #                 current_length = context_size

    #             # If a batch is full, concatenate and move to next batch
    #             if current_length == context_size:
    #                 full_batch = torch.cat(current_batch, dim=0)
    #                 batch_tokens = torch.cat(
    #                     (batch_tokens, full_batch.unsqueeze(0)), dim=0
    #                 )
    #                 current_batch = []
    #                 current_length = 0

    #         # pbar.n = batch_tokens.shape[0]
    #         # pbar.refresh()
    #     return batch_tokens[:batch_size]

    def get_activations(self, batch_tokens, get_loss=False):
        act_name = self.cfg.hook_point
        hook_point_layer = self.cfg.hook_point_layer
        if self.cfg.hook_point_head_index is not None:
            activations = self.model.run_with_cache(
                batch_tokens, names_filter=act_name, stop_at_layer=hook_point_layer + 1
            )[1][act_name][:, self.cfg.start_pos_offset:self.cfg.end_pos_offset, self.cfg.hook_point_head_index]
        else:
            activations = self.model.run_with_cache(
                batch_tokens, names_filter=act_name, stop_at_layer=hook_point_layer + 1
            )[1][act_name][:, self.cfg.start_pos_offset:self.cfg.end_pos_offset]

        return activations

    def get_buffer(self, n_batches_in_buffer):
        context_size = self.cfg.context_size
        batch_size = self.cfg.store_batch_size
        d_in = self.cfg.d_in
        total_size = batch_size * n_batches_in_buffer

        start_pos_offset = self.cfg.start_pos_offset
        end_pos_offset = self.cfg.end_pos_offset

        effective_context_size = context_size
        if start_pos_offset is not None:
            effective_context_size-=start_pos_offset
        if end_pos_offset is not None:
            effective_context_size+=end_pos_offset

        if self.cfg.use_cached_activations:
            # Load the activations from disk
            buffer_size = total_size * effective_context_size
            # Initialize an empty tensor (flattened along all dims except d_in)
            new_buffer = torch.zeros(
                (buffer_size, d_in), dtype=self.cfg.dtype, device=self.cfg.device
            )
            n_tokens_filled = 0

            # The activations may be split across multiple files,
            # Or we might only want a subset of one file (depending on the sizes)
            while n_tokens_filled < buffer_size:
                # Load the next file
                # Make sure it exists
                if not os.path.exists(
                    f"{self.cfg.cached_activations_path}/{self.next_cache_idx}.pt"
                ):
                    print(
                        "\n\nWarning: Ran out of cached activation files earlier than expected."
                    )
                    print(
                        f"Expected to have {buffer_size} activations, but only found {n_tokens_filled}."
                    )
                    if buffer_size % self.cfg.total_training_tokens != 0:
                        print(
                            "This might just be a rounding error — your batch_size * n_batches_in_buffer * context_size is not divisible by your total_training_tokens"
                        )
                    print(f"Returning a buffer of size {n_tokens_filled} instead.")
                    print("\n\n")
                    new_buffer = new_buffer[:n_tokens_filled]
                    break
                activations = torch.load(
                    f"{self.cfg.cached_activations_path}/{self.next_cache_idx}.pt"
                )

                # If we only want a subset of the file, take it
                taking_subset_of_file = False
                if n_tokens_filled + activations.shape[0] > buffer_size:
                    activations = activations[: buffer_size - n_tokens_filled]
                    taking_subset_of_file = True

                # Add it to the buffer
                new_buffer[n_tokens_filled : n_tokens_filled + activations.shape[0]] = (
                    activations
                )

                # Update counters
                n_tokens_filled += activations.shape[0]
                if taking_subset_of_file:
                    self.next_idx_within_buffer = activations.shape[0]
                else:
                    self.next_cache_idx += 1
                    self.next_idx_within_buffer = 0

            return new_buffer

        refill_iterator = range(0, batch_size * n_batches_in_buffer, batch_size)
        # refill_iterator = tqdm(refill_iterator, desc="generate activations")

        # Initialize empty tensor buffer of the maximum required size

        new_buffer = torch.zeros(
            (total_size, effective_context_size, d_in),
            dtype=self.cfg.dtype,
            device=self.cfg.device,
        )

        # Insert activations directly into pre-allocated buffer
        # pbar = tqdm(total=n_batches_in_buffer, desc="Filling buffer")
        for refill_batch_idx_start in refill_iterator:
            refill_batch_tokens = self.get_batch_tokens()
            refill_activations = self.get_activations(refill_batch_tokens)
            new_buffer[refill_batch_idx_start : refill_batch_idx_start + batch_size] = (
                refill_activations
            )

            # pbar.update(1)

        new_buffer = new_buffer.reshape(-1, d_in)
        new_buffer = new_buffer[torch.randperm(new_buffer.shape[0])]

        return new_buffer

    def get_data_loader(
        self,
    ) -> DataLoader:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """

        batch_size = self.cfg.train_batch_size

        # 1. # create new buffer by mixing stored and new buffer
        mixing_buffer = torch.cat(
            [self.get_buffer(self.cfg.n_batches_in_buffer // 2), self.storage_buffer]
        )

        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        # 2.  put 50 % in storage
        self.storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # 3. put other 50 % in a dataloader
        dataloader = iter(
            DataLoader(
                mixing_buffer[mixing_buffer.shape[0] // 2 :],
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
            self.dataloader = self.get_data_loader()
            return next(self.dataloader)
