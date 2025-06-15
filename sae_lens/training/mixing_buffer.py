from collections.abc import Iterator

import torch


@torch.no_grad()
def mixing_buffer(
    buffer_size: int,
    batch_size: int,
    activations_loader: Iterator[torch.Tensor],
) -> Iterator[torch.Tensor]:
    """
    A generator that maintains a mix of old and new activations for better training.
    It stores half of the activations and mixes them with new ones to create batches.

    Args:
        buffer_size: Total size of the buffer (will store buffer_size/2 activations)
        batch_size: Size of batches to return
        activations_loader: Iterator providing new activations

    Yields:
        Batches of activations of shape (batch_size, *activation_dims)
    """

    if buffer_size < batch_size:
        raise ValueError("Buffer size must be greater than batch size")

    storage_buffer: torch.Tensor | None = None

    for new_activations in activations_loader:
        storage_buffer = (
            new_activations
            if storage_buffer is None
            else torch.cat([storage_buffer, new_activations], dim=0)
        )

        if storage_buffer.shape[0] >= buffer_size:
            # Shuffle
            storage_buffer = storage_buffer[torch.randperm(storage_buffer.shape[0])]

            num_serving_batches = max(1, storage_buffer.shape[0] // (2 * batch_size))
            serving_cutoff = num_serving_batches * batch_size
            serving_buffer = storage_buffer[:serving_cutoff]
            storage_buffer = storage_buffer[serving_cutoff:]

            # Yield batches from the serving_buffer
            for batch_idx in range(num_serving_batches):
                yield serving_buffer[
                    batch_idx * batch_size : (batch_idx + 1) * batch_size
                ]

    # If there are any remaining activations, yield them
    if storage_buffer is not None:
        remaining_batches = storage_buffer.shape[0] // batch_size
        for i in range(remaining_batches):
            yield storage_buffer[i * batch_size : (i + 1) * batch_size]
