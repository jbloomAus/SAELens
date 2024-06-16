from typing import Generator, Iterator

import torch


def _add_tokens_to_batch(
    batch: torch.Tensor | None,
    tokens: torch.Tensor,
    context_size: int,
    is_start_of_sequence: bool,
    begin_batch_token_id: int | None = None,
    begin_sequence_token_id: int | None = None,
    sequence_separator_token_id: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_tokens = tokens
    # prepend the start of sequence token if needed
    if is_start_of_sequence and begin_sequence_token_id is not None:
        begin_sequence_token_id_tensor = torch.tensor(
            [begin_sequence_token_id], dtype=torch.long, device=tokens.device
        )
        if tokens[0] != begin_sequence_token_id_tensor:
            tokens = torch.concat([begin_sequence_token_id_tensor, tokens], dim=0)
    # We're at the start of a new batch
    if batch is None:
        # add the BOS token to the start if needed
        if begin_batch_token_id is not None:
            begin_batch_token_id_tensor = torch.tensor(
                [begin_batch_token_id], dtype=torch.long, device=tokens.device
            )
            if tokens[0] != begin_batch_token_id_tensor:
                tokens = torch.concat([begin_batch_token_id_tensor, tokens], dim=0)
            batch = tokens[:context_size]
        return tokens[:context_size], tokens[context_size:]
    # if we're concatting batches, add the separator token as needed
    if sequence_separator_token_id is not None:
        sequence_separator_token_id_tensor = torch.tensor(
            [sequence_separator_token_id], dtype=torch.long, device=tokens.device
        )
        if tokens[0] != sequence_separator_token_id_tensor:
            tokens = torch.concat([sequence_separator_token_id_tensor, tokens], dim=0)
    tokens_needed = context_size - batch.shape[0]
    batch = torch.concat([batch, tokens[:tokens_needed]])

    remaining_tokens = tokens[tokens_needed:]
    # it's possible we've prepending 2 tokens to original_tokens, but only removed 1
    # if so, we should only return the original tokens
    if len(remaining_tokens) > len(original_tokens):
        remaining_tokens = original_tokens
    return batch, remaining_tokens


@torch.no_grad()
def concat_and_batch_sequences(
    tokens_iterator: Iterator[torch.Tensor],
    context_size: int,
    begin_batch_token_id: int | None = None,
    begin_sequence_token_id: int | None = None,
    sequence_separator_token_id: int | None = None,
) -> Generator[torch.Tensor, None, None]:
    """
    Generator to concat token sequences together from the tokens_interator, yielding
    batches of size `context_size`.

    Args:
        tokens_iterator: An iterator which returns a 1D tensors of tokens
        context_size: Each batch will have this many tokens
        begin_batch_token_id: If provided, this token will be at position 0 of each batch
        begin_sequence_token_id: If provided, this token will be the first token of each sequence
        sequence_separator_token_id: If provided, this token will be inserted between concatenated sequences
        max_batches: If not provided, the iterator will be run to completion.
    """
    batch: torch.Tensor | None = None
    for tokens in tokens_iterator:
        assert (
            len(tokens.shape) == 1
        ), f"tokens.shape should be 1D but was {tokens.shape}"
        remaining_tokens = tokens
        is_start_of_sequence = True
        while len(remaining_tokens) > 0:
            batch, remaining_tokens = _add_tokens_to_batch(
                batch=batch,
                tokens=remaining_tokens,
                context_size=context_size,
                is_start_of_sequence=is_start_of_sequence,
                begin_batch_token_id=begin_batch_token_id,
                begin_sequence_token_id=begin_sequence_token_id,
                sequence_separator_token_id=sequence_separator_token_id,
            )
            is_start_of_sequence = False
            if batch.shape[0] == context_size:
                yield batch
                batch = None
