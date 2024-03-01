import re
from typing import Any, Dict, Tuple

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor

Arr = np.ndarray[Any, Any]


def k_largest_indices(
    x: Float[Tensor, "rows cols"],
    k: int,
    largest: bool = True,
    buffer: Tuple[int, int] = (5, 5),
) -> Int[Tensor, "k 2"]:
    """w
    Given a 2D array, returns the indices of the top or bottom `k` elements.

    Also has a `buffer` argument, which makes sure we don't pick too close to the left/right of sequence. If `buffer`
    is (5, 5), that means we shouldn't be allowed to pick the first or last 5 sequence positions, because we'll need
    to append them to the left/right of the sequence. We should only be allowed from [5:-5] in this case.
    """
    x = x[:, buffer[0] : -buffer[1]]
    indices = x.flatten().topk(k=k, largest=largest).indices
    rows = indices // x.size(1)
    cols = indices % x.size(1) + buffer[0]
    return torch.stack((rows, cols), dim=1)


def sample_unique_indices(large_number: int, small_number: int):
    """Samples a small number of unique indices from a large number of indices."""
    weights = torch.ones(large_number)  # Equal weights for all indices
    sampled_indices = torch.multinomial(weights, small_number, replacement=False)
    return sampled_indices


def random_range_indices(
    x: Float[Tensor, "batch seq"],
    bounds: Tuple[float | Tensor, float | Tensor],
    k: int,
    buffer: Tuple[int, int] = (5, 5),
) -> Int[Tensor, "k 2"]:
    """
    Given a 2D array, returns the indices of `k` elements whose values are in the range `bounds`.
    Will return fewer than `k` values if there aren't enough values in the range.

    Also has a `buffer` argument, which makes sure we don't pick too close to the left/right of sequence.
    """
    # Limit x, because our indices (bolded words) shouldn't be too close to the left/right of sequence
    x = x[:, buffer[0] : -buffer[1]]

    # Creat a mask for where x is in range, and get the indices as a tensor of shape (k, 2)
    mask = (bounds[0] <= x) & (x <= bounds[1])
    indices = torch.stack(torch.where(mask), dim=-1)

    # If we have more indices than we need, randomly select k of them
    if len(indices) > k:
        indices = indices[sample_unique_indices(len(indices), k)]

    # Adjust indices to account for the buffer
    return indices + torch.tensor([0, buffer[0]]).to(indices.device)


# # Example, where it'll pick the elements from the end of this 2D tensor, working backwards
# x = torch.tensor([[1, 2, 3], [4, 5, 6]])
# k = 3
# print(k_largest_indices(x, k))  # Output: tensor([[1, 2], [1, 1], [1, 0]])

# # Example, where it'll pick one of (0.2, 0.3) cause they're the only ones within range
# x = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
# bounds = (0.15, 0.35)
# k = 1
# print(random_range_indices(x, bounds, k))


def to_str_tokens(
    vocab_dict: Dict[int, str], tokens: int | torch.Tensor | np.ndarray[Any, Any]
) -> Any:
    """
    If tokens is 1D, does the same thing as model.to_str_tokens.
    If tokens is 2D or 3D, it flattens, does this thing, then reshapes.

    Also, makes sure that line breaks are replaced with their repr.
    """
    if isinstance(tokens, int):
        return vocab_dict[tokens]

    assert tokens.ndim <= 3

    # Get flattened list of tokens
    str_tokens = [vocab_dict[t] for t in tokens.flatten().tolist()]

    # Replace line breaks with things that will appear as the literal '\n' in HTML
    str_tokens = [s.replace("\n", "&bsol;n") for s in str_tokens]
    # str_tokens = [s.replace(" ", "&nbsp;") for s in str_tokens]

    # Reshape
    return reshape(str_tokens, tokens.shape)


def reshape(my_list: list[Any], shape: tuple[int, ...]):
    assert np.prod(shape) == len(my_list), "Shape is not compatible with list size"
    assert len(shape) in [1, 2, 3], "Only shapes of length 1, 2, or 3 are supported"

    if len(shape) == 1:
        return my_list

    it = iter(my_list)
    if len(shape) == 2:
        return [[next(it) for _ in range(shape[1])] for _ in range(shape[0])]

    return [
        [[next(it) for _ in range(shape[2])] for _ in range(shape[1])]
        for _ in range(shape[0])
    ]


class TopK:
    """
    Wrapper around the object returned by torch.topk, which has the following 3 advantages:

    > friendlier to type annotation
    > easy device moving, without having to do it separately for values & indices
    > easy indexing, without having to do it separately for values & indices
    > other classic tensor operations, like .ndim, .shape, etc. work as expected

    We initialise with a topk object, which is treated as a tuple of (values, indices).
    """

    def __init__(self, obj: Tuple[Arr | Tensor, Arr | Tensor]):  # type: ignore
        self.values: Arr = (
            obj[0] if isinstance(obj[0], np.ndarray) else obj[0].detach().cpu().numpy()
        )
        self.indices: Arr = (
            obj[1] if isinstance(obj[1], np.ndarray) else obj[1].detach().cpu().numpy()
        )

    def __getitem__(self, item: Any):
        return TopK((self.values[item], self.indices[item]))

    def concat(self, other: "TopK"):
        """If self is empty, returns the other (so we can start w/ empty & concatenate consistently)."""
        if self.numel() == 0:
            return other
        else:
            return TopK(
                (
                    np.concatenate((self.values, other.values)),
                    np.concatenate((self.indices, other.indices)),
                )
            )

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def shape(self):
        return self.values.shape

    @property
    def size(self):
        return self.values.size

    def numel(self):
        return self.values.size


class Output:
    """So I can type annotate the output of transformer."""

    loss: Tensor
    logits: Tensor


def merge_lists(*lists: Any):
    return [item for sublist in lists for item in sublist]


def extract_and_remove_scripts(html_content: str) -> Tuple[str, str]:
    # Pattern to find <script>...</script> tags
    pattern = r"<script[^>]*>.*?</script>"

    # Find all script tags
    scripts = re.findall(pattern, html_content, re.DOTALL)

    # Remove script tags from the original content
    html_without_scripts = re.sub(pattern, "", html_content, flags=re.DOTALL)

    return "\n".join(scripts), html_without_scripts
