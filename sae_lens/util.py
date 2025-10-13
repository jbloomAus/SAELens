import re
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Sequence, TypeVar

from transformers import PreTrainedTokenizerBase

K = TypeVar("K")
V = TypeVar("V")


def filter_valid_dataclass_fields(
    source: dict[str, V] | object,
    destination: object | type,
    whitelist_fields: Sequence[str] | None = None,
) -> dict[str, V]:
    """Filter a source dict or dataclass instance to only include fields that are present in the destination dataclass."""

    if not is_dataclass(destination):
        raise ValueError(f"{destination} is not a dataclass")

    if is_dataclass(source) and not isinstance(source, type):
        source_dict = asdict(source)
    elif isinstance(source, dict):
        source_dict = source
    else:
        raise ValueError(f"{source} is not a dict or dataclass")

    valid_field_names = {field.name for field in fields(destination)}
    if whitelist_fields is not None:
        valid_field_names = valid_field_names.union(whitelist_fields)
    return {key: val for key, val in source_dict.items() if key in valid_field_names}


def extract_stop_at_layer_from_tlens_hook_name(hook_name: str) -> int | None:
    """Extract the stop_at layer from a HookedTransformer hook name.

    Returns None if the hook name is not a valid HookedTransformer hook name.
    """
    layer = extract_layer_from_tlens_hook_name(hook_name)
    return None if layer is None else layer + 1


def extract_layer_from_tlens_hook_name(hook_name: str) -> int | None:
    """Extract the layer from a HookedTransformer hook name.

    Returns None if the hook name is not a valid HookedTransformer hook name.
    """
    hook_match = re.search(r"\.(\d+)\.", hook_name)
    return None if hook_match is None else int(hook_match.group(1))


@contextmanager
def path_or_tmp_dir(path: str | Path | None):
    """Context manager that yields a concrete Path for path.

    - If path is None, creates a TemporaryDirectory and yields its Path.
      The directory is cleaned up on context exit.
    - Otherwise, yields Path(path) without creating or cleaning.
    """
    if path is None:
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)
    else:
        yield Path(path)


def get_special_token_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Get all special token IDs from a tokenizer."""
    special_tokens = set()

    # Get special tokens from tokenizer attributes
    for attr in dir(tokenizer):
        if attr.endswith("_token_id"):
            token_id = getattr(tokenizer, attr)
            if token_id is not None:
                special_tokens.add(token_id)

    # Get any additional special tokens from the tokenizer's special tokens map
    if hasattr(tokenizer, "special_tokens_map"):
        for token in tokenizer.special_tokens_map.values():
            if isinstance(token, str):
                token_id = tokenizer.convert_tokens_to_ids(token)  # type: ignore
                special_tokens.add(token_id)
            elif isinstance(token, list):
                for t in token:
                    token_id = tokenizer.convert_tokens_to_ids(t)  # type: ignore
                    special_tokens.add(token_id)

    return list(special_tokens)
