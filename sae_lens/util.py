import re
from dataclasses import asdict, fields, is_dataclass
from typing import Sequence, TypeVar

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
