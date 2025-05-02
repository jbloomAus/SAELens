from dataclasses import asdict, fields, is_dataclass
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")


def copy_and_remove_keys(d: dict[K, V], keys: list[K]) -> dict[K, V]:
    return {k: v for k, v in d.items() if k not in keys}


def filter_valid_dataclass_fields(
    source: dict[str, V] | object, destination: object | type
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
    return {key: val for key, val in source_dict.items() if key in valid_field_names}
