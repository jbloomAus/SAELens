from dataclasses import fields, is_dataclass
from typing import TypeVar

K = TypeVar("K")
V = TypeVar("V")


def copy_and_remove_keys(d: dict[K, V], keys: list[K]) -> dict[K, V]:
    return {k: v for k, v in d.items() if k not in keys}


def filter_valid_dataclass_fields(
    d: dict[str, V], dataclass: object | type
) -> dict[str, V]:
    if not is_dataclass(dataclass):
        raise ValueError(f"{dataclass} is not a dataclass")
    valid_field_names = {field.name for field in fields(dataclass)}
    return {key: val for key, val in d.items() if key in valid_field_names}
