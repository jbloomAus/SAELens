from pathlib import Path

import pytest
from transformer_lens import HookedTransformer

from sae_lens.util import (
    extract_stop_at_layer_from_tlens_hook_name,
    get_special_token_ids,
    path_or_tmp_dir,
)


@pytest.mark.parametrize(
    "hook_name,expected_layer",
    [
        ("blocks.0.attn.hook_q", 1),
        ("blocks.12.attn.hook_k", 13),
        ("blocks.999.attn.hook_v", 1000),
        ("blocks.42.mlp.hook_pre", 43),
    ],
)
def test_extract_stop_at_layer_from_tlens_hook_name_valid(
    hook_name: str, expected_layer: int
):
    assert extract_stop_at_layer_from_tlens_hook_name(hook_name) == expected_layer


@pytest.mark.parametrize(
    "hook_name",
    [
        "blocks.attn.hook_q",  # missing layer number
        "blocks..attn.hook_q",  # empty layer number
        "hook_q",  # no layer info
        "blocks.abc.attn.hook_q",  # non-numeric layer
        "",  # empty string
    ],
)
def test_extract_stop_at_layer_from_tlens_hook_name_invalid(hook_name: str):
    assert extract_stop_at_layer_from_tlens_hook_name(hook_name) is None


def test_path_or_tmp_dir_with_none():
    with path_or_tmp_dir(None) as path:
        assert isinstance(path, Path)
        assert path.exists()
        assert path.is_dir()
        # Create a test file to verify the directory works
        test_file = path / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
    # Directory should be cleaned up after context exit
    assert not path.exists()


def test_path_or_tmp_dir_with_path(tmp_path: Path):
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    with path_or_tmp_dir(test_dir) as path:
        assert isinstance(path, Path)
        assert path == test_dir
        assert path.exists()
        assert path.is_dir()
    # Directory should still exist after context exit (not cleaned up)
    assert test_dir.exists()


def test_path_or_tmp_dir_with_string_path(tmp_path: Path):
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    with path_or_tmp_dir(str(test_dir)) as path:
        assert isinstance(path, Path)
        assert path == test_dir
        assert path.exists()
        assert path.is_dir()
    # Directory should still exist after context exit (not cleaned up)
    assert test_dir.exists()


def test_get_special_token_ids():
    # Create a mock tokenizer with some special tokens
    class MockTokenizer:
        def __init__(self):
            self.bos_token_id = 1
            self.eos_token_id = 2
            self.pad_token_id = 3
            self.unk_token_id = None  # Test handling of None values
            self.special_tokens_map = {
                "additional_special_tokens": ["<extra_0>", "<extra_1>"],
                "mask_token": "<mask>",
            }

        def convert_tokens_to_ids(self, token: str) -> int:
            token_map = {"<extra_0>": 4, "<extra_1>": 5, "<mask>": 6}
            return token_map[token]

    tokenizer = MockTokenizer()
    special_tokens = get_special_token_ids(tokenizer)  # type: ignore

    # Check that all expected token IDs are present
    assert set(special_tokens) == {1, 2, 3, 4, 5, 6}

    # Check that None values are properly handled
    assert None not in special_tokens


def test_get_special_token_ids_works_with_real_models(ts_model: HookedTransformer):
    special_tokens = get_special_token_ids(ts_model.tokenizer)  # type: ignore
    assert special_tokens == [50256]
