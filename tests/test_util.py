import pytest

from sae_lens.util import extract_stop_at_layer_from_tlens_hook_name


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
    assert extract_stop_at_layer_from_tlens_hook_name(hook_name) == -1
