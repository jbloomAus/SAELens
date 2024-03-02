import pytest
from transformer_lens import HookedTransformer

from tests.unit.helpers import TEST_MODEL


@pytest.fixture
def model():
    return HookedTransformer.from_pretrained(TEST_MODEL, device="cpu")
