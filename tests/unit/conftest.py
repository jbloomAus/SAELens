import pytest
from transformer_lens import HookedTransformer

from tests.unit.helpers import TINYSTORIES_MODEL


@pytest.fixture
def ts_model():
    return HookedTransformer.from_pretrained(TINYSTORIES_MODEL, device="cpu")
