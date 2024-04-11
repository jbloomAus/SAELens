import pytest

from tests.unit.helpers import TINYSTORIES_MODEL, load_model_cached


@pytest.fixture
def ts_model():
    return load_model_cached(TINYSTORIES_MODEL)
