import random

import numpy as np
import pytest
import torch

from .helpers import TINYSTORIES_MODEL, load_model_cached


@pytest.fixture(autouse=True)
def reproducibility():
    """Apply various mechanisms to try to prevent nondeterminism in test runs."""
    # I have not in general attempted to verify that the below are necessary
    # for reproducibility, only that they are likely to help and unlikely to
    # hurt.
    # https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    seed = 0x1234_5678_9ABC_DEF0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Python native RNG; docs don't give any limitations on seed range
    random.seed(seed)
    # this is a "legacy" method that operates on a global RandomState
    # sounds like the argument must be in [0, 2**32)
    np.random.seed(seed & 0xFFFF_FFFF)


@pytest.fixture
def ts_model():
    return load_model_cached(TINYSTORIES_MODEL)
