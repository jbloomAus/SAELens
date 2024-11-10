import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from sae_lens.sae import SAE
from tests.unit.helpers import TINYSTORIES_MODEL, load_model_cached

torch.set_grad_enabled(True)


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


# we started running out of space in CI, try cleaing up tmp paths after each test
@pytest.fixture(autouse=True)
def cleanup_tmp_path(tmp_path: Path):
    yield  # This line allows the test to run and use tmp_path
    # After the test is done, clean up the directory
    for item in tmp_path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


@pytest.fixture
def gpt2_res_jb_l4_sae() -> SAE:
    return SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.4.hook_resid_pre",
        device="cpu",
    )[0]
