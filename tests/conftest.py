import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from sae_lens.saes.sae import SAE
from sae_lens.saes.standard_sae import StandardSAEConfig
from tests.helpers import TINYSTORIES_MODEL, load_model_cached

torch.set_grad_enabled(True)

# sparsify's triton implementation breaks in CI, so just disable it
os.environ["SPARSIFY_DISABLE_TRITON"] = "1"


def get_hf_cache_size() -> int:
    """Get the size of the huggingface cache directory in bytes."""
    cache_dir = Path.home() / ".cache" / "huggingface"
    if not cache_dir.exists():
        return 0

    try:
        result = subprocess.run(
            ["du", "-s", str(cache_dir)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return 0

        size_kb = int(result.stdout.split()[0])
        return size_kb * 1024
    except Exception:
        return 0


DISK_USAGE_TRACKING = os.environ.get("TRACK_DISK_USAGE", "0") == "1"
DISK_USAGE_REPORT: list[tuple[str, int]] = []


@pytest.fixture(autouse=True)
def track_disk_usage(request: pytest.FixtureRequest):
    """Track disk usage before and after each test."""
    if not DISK_USAGE_TRACKING:
        yield
        return

    initial_size = get_hf_cache_size()
    yield
    final_size = get_hf_cache_size()

    increase = final_size - initial_size
    if increase > 0:
        test_name = request.node.nodeid
        DISK_USAGE_REPORT.append((test_name, increase))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):  # noqa
    """Print disk usage report at the end of the test session."""
    if not DISK_USAGE_TRACKING or not DISK_USAGE_REPORT:
        return

    # Use pytest's terminal writer to bypass output capture
    terminal_reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is None:
        return

    terminal_reporter.write_sep("=", "DISK USAGE REPORT (HuggingFace Cache Increases)")

    sorted_report = sorted(DISK_USAGE_REPORT, key=lambda x: x[1], reverse=True)

    for test_name, increase in sorted_report[:20]:
        mb = increase / (1024 * 1024)
        terminal_reporter.write_line(f"{mb:>8.2f} MB - {test_name}")

    total_increase = sum(increase for _, increase in DISK_USAGE_REPORT)
    terminal_reporter.write_sep(
        "=", f"Total cache increase: {total_increase / (1024 * 1024):.2f} MB"
    )


@pytest.fixture(autouse=True)
def set_grad_enabled():
    # somehow grad is getting disabled before some tests
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
def gpt2_res_jb_l4_sae() -> SAE[StandardSAEConfig]:
    return SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.4.hook_resid_pre",
        device="cpu",
    )
