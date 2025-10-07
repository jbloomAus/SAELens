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


def get_dir_size(path: Path) -> int:
    """Get the size of a directory in bytes."""
    if not path.exists():
        return 0

    try:
        result = subprocess.run(
            ["du", "-s", str(path)],
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


def get_tracked_disk_usage() -> dict[str, int]:
    """Get disk usage for all tracked directories."""
    workspace = Path.cwd()
    return {
        "cache": get_dir_size(Path.home() / ".cache"),
        "tmp": get_dir_size(Path("/tmp")),
        "workspace": get_dir_size(workspace),
    }


DISK_USAGE_TRACKING = os.environ.get("TRACK_DISK_USAGE", "0") == "1"
DISK_USAGE_REPORT: list[tuple[str, dict[str, int]]] = []


@pytest.fixture(autouse=True)
def track_disk_usage(request: pytest.FixtureRequest):
    """Track disk usage before and after each test."""
    if not DISK_USAGE_TRACKING:
        yield
        return

    initial_sizes = get_tracked_disk_usage()
    yield
    final_sizes = get_tracked_disk_usage()

    # Calculate increases for each tracked directory
    increases = {
        name: final_sizes[name] - initial_sizes[name]
        for name in initial_sizes
        if final_sizes[name] - initial_sizes[name] > 0
    }

    if increases:
        test_name = request.node.nodeid
        DISK_USAGE_REPORT.append((test_name, increases))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int):  # noqa
    """Print disk usage report at the end of the test session."""
    if not DISK_USAGE_TRACKING:
        return

    # Use pytest's terminal writer to bypass output capture
    terminal_reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is None:
        return

    terminal_reporter.write_sep(
        "=", "DISK USAGE REPORT (Tracked: cache, tmp, workspace)"
    )

    if not DISK_USAGE_REPORT:
        terminal_reporter.write_line("No disk usage increases detected.")
        terminal_reporter.write_line(
            "(Directories may already be populated or tracking failed)"
        )
    else:
        # Sort by total increase across all directories
        sorted_report = sorted(
            DISK_USAGE_REPORT,
            key=lambda x: sum(x[1].values()),
            reverse=True,
        )

        for test_name, increases in sorted_report[:20]:
            total_mb = sum(increases.values()) / (1024 * 1024)
            details = ", ".join(
                f"{name}: {size / (1024 * 1024):.1f}MB"
                for name, size in increases.items()
            )
            terminal_reporter.write_line(f"{total_mb:>8.2f} MB - {test_name}")
            terminal_reporter.write_line(f"            [{details}]")

        # Show totals by directory
        terminal_reporter.write_line("")
        total_by_dir: dict[str, int] = {}
        for _, increases in DISK_USAGE_REPORT:
            for name, size in increases.items():
                total_by_dir[name] = total_by_dir.get(name, 0) + size

        terminal_reporter.write_line("Total increases by directory:")
        for name, total in sorted(
            total_by_dir.items(), key=lambda x: x[1], reverse=True
        ):
            terminal_reporter.write_line(f"  {name}: {total / (1024 * 1024):.2f} MB")

    terminal_reporter.write_sep("=", "End of Disk Usage Report")


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
