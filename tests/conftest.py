import gc
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _pytest.capture import CaptureFixture
from _pytest.fixtures import FixtureRequest

from sae_lens.saes.sae import SAE
from sae_lens.saes.standard_sae import StandardSAEConfig
from tests.helpers import TINYSTORIES_MODEL, load_model_cached

torch.set_grad_enabled(True)

# sparsify's triton implementation breaks in CI, so just disable it
os.environ["SPARSIFY_DISABLE_TRITON"] = "1"

# Limit memory usage in CI
if os.getenv("CI"):
    # Reduce PyTorch memory allocation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    # Enable more aggressive garbage collection
    gc.set_threshold(100, 5, 5)


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
    try:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
    except (OSError, PermissionError):
        # Fallback to individual file deletion if rmtree fails
        for item in tmp_path.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except (OSError, PermissionError):
                pass  # Skip files that can't be deleted


@pytest.fixture(autouse=True)
def force_gc_after_test(request: FixtureRequest, capfd: CaptureFixture[Any]) -> Any:
    """Force garbage collection after each test to free memory."""
    yield
    # Force garbage collection and clear PyTorch cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print disk space after each test (only in CI to avoid spam in local dev)
    if os.getenv("CI"):
        test_name = request.node.name

        # Capture disk usage info
        try:
            disk_info = subprocess.run(
                ["df", "-h"], capture_output=True, text=True
            ).stdout.split("\n")[:2]
            hf_cache = (
                subprocess.run(
                    ["du", "-sh", os.path.expanduser("~/.cache/huggingface")],
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                or "No HF cache"
            )
            tmp_usage = (
                subprocess.run(
                    ["du", "-sh", "/tmp"], capture_output=True, text=True
                ).stdout.strip()
                or "No /tmp"
            )

            # Get detailed breakdown of root filesystem usage
            root_breakdown = (
                subprocess.run(
                    [
                        "du",
                        "-h",
                        "--max-depth=1",
                        "/",
                        "--exclude=/proc",
                        "--exclude=/sys",
                        "--exclude=/dev",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                .stdout.strip()
                .split("\n")
            )

            # Get top largest directories
            largest_dirs = (
                subprocess.run(
                    ["du", "-h", "--max-depth=2", "/home", "/usr", "/var", "/opt"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                .stdout.strip()
                .split("\n")
            )

        except Exception as e:
            disk_info = ["Unable to get disk info"]
            hf_cache = "Unable to get HF cache info"
            tmp_usage = "Unable to get /tmp info"
            root_breakdown = [f"Error getting root breakdown: {e}"]
            largest_dirs = []

        # Use capfd to ensure output is shown
        with capfd.disabled():
            # ruff: noqa: T201 - Print statements needed for CI debugging
            print(f"\n=== Disk space after test: {test_name} ===", flush=True)  # noqa: T201
            for line in disk_info:
                if line.strip():
                    print(line, flush=True)  # noqa: T201
            print(f"HF Cache: {hf_cache}", flush=True)  # noqa: T201
            print(f"Tmp: {tmp_usage}", flush=True)  # noqa: T201

            print("\n--- Root filesystem breakdown ---", flush=True)  # noqa: T201
            for line in root_breakdown[:10]:  # Show top 10 directories
                if line.strip() and not line.endswith("/"):
                    print(line, flush=True)  # noqa: T201

            print("\n--- Largest subdirectories ---", flush=True)  # noqa: T201
            # Sort and show largest directories
            size_lines = []
            for line in largest_dirs:
                if line.strip() and "\t" in line:
                    size_lines.append(line)

            # Sort by size (rough sort by first character/number)
            size_lines.sort(key=lambda x: x.split()[0], reverse=True)
            for line in size_lines[:15]:  # Show top 15
                print(line, flush=True)  # noqa: T201

            print("=" * 50, flush=True)  # noqa: T201


@pytest.fixture
def gpt2_res_jb_l4_sae() -> SAE[StandardSAEConfig]:
    return SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.4.hook_resid_pre",
        device="cpu",
    )
