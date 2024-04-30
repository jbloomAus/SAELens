from pathlib import Path

import pytest
import torch

from sae_lens.training.cache_activations_runner import cache_activations_runner
from tests.unit.helpers import TINYSTORIES_MODEL, build_cache_runner_cfg


def test_cache_activations_runner_outputs_look_correct(tmp_path: Path):
    target_dir = tmp_path / "activations"
    cfg = build_cache_runner_cfg(
        model_name=TINYSTORIES_MODEL,
        cached_activations_path=target_dir,
        d_in=64,  # tinystories has a 64-dim residual stream
        batch_size=4,
        context_size=6,
        n_batches_in_buffer=2,
        training_tokens=1_000,
    )
    cache_activations_runner(cfg)

    expected_tokens_per_buffer = 48  # 4 * 6 * 2
    expected_num_buffers = 21  # 1000 / 48
    saved_buffers = list(target_dir.glob("*.pt"))
    assert len(saved_buffers) == expected_num_buffers
    for buffer_path in saved_buffers:
        buffer = torch.load(buffer_path)
        assert buffer.shape == (expected_tokens_per_buffer, 1, 64)  # 64 is d_in


def test_cache_activations_runner_errors_if_cache_dir_is_nonempty(tmp_path: Path):
    target_dir = tmp_path / "activations"
    cfg = build_cache_runner_cfg(cached_activations_path=target_dir)
    target_dir.mkdir()
    (target_dir / "some_file.txt").touch()
    with pytest.raises(Exception):
        cache_activations_runner(cfg)
