import dataclasses
import os
from pathlib import Path
from typing import Any, Tuple

import datasets
import pytest
import torch
from datasets import Dataset, load_dataset
from tqdm import trange
from transformer_lens import HookedTransformer

from sae_lens.cache_activations_runner import CacheActivationsRunner
from sae_lens.config import CacheActivationsRunnerConfig, LanguageModelSAERunnerConfig
from sae_lens.load_model import load_model
from sae_lens.training.activations_store import ActivationsStore


def _create_dataset(tmp_path: Path) -> Dataset:
    torch.manual_seed(42)

    model_name = "gelu-1l"
    hook_name = "blocks.0.hook_mlp_out"
    dataset_path = "chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests"
    batch_size = 1
    batches_in_buffer = 2
    context_size = 8
    num_buffers = 4

    train_batch_size_tokens = 32

    tokens_in_buffer = batches_in_buffer * batch_size * context_size
    num_tokens = tokens_in_buffer * num_buffers

    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path),
        model_name=model_name,
        hook_name=hook_name,
        dataset_path=dataset_path,
        training_tokens=num_tokens,
        shuffle=False,
        store_batch_size_prompts=batch_size,
        train_batch_size_tokens=train_batch_size_tokens,
        n_batches_in_buffer=batches_in_buffer,
        ###
        hook_layer=0,
        d_in=512,
        context_size=context_size,
        is_dataset_tokenized=True,
        prepend_bos=False,
        normalize_activations="none",
        device="cpu",
        seed=42,
        dtype="float32",
    )

    runner = CacheActivationsRunner(cfg)
    return runner.run()


# The way to run this with this command:
# poetry run py.test tests/unit/test_cache_activations_runner.py --profile-svg -s
def test_cache_activations_runner(tmp_path: Path):

    # total_training_steps = 20_000
    context_size = 8
    n_batches_in_buffer = 32
    store_batch_size = 1
    n_buffers = 3

    tokens_in_buffer = n_batches_in_buffer * store_batch_size * context_size
    total_training_tokens = n_buffers * tokens_in_buffer
    total_rows = store_batch_size * n_batches_in_buffer * n_buffers

    # better if we can look at the files (change tmp_path to a real path to look at the files)
    # tmp_path = os.path.join(os.path.dirname(__file__), "tmp")
    # tmp_path = Path("/Volumes/T7 Shield/activations/gelu_1l")
    # if os.path.exists(tmp_path):
    #     shutil.rmtree(tmp_path)

    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path),
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        d_in=512,
        dataset_path="chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests",
        context_size=context_size,  # Speed things up.
        is_dataset_tokenized=True,
        prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=32,
        # Loss Function
        ## Reconstruction Coefficient.
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=store_batch_size,
        normalize_activations="none",
        # Misc
        device="cpu",
        seed=42,
        dtype="float16",
    )

    # look at the next cell to see some instruction for what to do while this is running.
    dataset = CacheActivationsRunner(cfg).run()
    assert len(dataset) == total_rows
    assert dataset.num_columns == 1 and dataset.column_names == [cfg.hook_name]

    features = dataset.features
    assert isinstance(features[cfg.hook_name], datasets.Array2D)
    assert features[cfg.hook_name].shape == (context_size, cfg.d_in)


def test_load_cached_activations(tmp_path: Path):

    # total_training_steps = 20_000
    context_size = 8
    n_batches_in_buffer = 4
    store_batch_size = 1
    n_buffers = 4

    tokens_in_buffer = n_batches_in_buffer * store_batch_size * context_size
    total_training_tokens = n_buffers * tokens_in_buffer

    _create_dataset(tmp_path)

    cfg = LanguageModelSAERunnerConfig(
        cached_activations_path=str(tmp_path),
        use_cached_activations=True,
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        d_in=512,
        dataset_path="chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests",
        context_size=context_size,
        is_dataset_tokenized=True,
        prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=total_training_tokens // 2,
        # Loss Function
        ## Reconstruction Coefficient.
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=store_batch_size,
        normalize_activations="none",
        # shuffle_every_n_buffers=2,
        # n_shuffles_with_last_section=1,
        # n_shuffles_in_entire_dir=1,
        # n_shuffles_final=1,
        # Misc
        device="cpu",
        seed=42,
        dtype="float16",
    )

    model = HookedTransformer.from_pretrained(cfg.model_name)
    activations_store = ActivationsStore.from_config(model, cfg)

    for _ in range(n_buffers):
        buffer = activations_store.get_buffer(
            cfg.n_batches_in_buffer
        )  # Adjusted to use n_batches_in_buffer
        assert buffer.shape == (
            cfg.n_batches_in_buffer * cfg.store_batch_size_prompts * cfg.context_size,
            1,
            cfg.d_in,
        )


def test_activations_store_refreshes_dataset_when_it_runs_out(tmp_path: Path):

    context_size = 8
    n_batches_in_buffer = 4
    store_batch_size = 1
    total_training_steps = 4
    batch_size = 4
    total_training_tokens = total_training_steps * batch_size

    _create_dataset(tmp_path)

    cfg = LanguageModelSAERunnerConfig(
        cached_activations_path=str(tmp_path),
        use_cached_activations=True,
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        d_in=512,
        dataset_path="",
        context_size=context_size,
        is_dataset_tokenized=True,
        prepend_bos=True,
        training_tokens=total_training_tokens // 2,
        train_batch_size_tokens=8,
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=store_batch_size,
        normalize_activations="none",
        device="cpu",
        seed=42,
        dtype="float16",
    )

    class MockModel:
        def to_tokens(self, *args: Tuple[Any, ...], **kwargs: Any) -> torch.Tensor:
            return torch.ones(context_size)

        @property
        def W_E(self) -> torch.Tensor:
            return torch.ones(16, 16)

        @property
        def cfg(self) -> LanguageModelSAERunnerConfig:
            return cfg

    dataset = Dataset.from_list(
        [
            {"text": "hello world1"},
        ]
        * 64
    )

    model = MockModel()
    activations_store = ActivationsStore.from_config(
        model,  # type: ignore
        cfg,
        override_dataset=dataset,
    )
    for _ in range(16):
        _ = activations_store.get_batch_tokens(batch_size, raise_at_epoch_end=True)

    # assert a stop iteration is raised when we do one more get_batch_tokens

    pytest.raises(
        StopIteration,
        activations_store.get_batch_tokens,
        batch_size,
        raise_at_epoch_end=True,
    )

    # no errors are ever raised if we do not ask for raise_at_epoch_end
    for _ in range(32):
        _ = activations_store.get_batch_tokens(batch_size, raise_at_epoch_end=False)


def test_compare_cached_activations_end_to_end_with_ground_truth(tmp_path: Path):
    """
    Creates activations using CacheActivationsRunner and compares them with ground truth
    model.run_with_cache
    """

    torch.manual_seed(42)

    model_name = "gelu-1l"
    hook_name = "blocks.0.hook_mlp_out"
    dataset_path = "chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests"
    batch_size = 8
    batches_in_buffer = 4
    context_size = 8
    num_buffers = 4

    train_batch_size_tokens = 8

    tokens_in_buffer = batches_in_buffer * batch_size * context_size
    num_tokens = tokens_in_buffer * num_buffers

    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path),
        model_name=model_name,
        hook_name=hook_name,
        dataset_path=dataset_path,
        training_tokens=num_tokens,
        shuffle=False,
        store_batch_size_prompts=batch_size,
        train_batch_size_tokens=train_batch_size_tokens,
        n_batches_in_buffer=batches_in_buffer,
        ###
        hook_layer=0,
        d_in=512,
        context_size=context_size,
        is_dataset_tokenized=True,
        prepend_bos=False,
        normalize_activations="none",
        device="cpu",
        seed=42,
        dtype="float32",
    )

    runner = CacheActivationsRunner(cfg)
    activation_dataset = runner.run()
    activation_dataset.set_format("torch", device=cfg.device)
    dataset_acts: torch.Tensor = activation_dataset[cfg.hook_name]  # type: ignore

    model = HookedTransformer.from_pretrained(model_name, device=cfg.device)
    token_dataset: Dataset = load_dataset(dataset_path, split=f"train[:{num_tokens}]")  # type: ignore
    token_dataset.set_format("torch", device=cfg.device)

    total_rows = batch_size * batches_in_buffer * num_buffers

    ground_truth_acts = []
    for i in trange(0, total_rows, batch_size):
        tokens = token_dataset[i : i + batch_size]["input_ids"][:, :context_size]
        _, layerwise_activations = model.run_with_cache(
            tokens,
            names_filter=[cfg.hook_name],
            stop_at_layer=cfg.hook_layer + 1,
            **cfg.model_kwargs,
        )
        acts = layerwise_activations[cfg.hook_name]
        ground_truth_acts.append(acts)

    ground_truth_acts = torch.cat(ground_truth_acts, dim=0)

    assert torch.allclose(ground_truth_acts, dataset_acts, rtol=1e-3, atol=5e-2)


def test_load_activations_store_with_nonexistent_dataset(tmp_path: Path):
    cfg = CacheActivationsRunnerConfig(
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        dataset_path="chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests",
        cached_activations_path=str(tmp_path),
        context_size=16,
    )

    model = load_model(
        model_class_name=cfg.model_class_name,
        model_name=cfg.model_name,
        device=cfg.device,
        model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
    )

    # Attempt to load from a non-existent dataset
    with pytest.raises(
        FileNotFoundError,
        match="is neither a `Dataset` directory nor a `DatasetDict` directory.",
    ):
        ActivationsStore.from_config(model, cfg)


def test_cache_activations_runner_with_nonempty_directory(tmp_path: Path):
    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path),
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        dataset_path="chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests",
        context_size=16,
    )
    runner = CacheActivationsRunner(cfg)

    # Create a file to make the directory non-empty
    with open(tmp_path / "some_file.txt", "w") as f:
        f.write("test")

    with pytest.raises(
        Exception, match="is not empty. Please delete it or specify a different path."
    ):
        runner.run()

    # Clean up
    os.remove(tmp_path / "some_file.txt")


def test_cache_activations_runner_with_incorrect_d_in(tmp_path: Path):
    d_in = 512
    context_size = 8
    n_batches_in_buffer = 4
    batch_size = 8
    num_buffers = 4
    num_tokens = batch_size * context_size * n_batches_in_buffer * num_buffers

    correct_cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path),
        d_in=d_in,
        context_size=context_size,
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        dataset_path="chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests",
        training_tokens=num_tokens,
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=batch_size,
        normalize_activations="none",
        device="cpu",
        seed=42,
        dtype="float32",
    )

    # d_in different from hook
    wrong_d_in_cfg = CacheActivationsRunnerConfig(
        **dataclasses.asdict(correct_cfg),
    )
    wrong_d_in_cfg.d_in = 513

    runner = CacheActivationsRunner(wrong_d_in_cfg)
    with pytest.raises(
        RuntimeError,
        match=r"The expanded size of the tensor \(513\) must match the existing size \(512\) at non-singleton dimension 2.",
    ):
        runner.run()


def test_cache_activations_runner_load_dataset_with_incorrect_config(tmp_path: Path):
    d_in = 512
    context_size = 16
    n_batches_in_buffer = 4
    batch_size = 8
    num_buffers = 2
    num_tokens = batch_size * context_size * n_batches_in_buffer * num_buffers

    correct_cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path),
        d_in=d_in,
        context_size=context_size,
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        dataset_path="chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests",
        training_tokens=num_tokens,
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=batch_size,
        normalize_activations="none",
        device="cpu",
        seed=42,
        dtype="float32",
    )

    # Run with correct configuration first
    CacheActivationsRunner(correct_cfg).run()

    ###

    # Context size different from dataset
    wrong_context_size_cfg = CacheActivationsRunnerConfig(
        **dataclasses.asdict(correct_cfg),
    )
    wrong_context_size_cfg.context_size = 13
    wrong_context_size_cfg.new_cached_activations_path = None
    wrong_context_size_cfg.cached_activations_path = str(tmp_path)

    with pytest.raises(
        ValueError,
        match=r"Given dataset of shape \(16, 512\) does not match context_size \(13\) and d_in \(512\)",
    ):
        CacheActivationsRunner(wrong_context_size_cfg).run()

    # d_in different from dataset
    wrong_d_in_cfg = CacheActivationsRunnerConfig(
        **dataclasses.asdict(correct_cfg),
    )
    wrong_d_in_cfg.d_in = 513
    wrong_d_in_cfg.new_cached_activations_path = None
    wrong_d_in_cfg.cached_activations_path = str(tmp_path)

    with pytest.raises(
        ValueError,
        match=r"Given dataset of shape \(16, 512\) does not match context_size \(16\) and d_in \(513\)",
    ):
        CacheActivationsRunner(wrong_d_in_cfg).run()

    # Incorrect hook_name
    wrong_hook_cfg = CacheActivationsRunnerConfig(
        **dataclasses.asdict(correct_cfg),
    )
    wrong_hook_cfg.hook_name = "blocks.1.hook_mlp_out"
    wrong_hook_cfg.new_cached_activations_path = None
    wrong_hook_cfg.cached_activations_path = str(tmp_path)

    with pytest.raises(
        ValueError,
        match=r"Columns \['blocks.1.hook_mlp_out'\] not in the dataset. Current columns in the dataset: \['blocks.0.hook_mlp_out'\]",
    ):
        CacheActivationsRunner(wrong_hook_cfg).run()


def test_cache_activations_runner_with_valid_seqpos(tmp_path: Path):
    context_size = 16
    seqpos_slice = (3, -3)
    training_context_size = len(range(context_size)[slice(*seqpos_slice)])
    n_batches_in_buffer = 4
    store_batch_size = 1
    n_buffers = 3

    tokens_in_buffer = n_batches_in_buffer * store_batch_size * training_context_size
    total_training_tokens = n_buffers * tokens_in_buffer

    cfg = CacheActivationsRunnerConfig(
        new_cached_activations_path=str(tmp_path),
        d_in=512,
        context_size=context_size,
        model_name="gelu-1l",
        hook_name="blocks.0.hook_mlp_out",
        dataset_path="chanind/c4-10k-mini-tokenized-16-ctx-gelu-1l-tests",
        training_tokens=total_training_tokens,
        n_batches_in_buffer=n_batches_in_buffer,
        store_batch_size_prompts=store_batch_size,
        normalize_activations="none",
        device="cpu",
        seed=42,
        dtype="float32",
        seqpos_slice=seqpos_slice,
    )

    runner = CacheActivationsRunner(cfg)

    activation_dataset = runner.run()
    activation_dataset.set_format("torch", device=cfg.device)
    dataset_acts: torch.Tensor = activation_dataset[cfg.hook_name]  # type: ignore

    assert os.path.exists(tmp_path)

    # assert that there are n_buffer files in the directory.
    assert len(os.listdir(tmp_path)) == n_buffers

    assert len(dataset_acts) == n_buffers * n_batches_in_buffer
    for act in dataset_acts:
        # should be 16 - 3 - 3 = 10
        assert act.shape == (10, cfg.d_in)
