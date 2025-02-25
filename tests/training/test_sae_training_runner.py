import argparse
import json
import os
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae import SAE
from sae_lens.sae_training_runner import (
    SAETrainingRunner,
    _load_checkpoint_state,
    _parse_cfg_args,
    _run_cli,
)
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.training_sae import TrainingSAE
from tests.helpers import (
    TINYSTORIES_DATASET,
    TINYSTORIES_MODEL,
    build_sae_cfg,
    load_model_cached,
)


@pytest.fixture
def cfg(tmp_path: Path):
    return build_sae_cfg(
        d_in=64, d_sae=128, hook_layer=0, checkpoint_path=str(tmp_path)
    )


@pytest.fixture
def model():
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def activation_store(model: HookedTransformer, cfg: LanguageModelSAERunnerConfig):
    return ActivationsStore.from_config(
        model, cfg, override_dataset=Dataset.from_list([{"text": "hello world"}] * 2000)
    )


@pytest.fixture
def training_sae(cfg: LanguageModelSAERunnerConfig):
    return TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())


@pytest.fixture
def trainer(
    cfg: LanguageModelSAERunnerConfig,
    training_sae: TrainingSAE,
    model: HookedTransformer,
    activation_store: ActivationsStore,
):
    return SAETrainer(
        model=model,
        sae=training_sae,
        activation_store=activation_store,
        save_checkpoint_fn=lambda *args, **kwargs: None,  # noqa: ARG005
        cfg=cfg,
    )


@pytest.fixture
def training_runner(
    cfg: LanguageModelSAERunnerConfig,
):
    return SAETrainingRunner(cfg)


def test_save_checkpoint(training_runner: SAETrainingRunner, trainer: SAETrainer):
    training_runner.save_checkpoint(
        trainer=trainer,
        checkpoint_name="test",
    )

    contents = os.listdir(training_runner.cfg.checkpoint_path + "/test")

    assert "sae_weights.safetensors" in contents
    assert "sparsity.safetensors" in contents
    assert "cfg.json" in contents

    sae = SAE.load_from_pretrained(training_runner.cfg.checkpoint_path + "/test")

    assert isinstance(sae, SAE)


def test_training_runner_works_with_from_pretrained_path(
    trainer: SAETrainer,
    cfg: LanguageModelSAERunnerConfig,
):
    SAETrainingRunner.save_checkpoint(
        trainer=trainer,
        checkpoint_name="test",
    )

    cfg.from_pretrained_path = trainer.cfg.checkpoint_path + "/test"
    loaded_runner = SAETrainingRunner(cfg)

    # the loaded runner should load the pretrained SAE
    orig_sae = trainer.sae
    new_sae = loaded_runner.sae

    assert orig_sae.cfg.to_dict() == new_sae.cfg.to_dict()
    assert torch.allclose(orig_sae.W_dec, new_sae.W_dec)
    assert torch.allclose(orig_sae.W_enc, new_sae.W_enc)
    assert torch.allclose(orig_sae.b_enc, new_sae.b_enc)
    assert torch.allclose(orig_sae.b_dec, new_sae.b_dec)


def test_parse_cfg_args_prints_help_if_no_args():
    args = []
    with pytest.raises(SystemExit):
        _parse_cfg_args(args)


def test_parse_cfg_args_override():
    args = [
        "--model_name",
        "test-model",
        "--d_in",
        "1024",
        "--d_sae",
        "4096",
        "--activation_fn",
        "tanh-relu",
        "--normalize_sae_decoder",
        "False",
        "--dataset_path",
        "my/dataset",
    ]
    cfg, resume_path = _parse_cfg_args(args)

    assert cfg.model_name == "test-model"
    assert cfg.d_in == 1024
    assert cfg.d_sae == 4096
    assert cfg.activation_fn == "tanh-relu"
    assert cfg.normalize_sae_decoder is False
    assert cfg.dataset_path == "my/dataset"
    assert resume_path is None


def test_parse_cfg_args_dict_args():
    # Test that we can pass dict args as json strings
    args = [
        "--model_kwargs",
        '{"foo": "bar", "baz": 123}',
        "--model_from_pretrained_kwargs",
        '{"center_writing_weights": false}',
        "--activation_fn_kwargs",
        '{"k": 100}',
    ]
    cfg, resume_path = _parse_cfg_args(args)

    assert cfg.model_kwargs == {"foo": "bar", "baz": 123}
    assert cfg.model_from_pretrained_kwargs == {"center_writing_weights": False}
    assert cfg.activation_fn_kwargs == {"k": 100}
    assert resume_path is None


def test_parse_cfg_args_invalid_json():
    args = ["--model_kwargs", "{invalid json"]
    with pytest.raises(argparse.ArgumentError, match="invalid json_dict value"):
        _parse_cfg_args(args)


def test_parse_cfg_args_invalid_dict_type():
    # Test that we reject non-dict values for dict fields
    args = ["--model_kwargs", "[1, 2, 3]"]  # Array instead of dict
    with pytest.raises(argparse.ArgumentError, match="invalid json_dict value"):
        _parse_cfg_args(args)

    args = ["--model_from_pretrained_kwargs", '"not_a_dict"']  # String instead of dict
    with pytest.raises(argparse.ArgumentError, match="invalid json_dict value"):
        _parse_cfg_args(args)

    args = ["--activation_fn_kwargs", "123"]  # Number instead of dict
    with pytest.raises(argparse.ArgumentError, match="invalid json_dict value"):
        _parse_cfg_args(args)


def test_parse_cfg_args_expansion_factor():
    # Test that we can't set both d_sae and expansion_factor
    args = ["--d_sae", "1024", "--expansion_factor", "8"]
    with pytest.raises(ValueError):
        _parse_cfg_args(args)


def test_parse_cfg_args_b_dec_init_method():
    # Test validation of b_dec_init_method
    args = ["--b_dec_init_method", "invalid"]
    with pytest.raises(ValueError):
        cfg, _ = _parse_cfg_args(args)

    valid_methods = ["geometric_median", "mean", "zeros"]
    for method in valid_methods:
        args = ["--b_dec_init_method", method]
        cfg, resume_path = _parse_cfg_args(args)
        assert cfg.b_dec_init_method == method
        assert resume_path is None


def test_run_cli_saves_config(tmp_path: Path):
    # Set up args for a minimal training run
    args = [
        "--model_name",
        TINYSTORIES_MODEL,
        "--dataset_path",
        TINYSTORIES_DATASET,
        "--checkpoint_path",
        str(tmp_path),
        "--n_checkpoints",
        "1",  # Save one checkpoint
        "--training_tokens",
        "128",
        "--train_batch_size_tokens",
        "4",
        "--store_batch_size_prompts",
        "4",
        "--log_to_wandb",
        "False",  # Don't log to wandb in test
        "--d_in",
        "64",  # Match gelu-1l hidden size
        "--d_sae",
        "128",  # Small SAE for test
        "--activation_fn",
        "relu",
        "--normalize_sae_decoder",
        "False",
    ]

    # Run training
    _run_cli(args)

    # Check that checkpoint was saved
    run_dirs = list(tmp_path.glob("*"))  # run dirs
    assert len(run_dirs) == 1
    checkpoint_dirs = list(run_dirs[0].glob("*"))
    assert len(checkpoint_dirs) == 1

    # Load and verify saved config
    with open(checkpoint_dirs[0] / "cfg.json") as f:
        saved_cfg = json.load(f)

    # Verify key config values were saved correctly
    assert saved_cfg["model_name"] == TINYSTORIES_MODEL
    assert saved_cfg["d_in"] == 64
    assert saved_cfg["d_sae"] == 128
    assert saved_cfg["activation_fn"] == "relu"
    assert saved_cfg["normalize_sae_decoder"] is False
    assert saved_cfg["dataset_path"] == TINYSTORIES_DATASET
    assert saved_cfg["n_checkpoints"] == 1
    assert saved_cfg["training_tokens"] == 128
    assert saved_cfg["train_batch_size_tokens"] == 4
    assert saved_cfg["store_batch_size_prompts"] == 4
    assert saved_cfg["model_name"] == TINYSTORIES_MODEL


def test_sae_training_runner_works_with_huggingface_models(tmp_path: Path):
    cfg = build_sae_cfg(
        d_in=64,
        d_sae=128,
        hook_layer=0,
        hook_name="transformer.h.0",
        checkpoint_path=str(tmp_path),
        model_class_name="AutoModelForCausalLM",
        model_name="roneneldan/TinyStories-1M",
        training_tokens=128,
        train_batch_size_tokens=4,
        store_batch_size_prompts=4,
        n_checkpoints=1,
        log_to_wandb=False,
    )

    runner = SAETrainingRunner(cfg)
    runner.run()

    # Check that checkpoint was saved
    checkpoint_dirs = list(tmp_path.glob("*"))  # run dirs
    assert len(checkpoint_dirs) == 1

    # Load and verify saved config
    with open(checkpoint_dirs[0] / "cfg.json") as f:
        saved_cfg = json.load(f)

    assert saved_cfg["model_name"] == "roneneldan/TinyStories-1M"
    assert saved_cfg["training_tokens"] == 128
    assert saved_cfg["train_batch_size_tokens"] == 4
    assert saved_cfg["store_batch_size_prompts"] == 4
    assert saved_cfg["log_to_wandb"] is False
    assert saved_cfg["model_class_name"] == "AutoModelForCausalLM"

    sae = SAE.load_from_pretrained(str(checkpoint_dirs[0]))
    assert isinstance(sae, SAE)


class TestResumeFromCheckpoint:
    """Tests for the resume-from-checkpoint functionality."""

    @pytest.fixture
    def small_training_cfg(self, tmp_path: Path):
        """Creates a config for a small training run."""
        return build_sae_cfg(
            d_in=64,
            d_sae=128,
            hook_layer=0,
            checkpoint_path=str(tmp_path),
            model_name=TINYSTORIES_MODEL,
            dataset_path=TINYSTORIES_DATASET,
            training_tokens=128,
            train_batch_size_tokens=4,
            store_batch_size_prompts=4,
            n_checkpoints=1,
            log_to_wandb=False,
            l1_coefficient=0.01,
            lr=0.001,
        )

    def test_save_checkpoint_includes_training_state(
        self, tmp_path: Path, small_training_cfg: LanguageModelSAERunnerConfig
    ):
        """Test that save_checkpoint saves training state."""
        # Run a small training session
        runner = SAETrainingRunner(small_training_cfg)
        runner.run()

        # Verify training state was saved
        checkpoint_dirs = list(Path(small_training_cfg.checkpoint_path).glob("*"))
        assert len(checkpoint_dirs) == 1

        training_state_path = checkpoint_dirs[0] / "training_state.pt"
        assert training_state_path.exists(), "Training state wasn't saved"

        # Load the training state
        training_state = torch.load(training_state_path)

        # Check the contents of the training state
        assert "optimizer" in training_state
        assert "lr_scheduler" in training_state
        assert "l1_scheduler" in training_state
        assert "n_training_tokens" in training_state
        assert "n_training_steps" in training_state
        assert "act_freq_scores" in training_state
        assert "n_forward_passes_since_fired" in training_state
        assert "n_frac_active_tokens" in training_state
        assert "started_fine_tuning" in training_state

        # Verify activations store state was saved
        activations_store_path = (
            checkpoint_dirs[0] / "activations_store_state.safetensors"
        )
        assert activations_store_path.exists(), "Activations store state wasn't saved"

    def test_cli_args_parsing_with_resume(self):
        """Test that CLI args parsing works with --resume-from-checkpoint."""
        args = [
            "--model_name",
            "test-model",
            "--dataset_path",
            "test-dataset",
            "--resume-from-checkpoint",
            "/path/to/checkpoint",
        ]
        cfg, resume_path = _parse_cfg_args(args)

        assert cfg.model_name == "test-model"
        assert cfg.dataset_path == "test-dataset"
        assert resume_path == "/path/to/checkpoint"

    def test_resume_from_checkpoint(
        self, tmp_path: Path, small_training_cfg: LanguageModelSAERunnerConfig
    ):
        """Test that training can be resumed from a checkpoint."""
        # First part: train for a small number of tokens
        first_cfg = small_training_cfg
        first_cfg.training_tokens = 64  # Half of total

        runner1 = SAETrainingRunner(first_cfg)
        runner1.run()

        # Get the checkpoint directory
        checkpoint_dirs = list(Path(first_cfg.checkpoint_path).glob("*"))
        assert len(checkpoint_dirs) == 1
        checkpoint_path = checkpoint_dirs[0]

        # Second part: resume training from the checkpoint
        second_cfg = small_training_cfg
        second_cfg.training_tokens = 128  # Full amount

        # Resume training from checkpoint
        runner2 = SAETrainingRunner(
            second_cfg, resume_from_checkpoint=str(checkpoint_path)
        )
        runner2.run()

        # The resumed SAE should have trained on all tokens
        # Check if various metrics are reasonable

        # Get the final checkpoint and check its metrics
        final_checkpoint_dirs = list(Path(second_cfg.checkpoint_path).glob("*"))
        assert (
            len(final_checkpoint_dirs) >= 1
        )  # Should have at least the original checkpoint

        # Find the latest checkpoint (could be different from the first if new one was created)
        latest_checkpoint = max(final_checkpoint_dirs, key=lambda p: p.stat().st_mtime)

        # Load the final state
        final_training_state_path = latest_checkpoint / "training_state.pt"
        final_training_state = torch.load(final_training_state_path)

        # Ensure the resumed training completed the full training
        assert final_training_state["n_training_tokens"] >= 128

    def test_optimizer_state_preserved(
        self, tmp_path: Path, small_training_cfg: LanguageModelSAERunnerConfig
    ):
        """Test that optimizer state is preserved exactly when saving and loading."""
        # Create a runner
        runner = SAETrainingRunner(small_training_cfg)

        # Create a trainer and run a step to initialize states
        trainer = SAETrainer(
            model=runner.model,
            sae=runner.sae,
            activation_store=runner.activations_store,
            save_checkpoint_fn=runner.save_checkpoint,
            cfg=runner.cfg,
        )

        # Run a step to populate optimizer state
        batch = runner.activations_store.next_batch()
        layer_acts = batch[:, 0, :].to(runner.sae.device)
        trainer._train_step(sae=trainer.sae, sae_in=layer_acts)

        # Save the original optimizer state
        orig_optimizer_state = trainer.optimizer.state_dict()

        # Save checkpoint
        checkpoint_dir = tmp_path / "optimizer_test"
        checkpoint_dir.mkdir(exist_ok=True)
        SAETrainingRunner.save_checkpoint(trainer, str(checkpoint_dir))

        # Create a new trainer with fresh optimizer
        new_trainer = SAETrainer(
            model=runner.model,
            sae=runner.sae,
            activation_store=runner.activations_store,
            save_checkpoint_fn=runner.save_checkpoint,
            cfg=runner.cfg,
        )

        # Load the optimizer state
        _load_checkpoint_state(
            trainer=new_trainer,
            checkpoint_path_str=str(checkpoint_dir),
            activations_store=runner.activations_store,
            device=small_training_cfg.device,
        )

        # Compare optimizer states
        new_optimizer_state = new_trainer.optimizer.state_dict()

        # Compare optimizer param_groups
        assert len(orig_optimizer_state["param_groups"]) == len(
            new_optimizer_state["param_groups"]
        )
        for i, (orig_group, new_group) in enumerate(
            zip(
                orig_optimizer_state["param_groups"],
                new_optimizer_state["param_groups"],
            )
        ):
            for key, orig_value in orig_group.items():
                if key != "params":  # params are memory addresses, which will differ
                    assert orig_value == new_group[key], f"Group {i}, key {key} differs"

        # Compare optimizer state
        assert set(orig_optimizer_state["state"].keys()) == set(
            new_optimizer_state["state"].keys()
        )
        for param_id in orig_optimizer_state["state"]:
            for state_name, orig_value in orig_optimizer_state["state"][
                param_id
            ].items():
                new_value = new_optimizer_state["state"][param_id][state_name]
                if isinstance(orig_value, torch.Tensor):
                    assert torch.allclose(
                        orig_value, new_value
                    ), f"Parameter {param_id}, state {state_name} differs"
                else:
                    assert (
                        orig_value == new_value
                    ), f"Parameter {param_id}, state {state_name} differs"

    def test_activations_store_state_preserved(
        self, tmp_path: Path, small_training_cfg: LanguageModelSAERunnerConfig
    ):
        """Test that activations store state is preserved exactly when saving and loading."""
        # Create a runner and get some activations
        runner = SAETrainingRunner(small_training_cfg)

        # Get a few batches to ensure the store has some state
        for _ in range(300):
            runner.activations_store.next_batch()
        assert runner.activations_store.n_dataset_processed > 0

        # Save original activations store state
        orig_state = runner.activations_store.state_dict()

        # Save to disk
        store_path = tmp_path / "activations_store_state.safetensors"
        runner.activations_store.save(str(store_path))

        # Create a new activations store
        new_store = ActivationsStore.from_config(
            runner.model,
            runner.cfg,
        )

        # Load the saved state
        new_store.load(str(store_path))

        # Compare states
        new_state = new_store.state_dict()

        # Check n_dataset_processed
        assert (
            orig_state["n_dataset_processed"].item()
            == new_state["n_dataset_processed"].item()
        )

        # Check storage buffer if present
        assert torch.allclose(
            orig_state["storage_buffer_activations"],
            new_state["storage_buffer_activations"],
        )

    def test_activations_store_load_method(
        self, tmp_path: Path, small_training_cfg: LanguageModelSAERunnerConfig
    ):
        """Test that the ActivationsStore load method works correctly."""
        # Run a small training session
        runner = SAETrainingRunner(small_training_cfg)
        runner.activations_store.estimated_norm_scaling_factor = 17.5
        runner.run()

        # Get the checkpoint directory
        checkpoint_dirs = list(Path(small_training_cfg.checkpoint_path).glob("*"))
        assert len(checkpoint_dirs) == 1

        # Get the activations store state
        activations_store_path = (
            checkpoint_dirs[0] / "activations_store_state.safetensors"
        )
        assert activations_store_path.exists()

        # Create a new runner
        new_runner = SAETrainingRunner(small_training_cfg)

        # Test that we can load the activations store state
        new_runner.activations_store.load(str(activations_store_path))

        # Check that the estimator scaling factor was loaded if it exists
        assert (
            "estimated_norm_scaling_factor" in new_runner.activations_store.state_dict()
        )
        assert new_runner.activations_store.estimated_norm_scaling_factor == 17.5

    def test_resume_from_cli(self, tmp_path: Path):
        """Test resume from checkpoint using the CLI interface."""
        # First run: short training
        run_id = "test-run-id"  # Fixed run ID for reproducibility
        checkpoint_dir = tmp_path / run_id

        # Create a checkpoint directory structure that matches what the real system expects
        first_args = [
            "--model_name",
            TINYSTORIES_MODEL,
            "--dataset_path",
            TINYSTORIES_DATASET,
            "--checkpoint_path",
            str(checkpoint_dir),  # Use the fixed path
            "--wandb_id",
            run_id,  # Use fixed run ID
            "--training_tokens",
            "64",  # Short training
            "--train_batch_size_tokens",
            "4",
            "--store_batch_size_prompts",
            "4",
            "--log_to_wandb",
            "False",
            "--d_in",
            "64",
            "--d_sae",
            "128",
            "--n_checkpoints",
            "1",  # Just one checkpoint at the end
        ]

        _run_cli(first_args)

        # Find the checkpoint files by exploring the directory structure

        # Find the checkpoint subdirectory - based on the output, we have a nested structure
        # The actual checkpoint is in test-run-id/test-run-id/final_64
        nested_run_dir = checkpoint_dir / run_id
        assert (
            nested_run_dir.exists()
        ), f"Expected nested run directory at {nested_run_dir}"

        # Look for the final_* directory that contains the checkpoint files
        checkpoint_subdirs = list(nested_run_dir.glob("final_*"))
        assert (
            len(checkpoint_subdirs) >= 1
        ), f"Expected at least one final_* directory in {nested_run_dir}"

        # Verify the checkpoint files exist
        final_checkpoint_dir = checkpoint_subdirs[0]
        assert (
            final_checkpoint_dir / "cfg.json"
        ).exists(), f"No cfg.json found in {final_checkpoint_dir}"
        assert (
            final_checkpoint_dir / "sae_weights.safetensors"
        ).exists(), f"No sae_weights.safetensors found in {final_checkpoint_dir}"

        # Second run: resume with more tokens
        second_args = [
            "--model_name",
            TINYSTORIES_MODEL,
            "--dataset_path",
            TINYSTORIES_DATASET,
            "--checkpoint_path",
            str(tmp_path / "continued"),  # Different path
            "--wandb_id",
            run_id,  # Same run ID
            "--training_tokens",
            "128",  # Twice the tokens
            "--train_batch_size_tokens",
            "4",
            "--store_batch_size_prompts",
            "4",
            "--log_to_wandb",
            "False",
            "--d_in",
            "64",
            "--d_sae",
            "128",
            "--resume-from-checkpoint",
            str(final_checkpoint_dir),
        ]

        _run_cli(second_args)

        # Find all training_state.pt files in the continued directory
        training_state_files = list(
            (tmp_path / "continued").glob("**/training_state.pt")
        )
        assert (
            len(training_state_files) >= 1
        ), f"Expected at least one training_state.pt file in {tmp_path / 'continued'}"

        # The parent directory of the training_state.pt file is our checkpoint directory
        final_continued_checkpoint = training_state_files[0].parent

        # Load the training state
        training_state = torch.load(final_continued_checkpoint / "training_state.pt")

        # Verify it ran the full training
        assert training_state["n_training_tokens"] >= 128
