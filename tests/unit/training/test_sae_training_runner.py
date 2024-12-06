import json
import os
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae import SAE
from sae_lens.sae_training_runner import SAETrainingRunner, _parse_cfg_args, _run_cli
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.training_sae import TrainingSAE
from tests.unit.helpers import (
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
    cfg = _parse_cfg_args(args)

    assert cfg.model_name == "test-model"
    assert cfg.d_in == 1024
    assert cfg.d_sae == 4096
    assert cfg.activation_fn == "tanh-relu"
    assert cfg.normalize_sae_decoder is False
    assert cfg.dataset_path == "my/dataset"


def test_parse_cfg_args_expansion_factor():
    # Test that we can't set both d_sae and expansion_factor
    args = ["--d_sae", "1024", "--expansion_factor", "8"]
    with pytest.raises(ValueError):
        _parse_cfg_args(args)


def test_parse_cfg_args_b_dec_init_method():
    # Test validation of b_dec_init_method
    args = ["--b_dec_init_method", "invalid"]
    with pytest.raises(ValueError):
        cfg = _parse_cfg_args(args)

    valid_methods = ["geometric_median", "mean", "zeros"]
    for method in valid_methods:
        args = ["--b_dec_init_method", method]
        cfg = _parse_cfg_args(args)
        assert cfg.b_dec_init_method == method


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
