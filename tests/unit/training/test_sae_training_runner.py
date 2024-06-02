import os
from pathlib import Path

import pytest
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae import SAE
from sae_lens.sae_training_runner import SAETrainingRunner
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.training_sae import TrainingSAE
from tests.unit.helpers import TINYSTORIES_MODEL, build_sae_cfg, load_model_cached


@pytest.fixture
def cfg(tmp_path: Path):
    cfg = build_sae_cfg(d_in=64, d_sae=128, hook_layer=0, checkpoint_path=str(tmp_path))
    return cfg


@pytest.fixture
def model():
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def activation_store(model: HookedTransformer, cfg: LanguageModelSAERunnerConfig):
    return ActivationsStore.from_config(
        model, cfg, dataset=Dataset.from_list([{"text": "hello world"}] * 2000)
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

    trainer = SAETrainer(
        model=model,
        sae=training_sae,
        activation_store=activation_store,
        save_checkpoint_fn=lambda *args, **kwargs: None,
        cfg=cfg,
    )

    return trainer


@pytest.fixture
def training_runner(
    cfg: LanguageModelSAERunnerConfig,
):
    runner = SAETrainingRunner(cfg)
    return runner


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
