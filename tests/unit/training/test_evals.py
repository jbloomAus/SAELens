import pytest
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.evals import run_evals
from sae_lens.training.sparse_autoencoder import (
    SparseAutoencoderBase,
    TrainingSparseAutoencoder,
)
from tests.unit.helpers import TINYSTORIES_MODEL, build_sae_cfg, load_model_cached


@pytest.fixture
def cfg():
    cfg = build_sae_cfg(d_in=64, d_sae=128, hook_point_layer=0)
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
def base_sae(cfg: LanguageModelSAERunnerConfig):
    return SparseAutoencoderBase.from_dict(cfg.get_base_sae_cfg_dict())


@pytest.fixture
def training_sae(cfg: LanguageModelSAERunnerConfig):
    return TrainingSparseAutoencoder.from_dict(cfg.get_training_sae_cfg_dict())


def test_run_evals_base_sae(
    base_sae: SparseAutoencoderBase,
    activation_store: ActivationsStore,
    model: HookedTransformer,
):

    eval_metrics = run_evals(
        sparse_autoencoder=base_sae,
        activation_store=activation_store,
        model=model,
        n_eval_batches=2,
        eval_batch_size_prompts=None,
    )

    expected_keys = [
        "metrics/l2_norm",
        "metrics/l2_ratio",
        "metrics/l2_norm_in",
        "metrics/CE_loss_score",
        "metrics/ce_loss_without_sae",
        "metrics/ce_loss_with_sae",
        "metrics/ce_loss_with_ablation",
    ]

    # results will be garbage without a real model.
    for key in expected_keys:
        assert key in eval_metrics


def test_run_evals_training_sae(
    training_sae: TrainingSparseAutoencoder,
    activation_store: ActivationsStore,
    model: HookedTransformer,
):

    eval_metrics = run_evals(
        sparse_autoencoder=training_sae,
        activation_store=activation_store,
        model=model,
        n_eval_batches=10,
        eval_batch_size_prompts=None,
    )

    expected_keys = [
        "metrics/l2_norm",
        "metrics/l2_ratio",
        "metrics/l2_norm_in",
        "metrics/CE_loss_score",
        "metrics/ce_loss_without_sae",
        "metrics/ce_loss_with_sae",
        "metrics/ce_loss_with_ablation",
    ]

    for key in expected_keys:
        assert key in eval_metrics
