import argparse
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.evals import (
    EvalConfig,
    get_eval_everything_config,
    process_results,
    run_evals,
    run_evaluations,
)
from sae_lens.sae import SAE
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.training_sae import TrainingSAE
from tests.unit.helpers import TINYSTORIES_MODEL, build_sae_cfg, load_model_cached

TRAINER_EVAL_CONFIG = EvalConfig(
    n_eval_reconstruction_batches=10,
    compute_ce_loss=True,
    n_eval_sparsity_variance_batches=1,
    compute_l2_norms=True,
)


@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "hook_name": "blocks.1.attn.hook_z",
            "hook_layer": 1,
            "d_in": 16 * 4,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "hook_name": "blocks.1.attn.hook_q",
            "hook_layer": 1,
            "d_in": 16 * 4,
        },
    ],
    ids=[
        "tiny-stories-1M-resid-pre",
        "tiny-stories-1M-resid-pre-L1-W-dec-Norm",
        "tiny-stories-1M-resid-pre-pretokenized",
        "tiny-stories-1M-hook-z",
        "tiny-stories-1M-hook-q",
    ],
)
def cfg(request: pytest.FixtureRequest):
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    params = request.param
    return build_sae_cfg(**params)


@pytest.fixture
def model():
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def activation_store(model: HookedTransformer, cfg: LanguageModelSAERunnerConfig):
    return ActivationsStore.from_config(
        model, cfg, override_dataset=Dataset.from_list([{"text": "hello world"}] * 2000)
    )


@pytest.fixture
def base_sae(cfg: LanguageModelSAERunnerConfig):
    return SAE.from_dict(cfg.get_base_sae_cfg_dict())


@pytest.fixture
def training_sae(cfg: LanguageModelSAERunnerConfig):
    return TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())


all_possible_keys = [
    "model_behavior_preservation",
    "model_performance_preservation",
    "reconstruction_quality",
    "shrinkage",
    "sparsity",
    "token_stats",
]

all_featurewise_keys_expected = [
    "feature_density",
    "consistent_activation_heuristic",
    "encoder_bias",
    "encoder_decoder_cosine_sim",
    "encoder_norm",
]


def test_run_evals_base_sae(
    base_sae: SAE,
    activation_store: ActivationsStore,
    model: HookedTransformer,
):
    eval_metrics, _ = run_evals(
        sae=base_sae,
        activation_store=activation_store,
        model=model,
        eval_config=get_eval_everything_config(),
    )

    assert set(eval_metrics.keys()).issubset(set(all_possible_keys))
    assert len(eval_metrics) > 0


def test_run_evals_training_sae(
    training_sae: TrainingSAE,
    activation_store: ActivationsStore,
    model: HookedTransformer,
):
    eval_metrics, feature_metrics = run_evals(
        sae=training_sae,
        activation_store=activation_store,
        model=model,
        eval_config=get_eval_everything_config(),
    )

    assert set(eval_metrics.keys()).issubset(set(all_possible_keys))
    assert len(eval_metrics) > 0
    assert set(feature_metrics.keys()).issubset(set(all_featurewise_keys_expected))


def test_run_evals_training_sae_ignore_bos(
    training_sae: TrainingSAE,
    activation_store: ActivationsStore,
    model: HookedTransformer,
):
    eval_metrics, _ = run_evals(
        sae=training_sae,
        activation_store=activation_store,
        model=model,
        eval_config=get_eval_everything_config(),
        ignore_tokens={
            model.tokenizer.bos_token_id,  # type: ignore
            model.tokenizer.eos_token_id,  # type: ignore
            model.tokenizer.pad_token_id,  # type: ignore
        },
    )

    assert set(eval_metrics.keys()).issubset(set(all_possible_keys))
    assert len(eval_metrics) > 0


def test_training_eval_config(
    base_sae: SAE,
    activation_store: ActivationsStore,
    model: HookedTransformer,
):
    expected_keys = [
        "model_performance_preservation",
        "shrinkage",
        "token_stats",
    ]
    eval_config = TRAINER_EVAL_CONFIG
    eval_metrics, _ = run_evals(
        sae=base_sae,
        activation_store=activation_store,
        model=model,
        eval_config=eval_config,
    )
    assert set(eval_metrics.keys()) == set(expected_keys)


def test_training_eval_config_ignore_control_tokens(
    base_sae: SAE,
    activation_store: ActivationsStore,
    model: HookedTransformer,
):
    expected_keys = [
        "model_performance_preservation",
        "shrinkage",
        "token_stats",
    ]
    eval_config = TRAINER_EVAL_CONFIG
    eval_metrics, _ = run_evals(
        sae=base_sae,
        activation_store=activation_store,
        model=model,
        eval_config=eval_config,
        ignore_tokens={
            model.tokenizer.pad_token_id,  # type: ignore
            model.tokenizer.eos_token_id,  # type: ignore
            model.tokenizer.bos_token_id,  # type: ignore
        },
    )
    assert set(eval_metrics.keys()) == set(expected_keys)


def test_run_empty_evals(
    base_sae: SAE,
    activation_store: ActivationsStore,
    model: HookedTransformer,
):
    empty_config = EvalConfig(
        n_eval_reconstruction_batches=0,
        n_eval_sparsity_variance_batches=0,
        compute_ce_loss=False,
        compute_kl=False,
        compute_l2_norms=False,
        compute_sparsity_metrics=False,
        compute_variance_metrics=False,
        compute_featurewise_density_statistics=False,
    )
    eval_metrics, feature_metrics = run_evals(
        sae=base_sae,
        activation_store=activation_store,
        model=model,
        eval_config=empty_config,
    )

    assert len(eval_metrics) == 1, "Expected only token_stats in eval_metrics"
    assert "token_stats" in eval_metrics, "Expected token_stats in eval_metrics"
    assert len(feature_metrics) == 0, "Expected empty feature_metrics"


@pytest.fixture
def mock_args():
    args = argparse.Namespace()
    args.sae_regex_pattern = "test_pattern"
    args.sae_block_pattern = "test_block"
    args.num_eval_batches = 2
    args.batch_size_prompts = 4
    args.eval_batch_size_prompts = 4
    args.n_eval_reconstruction_batches = 1
    args.n_eval_sparsity_variance_batches = 1
    args.datasets = ["test_dataset"]
    args.ctx_lens = [64]
    args.output_dir = "test_output"
    args.verbose = False
    return args


@patch("sae_lens.evals.get_saes_from_regex")
@patch("sae_lens.evals.multiple_evals")
def test_run_evaluations(
    mock_multiple_evals: MagicMock,
    mock_get_saes_from_regex: MagicMock,
    mock_args: argparse.Namespace,
):
    mock_get_saes_from_regex.return_value = [
        ("release1", "sae1", 0.8, 10),
        ("release2", "sae2", 0.7, 8),
    ]
    mock_multiple_evals.return_value = [{"test": "result"}]

    result = run_evaluations(mock_args)

    mock_get_saes_from_regex.assert_called_once_with(
        mock_args.sae_regex_pattern, mock_args.sae_block_pattern
    )
    mock_multiple_evals.assert_called_once_with(
        sae_regex_pattern=mock_args.sae_regex_pattern,
        sae_block_pattern=mock_args.sae_block_pattern,
        eval_batch_size_prompts=mock_args.eval_batch_size_prompts,
        n_eval_reconstruction_batches=mock_args.n_eval_reconstruction_batches,
        n_eval_sparsity_variance_batches=mock_args.n_eval_sparsity_variance_batches,
        datasets=mock_args.datasets,
        ctx_lens=mock_args.ctx_lens,
        output_dir=mock_args.output_dir,
        verbose=mock_args.verbose,
    )
    assert result == [{"test": "result"}]


def test_process_results(tmp_path: Path):
    eval_results = [
        {
            "unique_id": "test-sae",
            "eval_cfg": {"context_size": 64, "dataset": "test/dataset"},
            "metrics": {"metric1": 0.5, "metric2": 0.7},
            "sae_cfg": {"config1": "value1"},
        }
    ]
    output_dir = tmp_path / "test_output"

    process_results(eval_results, str(output_dir))  # type: ignore

    # Check if individual JSON file is created
    individual_json_path = output_dir / "test-sae_64_test_dataset.json"
    assert individual_json_path.exists()
    with open(individual_json_path, "r") as f:
        assert json.load(f) == eval_results[0]

    # Check if combined JSON file is created
    combined_json_path = output_dir / "all_eval_results.json"
    assert combined_json_path.exists()
    with open(combined_json_path, "r") as f:
        assert json.load(f) == eval_results

    # Check if CSV file is created
    csv_path = output_dir / "all_eval_results.csv"
    assert csv_path.exists()
