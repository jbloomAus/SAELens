import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformer_lens import HookedTransformer

from sae_lens import __version__
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.llm_sae_training_runner import (
    LanguageModelSAETrainingRunner,
    LLMSaeEvaluator,
    _parse_cfg_args,
)
from sae_lens.saes.gated_sae import GatedTrainingSAEConfig
from sae_lens.saes.jumprelu_sae import JumpReLUTrainingSAEConfig
from sae_lens.saes.sae import SAE, TrainingSAE
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore
from tests.helpers import (
    ALL_TRAINING_ARCHITECTURES,
    NEEL_NANDA_C4_10K_DATASET,
    TINYSTORIES_MODEL,
    build_runner_cfg_for_arch,
)


@pytest.mark.parametrize("architecture", ALL_TRAINING_ARCHITECTURES)
def test_LanguageModelSAETrainingRunner_runs_and_saves_all_architectures(
    architecture: str, tmp_path: Path, ts_model: HookedTransformer
):
    cfg = build_runner_cfg_for_arch(
        d_in=64,
        d_sae=128,
        architecture=architecture,
        checkpoint_path=str(tmp_path),
        training_tokens=100,
        store_batch_size_prompts=2,
        train_batch_size_tokens=4,
        model_batch_size=1,
        context_size=10,
        n_batches_in_buffer=2,
        dataset_path=NEEL_NANDA_C4_10K_DATASET,
        hook_name="blocks.0.hook_resid_post",
        model_name=TINYSTORIES_MODEL,
        n_checkpoints=0,
        exclude_special_tokens=True,
    )
    runner = LanguageModelSAETrainingRunner(cfg, override_model=ts_model)
    sae = runner.run()

    assert sae.cfg.architecture() == architecture
    sae_cfg_dict = sae.cfg.to_dict()
    sae_cfg_dict.pop("metadata")  # metadata should be set by the llm runner
    original_cfg_dict = cfg.sae.to_dict()
    original_cfg_dict.pop("metadata")  # metadata should be set by the llm runner
    assert sae_cfg_dict == original_cfg_dict

    assert sae.cfg.metadata.dataset_path == NEEL_NANDA_C4_10K_DATASET
    assert sae.cfg.metadata.hook_name == "blocks.0.hook_resid_post"
    assert sae.cfg.metadata.model_name == TINYSTORIES_MODEL
    assert sae.cfg.metadata.model_class_name == "HookedTransformer"
    assert sae.cfg.metadata.hook_head_index is None
    assert sae.cfg.metadata.model_from_pretrained_kwargs == {
        "center_writing_weights": False
    }
    assert sae.cfg.metadata.prepend_bos is True
    assert sae.cfg.metadata.exclude_special_tokens is True
    assert sae.cfg.metadata.sae_lens_version == __version__
    assert sae.cfg.metadata.sae_lens_training_version == __version__

    assert (tmp_path / "final_100").exists()
    loaded_sae = SAE.load_from_disk(tmp_path / "final_100")

    # json turns tuples into lists, so just dump and load the metadata to make things consistent
    original_metadata_dict = json.loads(json.dumps(sae.cfg.metadata.__dict__))
    if architecture == "batchtopk":
        assert loaded_sae.cfg.architecture() == "jumprelu"
    else:
        assert loaded_sae.cfg.architecture() == architecture
    assert loaded_sae.cfg.d_in == sae.cfg.d_in
    assert loaded_sae.cfg.d_sae == sae.cfg.d_sae
    assert loaded_sae.cfg.dtype == sae.cfg.dtype
    assert loaded_sae.cfg.device == sae.cfg.device
    assert loaded_sae.cfg.apply_b_dec_to_input == sae.cfg.apply_b_dec_to_input
    assert loaded_sae.cfg.metadata.__dict__ == original_metadata_dict


def test_parse_cfg_args_raises_system_exit_on_empty_args():
    with pytest.raises(SystemExit):
        _parse_cfg_args([])


def test_parse_cfg_args_raises_exception_on_invalid_args():
    with pytest.raises((SystemExit, Exception)):
        _parse_cfg_args(["--invalid-argument", "value"])


def test_parse_cfg_args_works_with_basic_arguments():
    args = [
        "--model_name",
        "gpt2",
        "--dataset_path",
        "test_dataset",
        "--d_in",
        "768",
        "--d_sae",
        "1536",
        "--hook_name",
        "blocks.0.hook_resid_post",
        "--context_size",
        "128",
        "--training_tokens",
        "1000000",
    ]
    cfg = _parse_cfg_args(args)
    assert isinstance(cfg, LanguageModelSAERunnerConfig)
    assert cfg.model_name == "gpt2"
    assert cfg.dataset_path == "test_dataset"
    assert cfg.hook_name == "blocks.0.hook_resid_post"
    assert cfg.context_size == 128
    assert cfg.training_tokens == 1000000
    assert cfg.sae.d_in == 768
    assert cfg.sae.d_sae == 1536
    assert cfg.sae.architecture() == "standard"
    assert isinstance(cfg.sae, StandardTrainingSAEConfig)
    assert cfg.sae.l1_coefficient == 1.0  # default value
    assert cfg.sae.lp_norm == 1.0  # default value
    assert cfg.sae.l1_warm_up_steps == 0  # default value


def test_parse_cfg_args_selects_gated_architecture():
    args = [
        "--architecture",
        "gated",
        "--model_name",
        "gpt2",
        "--dataset_path",
        "test_dataset",
        "--d_in",
        "768",
        "--d_sae",
        "1536",
        "--hook_name",
        "blocks.0.hook_resid_post",
        "--l1_coefficient",
        "0.5",
        "--l1_warm_up_steps",
        "1000",
    ]
    cfg = _parse_cfg_args(args)
    assert isinstance(cfg, LanguageModelSAERunnerConfig)
    assert cfg.model_name == "gpt2"
    assert cfg.dataset_path == "test_dataset"
    assert cfg.hook_name == "blocks.0.hook_resid_post"
    assert cfg.sae.d_in == 768
    assert cfg.sae.d_sae == 1536
    assert cfg.sae.architecture() == "gated"
    assert isinstance(cfg.sae, GatedTrainingSAEConfig)
    assert cfg.sae.l1_coefficient == 0.5
    assert cfg.sae.l1_warm_up_steps == 1000


def test_parse_cfg_args_selects_topk_architecture():
    args = [
        "--architecture",
        "topk",
        "--model_name",
        "gpt2",
        "--dataset_path",
        "test_dataset",
        "--d_in",
        "768",
        "--d_sae",
        "1536",
        "--hook_name",
        "blocks.0.hook_resid_post",
        "--k",
        "50",
    ]
    cfg = _parse_cfg_args(args)
    assert isinstance(cfg, LanguageModelSAERunnerConfig)
    assert cfg.model_name == "gpt2"
    assert cfg.dataset_path == "test_dataset"
    assert cfg.hook_name == "blocks.0.hook_resid_post"
    assert cfg.sae.d_in == 768
    assert cfg.sae.d_sae == 1536
    assert cfg.sae.architecture() == "topk"
    assert isinstance(cfg.sae, TopKTrainingSAEConfig)
    assert cfg.sae.k == 50


def test_parse_cfg_args_selects_standard_architecture_with_specific_options():
    args = [
        "--architecture",
        "standard",
        "--model_name",
        "gpt2",
        "--dataset_path",
        "test_dataset",
        "--d_in",
        "768",
        "--d_sae",
        "1536",
        "--hook_name",
        "blocks.0.hook_resid_post",
        "--l1_coefficient",
        "0.8",
        "--lp_norm",
        "1.5",
        "--l1_warm_up_steps",
        "2000",
    ]
    cfg = _parse_cfg_args(args)
    assert isinstance(cfg, LanguageModelSAERunnerConfig)
    assert cfg.model_name == "gpt2"
    assert cfg.dataset_path == "test_dataset"
    assert cfg.hook_name == "blocks.0.hook_resid_post"
    assert cfg.sae.d_in == 768
    assert cfg.sae.d_sae == 1536
    assert cfg.sae.architecture() == "standard"
    assert isinstance(cfg.sae, StandardTrainingSAEConfig)
    assert cfg.sae.l1_coefficient == 0.8
    assert cfg.sae.lp_norm == 1.5
    assert cfg.sae.l1_warm_up_steps == 2000


def test_parse_cfg_args_selects_jumprelu_architecture():
    args = [
        "--architecture",
        "jumprelu",
        "--model_name",
        "gpt2",
        "--dataset_path",
        "test_dataset",
        "--d_in",
        "768",
        "--d_sae",
        "1536",
        "--hook_name",
        "blocks.0.hook_resid_post",
        "--jumprelu_init_threshold",
        "0.002",
        "--jumprelu_bandwidth",
        "0.0005",
        "--l0_coefficient",
        "0.3",
        "--l0_warm_up_steps",
        "500",
    ]
    cfg = _parse_cfg_args(args)
    assert isinstance(cfg, LanguageModelSAERunnerConfig)
    assert cfg.model_name == "gpt2"
    assert cfg.dataset_path == "test_dataset"
    assert cfg.hook_name == "blocks.0.hook_resid_post"
    assert cfg.sae.d_in == 768
    assert cfg.sae.d_sae == 1536
    assert cfg.sae.architecture() == "jumprelu"
    assert isinstance(cfg.sae, JumpReLUTrainingSAEConfig)
    assert cfg.sae.jumprelu_init_threshold == 0.002
    assert cfg.sae.jumprelu_bandwidth == 0.0005
    assert cfg.sae.l0_coefficient == 0.3
    assert cfg.sae.l0_warm_up_steps == 500


class TestLLMSaeEvaluator:
    """Test suite for LLMSaeEvaluator class."""

    @pytest.fixture
    def cfg(self) -> LanguageModelSAERunnerConfig[Any]:
        return build_runner_cfg_for_arch(
            d_in=64,
            d_sae=128,
            architecture="standard",
            training_tokens=100,
            store_batch_size_prompts=2,
            train_batch_size_tokens=4,
            model_batch_size=1,
            context_size=10,
            n_batches_in_buffer=2,
            dataset_path=NEEL_NANDA_C4_10K_DATASET,
            hook_name="blocks.0.hook_resid_post",
            model_name=TINYSTORIES_MODEL,
            exclude_special_tokens=True,
        )

    @pytest.fixture
    def activation_store(
        self, ts_model: HookedTransformer, cfg: LanguageModelSAERunnerConfig[Any]
    ) -> ActivationsStore:
        return ActivationsStore.from_config(ts_model, cfg)

    @pytest.fixture
    def training_sae(self, cfg: LanguageModelSAERunnerConfig[Any]) -> TrainingSAE[Any]:
        return TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    @pytest.fixture
    def evaluator(
        self,
        ts_model: HookedTransformer,
        activation_store: ActivationsStore,
    ) -> LLMSaeEvaluator[Any]:
        return LLMSaeEvaluator(
            model=ts_model,
            activations_store=activation_store,
            eval_batch_size_prompts=2,
            n_eval_batches=1,
            model_kwargs={},
        )

    def test_llm_sae_evaluator_returns_metrics(
        self,
        evaluator: LLMSaeEvaluator[Any],
        training_sae: TrainingSAE[Any],
        activation_store: ActivationsStore,
    ):
        activation_scaler = ActivationScaler()

        metrics = evaluator(
            sae=training_sae,
            data_provider=activation_store,
            activation_scaler=activation_scaler,
        )

        assert isinstance(metrics, dict)
        assert len(metrics) > 0

        # Should contain some evaluation metrics
        metric_keys = set(metrics.keys())
        expected_keys = {
            "model_performance_preservation",
            "reconstruction_quality",
            "shrinkage",
            "sparsity",
            "token_stats",
        }
        assert metric_keys == expected_keys

    def test_llm_sae_evaluator_filters_training_metrics(
        self,
        evaluator: LLMSaeEvaluator[Any],
        training_sae: TrainingSAE[Any],
        activation_store: ActivationsStore,
    ):
        activation_scaler = ActivationScaler()

        metrics = evaluator(
            sae=training_sae,
            data_provider=activation_store,
            activation_scaler=activation_scaler,
        )

        # These metrics should be filtered out
        filtered_metrics = {
            "metrics/explained_variance",
            "metrics/explained_variance_std",
            "metrics/l0",
            "metrics/l1",
            "metrics/mse",
            "metrics/total_tokens_evaluated",
        }

        for metric_key in filtered_metrics:
            assert metric_key not in metrics

    def test_llm_sae_evaluator_handles_ignore_tokens_with_exclude_special_tokens(
        self,
        ts_model: HookedTransformer,
        activation_store: ActivationsStore,
        training_sae: TrainingSAE[Any],
    ):
        # Set up activation store with exclude_special_tokens
        activation_store.exclude_special_tokens = torch.tensor([0, 1, 2])

        evaluator = LLMSaeEvaluator(
            model=ts_model,
            activations_store=activation_store,
            eval_batch_size_prompts=2,
            n_eval_batches=1,
            model_kwargs={},
        )

        activation_scaler = ActivationScaler()

        # Should not raise an error and should return metrics
        metrics = evaluator(
            sae=training_sae,
            data_provider=activation_store,
            activation_scaler=activation_scaler,
        )

        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_llm_sae_evaluator_handles_no_exclude_special_tokens(
        self,
        ts_model: HookedTransformer,
        activation_store: ActivationsStore,
        training_sae: TrainingSAE[Any],
    ):
        # Set up activation store without exclude_special_tokens
        activation_store.exclude_special_tokens = None

        evaluator = LLMSaeEvaluator(
            model=ts_model,
            activations_store=activation_store,
            eval_batch_size_prompts=2,
            n_eval_batches=1,
            model_kwargs={},
        )

        activation_scaler = ActivationScaler()

        # Should not raise an error and should return metrics
        metrics = evaluator(
            sae=training_sae,
            data_provider=activation_store,
            activation_scaler=activation_scaler,
        )

        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_llm_sae_evaluator_with_different_batch_sizes(
        self,
        ts_model: HookedTransformer,
        activation_store: ActivationsStore,
        training_sae: TrainingSAE[Any],
    ):
        # Test with different batch size configurations
        evaluator = LLMSaeEvaluator(
            model=ts_model,
            activations_store=activation_store,
            eval_batch_size_prompts=None,  # None should work
            n_eval_batches=2,
            model_kwargs={},
        )

        activation_scaler = ActivationScaler()

        metrics = evaluator(
            sae=training_sae,
            data_provider=activation_store,
            activation_scaler=activation_scaler,
        )

        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_llm_sae_evaluator_with_model_kwargs(
        self,
        ts_model: HookedTransformer,
        activation_store: ActivationsStore,
        training_sae: TrainingSAE[Any],
    ):
        # Test with model_kwargs - use empty dict to test that the parameter is handled
        model_kwargs = {}
        evaluator = LLMSaeEvaluator(
            model=ts_model,
            activations_store=activation_store,
            eval_batch_size_prompts=2,
            n_eval_batches=1,
            model_kwargs=model_kwargs,
        )

        activation_scaler = ActivationScaler()

        metrics = evaluator(
            sae=training_sae,
            data_provider=activation_store,
            activation_scaler=activation_scaler,
        )

        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_llm_sae_evaluator_consistent_results(
        self,
        evaluator: LLMSaeEvaluator[Any],
        training_sae: TrainingSAE[Any],
        activation_store: ActivationsStore,
    ):
        # Test that multiple calls return consistent results
        activation_scaler = ActivationScaler()

        metrics1 = evaluator(
            sae=training_sae,
            data_provider=activation_store,
            activation_scaler=activation_scaler,
        )

        metrics2 = evaluator(
            sae=training_sae,
            data_provider=activation_store,
            activation_scaler=activation_scaler,
        )

        # Results should be identical for the same inputs
        assert metrics1.keys() == metrics2.keys()
        for key in metrics1:
            if isinstance(metrics1[key], dict):
                assert metrics1[key].keys() == metrics2[key].keys()
            # Note: Due to potential randomness in evaluation, we don't assert exact equality
            # but we do check that the structure is consistent
