import json
from pathlib import Path

import pytest
from transformer_lens import HookedTransformer

from sae_lens import __version__
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.llm_sae_training_runner import (
    LanguageModelSAETrainingRunner,
    _parse_cfg_args,
)
from sae_lens.saes.gated_sae import GatedTrainingSAEConfig
from sae_lens.saes.jumprelu_sae import JumpReLUTrainingSAEConfig
from sae_lens.saes.sae import SAE
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
from sae_lens.saes.topk_sae import TopKTrainingSAEConfig
from tests.helpers import (
    ALL_ARCHITECTURES,
    NEEL_NANDA_C4_10K_DATASET,
    TINYSTORIES_MODEL,
    build_runner_cfg_for_arch,
)


@pytest.mark.parametrize("architecture", ALL_ARCHITECTURES)
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
