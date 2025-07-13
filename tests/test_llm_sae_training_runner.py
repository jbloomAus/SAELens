import json
from pathlib import Path

import pytest
from transformer_lens import HookedTransformer

from sae_lens import __version__
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner
from sae_lens.saes.sae import SAE
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
    original_metadat_dict = json.loads(json.dumps(sae.cfg.metadata.__dict__))
    assert loaded_sae.cfg.architecture() == architecture
    assert loaded_sae.cfg.d_in == sae.cfg.d_in
    assert loaded_sae.cfg.d_sae == sae.cfg.d_sae
    assert loaded_sae.cfg.dtype == sae.cfg.dtype
    assert loaded_sae.cfg.device == sae.cfg.device
    assert loaded_sae.cfg.apply_b_dec_to_input == sae.cfg.apply_b_dec_to_input
    assert loaded_sae.cfg.metadata.__dict__ == original_metadat_dict
