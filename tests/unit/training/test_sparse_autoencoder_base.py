import os
from pathlib import Path
from typing import Any

import pytest
import torch

from sae_lens import __version__
from sae_lens.training.sparse_autoencoder import SparseAutoencoderBase
from tests.unit.helpers import build_sae_cfg


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "tokenized": False,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_point": "blocks.1.attn.hook_z",
            "hook_point_layer": 1,
            "d_in": 64,
        },
    ],
    ids=[
        "tiny-stories-1M-resid-pre",
        "tiny-stories-1M-resid-pre-L1-W-dec-Norm",
        "tiny-stories-1M-resid-pre-pretokenized",
        "tiny-stories-1M-attn-out",
    ],
)
def cfg(request: pytest.FixtureRequest):
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    params = request.param
    return build_sae_cfg(**params)


def test_sparse_autoencoder_init(cfg: Any):
    sparse_autoencoder = SparseAutoencoderBase(cfg)

    assert isinstance(sparse_autoencoder, SparseAutoencoderBase)

    assert sparse_autoencoder.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sparse_autoencoder.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sparse_autoencoder.b_enc.shape == (cfg.d_sae,)
    assert sparse_autoencoder.b_dec.shape == (cfg.d_in,)


def test_SparseAutoencoder_save_and_load_from_pretrained(tmp_path: Path) -> None:
    cfg = build_sae_cfg(device="cpu")
    model_path = str(tmp_path)
    sparse_autoencoder = SparseAutoencoderBase(cfg)
    sparse_autoencoder_state_dict = sparse_autoencoder.state_dict()
    sparse_autoencoder.save_model(model_path)

    assert os.path.exists(model_path)

    sparse_autoencoder_loaded = SparseAutoencoderBase.load_from_pretrained(
        model_path, device="cpu"
    )
    sparse_autoencoder_loaded.cfg.verbose = True
    sparse_autoencoder_loaded.cfg.checkpoint_path = cfg.checkpoint_path
    sparse_autoencoder_loaded_state_dict = sparse_autoencoder_loaded.state_dict()
    # check cfg matches the original
    assert sparse_autoencoder_loaded.cfg == cfg

    # check state_dict matches the original
    for key in sparse_autoencoder.state_dict().keys():
        assert torch.allclose(
            sparse_autoencoder_state_dict[key],
            sparse_autoencoder_loaded_state_dict[key],
        )


def test_SparseAutoencoder_save_and_load_from_pretrained_update_the_sae_lens_version_on_load(
    tmp_path: Path,
):
    cfg = build_sae_cfg(
        device="cpu", sae_lens_training_version="9999.0.0", sae_lens_version="9999.0.0"
    )
    model_path = str(tmp_path)
    sparse_autoencoder = SparseAutoencoderBase(cfg)
    sparse_autoencoder.save_model(model_path)

    assert os.path.exists(model_path)

    sparse_autoencoder_loaded = SparseAutoencoderBase.load_from_pretrained(
        model_path, device="cpu"
    )
    # the sae_lens_version should update, but not the training version unless we actually run training
    assert sparse_autoencoder_loaded.cfg.sae_lens_version == __version__
    assert sparse_autoencoder_loaded.cfg.sae_lens_training_version == "9999.0.0"


def test_SparseAutoencoder_save_and_load_from_pretrained_lacks_scaling_factor(
    tmp_path: Path,
) -> None:
    cfg = build_sae_cfg(device="cpu")
    model_path = str(tmp_path)
    sparse_autoencoder = SparseAutoencoderBase(cfg)
    sparse_autoencoder_state_dict = sparse_autoencoder.state_dict()
    # sometimes old state dicts will be missing the scaling factor
    del sparse_autoencoder_state_dict["scaling_factor"]  # = torch.tensor(0.0)
    sparse_autoencoder.save_model(model_path)

    assert os.path.exists(model_path)

    sparse_autoencoder_loaded = SparseAutoencoderBase.load_from_pretrained(model_path)
    sparse_autoencoder_loaded.cfg.verbose = True
    sparse_autoencoder_loaded.cfg.checkpoint_path = cfg.checkpoint_path
    sparse_autoencoder_loaded = sparse_autoencoder_loaded.to("cpu")
    sparse_autoencoder_loaded_state_dict = sparse_autoencoder_loaded.state_dict()
    # check cfg matches the original
    assert sparse_autoencoder_loaded.cfg == cfg

    # check state_dict matches the original
    for key in sparse_autoencoder.state_dict().keys():
        if key == "scaling_factor":
            assert isinstance(cfg.d_sae, int)
            assert torch.allclose(
                torch.ones(cfg.d_sae, dtype=cfg.dtype, device=cfg.device),
                sparse_autoencoder_loaded_state_dict[key],
            )
        else:
            assert torch.allclose(
                sparse_autoencoder_state_dict[key],
                sparse_autoencoder_loaded_state_dict[key],
            )


def test_SparseAutoencoder_get_name_returns_correct_name_from_cfg_vals() -> None:
    cfg = build_sae_cfg(
        model_name="test_model", hook_point="test_hook_point", d_sae=128
    )
    sae = SparseAutoencoderBase(cfg)
    assert sae.get_name() == "sparse_autoencoder_test_model_test_hook_point_128"
