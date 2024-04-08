import tempfile
from typing import Any

import pytest
import torch
from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from tests.unit.helpers import build_sae_cfg

TEST_MODEL = "tiny-stories-1M"
TEST_DATASET = "roneneldan/TinyStories"


@pytest.fixture
def cfg():
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    cfg = build_sae_cfg(
        model_name=TEST_MODEL,
        hook_point="blocks.0.hook_mlp_out",
        hook_point_layer=0,
        dataset_path=TEST_DATASET,
        is_dataset_tokenized=False,
    )
    return cfg


def test_LMSparseAutoencoderSessionloader_init(cfg: Any):
    loader = LMSparseAutoencoderSessionloader(cfg)
    assert loader.cfg == cfg


def test_LMSparseAutoencoderSessionloader_load_session(cfg: Any):
    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sae_group, activations_loader = loader.load_session()

    assert isinstance(model, HookedTransformer)
    assert isinstance(sae_group.autoencoders[0], SparseAutoencoder)
    assert isinstance(activations_loader, ActivationsStore)


def test_LMSparseAutoencoderSessionloader_load_session_from_trained(cfg: Any):
    loader = LMSparseAutoencoderSessionloader(cfg)
    _, sae_group, _ = loader.load_session()

    with tempfile.TemporaryDirectory() as tmpdirname:
        tempfile_path = f"{tmpdirname}/test.pt"
        sae_group.save_model(tempfile_path)

        (
            _,
            new_sae_group,
            _,
        ) = LMSparseAutoencoderSessionloader.load_session_from_pretrained(tempfile_path)
    new_sae_group.cfg.device = torch.device("cpu")
    new_sae_group.to("cpu")
    assert new_sae_group.cfg == sae_group.cfg
    # assert weights are the same
    new_parameters = dict(new_sae_group.autoencoders[0].named_parameters())
    for name, param in sae_group.autoencoders[0].named_parameters():
        assert torch.allclose(param, new_parameters[name])
