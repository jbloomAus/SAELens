import os
import tempfile

import pytest
import torch
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.config import LanguageModelSAERunnerConfig
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


def test_LMSparseAutoencoderSessionloader_init(cfg: LanguageModelSAERunnerConfig):
    loader = LMSparseAutoencoderSessionloader(cfg)
    assert loader.cfg == cfg


def test_LMSparseAutoencoderSessionloader_load_session(
    cfg: LanguageModelSAERunnerConfig,
):
    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sparse_autoencoder, activations_loader = loader.load_sae_session()

    assert isinstance(model, HookedTransformer)
    assert isinstance(sparse_autoencoder, SparseAutoencoder)
    assert isinstance(activations_loader, ActivationsStore)
    assert model.cfg.checkpoint_index is None


def test_LMSparseAutoencoderSessionloader_load_session_can_load_model_with_kwargs():
    cfg = build_sae_cfg(
        model_name="pythia-14m",
        hook_point="blocks.0.hook_mlp_out",
        hook_point_layer=0,
        dataset_path="roneneldan/TinyStories",
        is_dataset_tokenized=False,
        model_from_pretrained_kwargs={"checkpoint_index": 0},
    )
    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sparse_autoencoder, activations_loader = loader.load_sae_session()

    assert isinstance(model, HookedTransformer)
    assert isinstance(sparse_autoencoder, SparseAutoencoder)
    assert isinstance(activations_loader, ActivationsStore)
    assert (
        model.cfg.checkpoint_index
        == cfg.model_from_pretrained_kwargs["checkpoint_index"]
    )


def test_LMSparseAutoencoderSessionloader_load_sae_session_from_pretrained(
    cfg: LanguageModelSAERunnerConfig,
):
    # make a
    loader = LMSparseAutoencoderSessionloader(cfg)
    _, old_sparse_autoencoder, _ = loader.load_sae_session()

    with tempfile.TemporaryDirectory() as tmpdirname:
        old_sparse_autoencoder.save_model(tmpdirname)
        (
            _,
            new_sparse_autoencoder,
            _,
        ) = LMSparseAutoencoderSessionloader.load_pretrained_sae(tmpdirname)
    new_sparse_autoencoder.cfg.device = torch.device("cpu")
    new_sparse_autoencoder.to("cpu")

    # don't care about verbose or the checkpoint_path
    old_sparse_autoencoder.cfg.verbose = new_sparse_autoencoder.cfg.verbose
    old_sparse_autoencoder.cfg.checkpoint_path = (
        new_sparse_autoencoder.cfg.checkpoint_path
    )

    assert new_sparse_autoencoder.cfg == old_sparse_autoencoder.cfg
    # assert weights are the same
    new_parameters = dict(old_sparse_autoencoder.named_parameters())
    for name, param in old_sparse_autoencoder.named_parameters():
        assert torch.allclose(param, new_parameters[name])


def test_load_pretrained_sae_from_huggingface():
    layer = 8  # pick a layer you want.

    GPT2_SMALL_RESIDUAL_SAES_REPO_ID = "jbloom/GPT2-Small-SAEs-Reformatted"
    hook_point = f"blocks.{layer}.hook_resid_pre"

    FILENAME = f"{hook_point}/cfg.json"
    path = hf_hub_download(repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, filename=FILENAME)

    FILENAME = f"{hook_point}/sae_weights.safetensors"
    hf_hub_download(repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, filename=FILENAME)

    FILENAME = f"{hook_point}/sparsity.safetensors"
    hf_hub_download(repo_id=GPT2_SMALL_RESIDUAL_SAES_REPO_ID, filename=FILENAME)

    folder_path = os.path.dirname(path)

    model, sae, activation_store = LMSparseAutoencoderSessionloader.load_pretrained_sae(
        path=folder_path
    )
    assert isinstance(model, HookedTransformer)
    assert isinstance(sae, SparseAutoencoder)
    assert isinstance(activation_store, ActivationsStore)
    assert sae.cfg.hook_point_layer == layer
    assert sae.cfg.model_name == "gpt2-small"
