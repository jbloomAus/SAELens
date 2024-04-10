import os
import tempfile
from typing import Any

import pytest
import torch
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.config import LanguageModelSAERunnerConfig

# from sae_lens.training.sae_group import SAETrainingGroup
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.training.sparse_autoencoder import SparseAutoencoder

TEST_MODEL = "tiny-stories-1M"
TEST_DATASET = "roneneldan/TinyStories"


@pytest.fixture
def cfg():
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # # Create a mock object with the necessary attributes
    # mock_config = SimpleNamespace()
    # mock_config.model_name = TEST_MODEL
    # mock_config.hook_point = "blocks.0.hook_mlp_out"
    # mock_config.hook_point_layer = 0
    # mock_config.dataset_path = TEST_DATASET
    # mock_config.is_dataset_tokenized = False
    # mock_config.d_in = 64
    # mock_config.expansion_factor = 2
    # mock_config.d_sae = mock_config.d_in * mock_config.expansion_factor
    # mock_config.l1_coefficient = 2e-3
    # mock_config.lr = 2e-4
    # mock_config.train_batch_size = 512
    # mock_config.context_size = 64
    # mock_config.feature_sampling_method = None
    # mock_config.feature_sampling_window = 50
    # mock_config.feature_reinit_scale = 0.2
    # mock_config.dead_feature_threshold = 1e-7
    # mock_config.n_batches_in_buffer = 2
    # mock_config.total_training_tokens = 1_000_000
    # mock_config.store_batch_size = 128
    # mock_config.log_to_wandb = False
    # mock_config.wandb_project = "test_project"
    # mock_config.wandb_entity = "test_entity"
    # mock_config.wandb_log_frequency = 10
    # mock_config.device = "cpu"
    # mock_config.seed = 24
    # mock_config.checkpoint_path = "test/checkpoints"
    # mock_config.dtype = torch.float32
    # mock_config.use_cached_activations = False
    # mock_config.hook_point_head_index = None

    cfg = LanguageModelSAERunnerConfig(
        model_name=TEST_MODEL,
        hook_point="blocks.0.hook_mlp_out",
        hook_point_layer=0,
        dataset_path=TEST_DATASET,
        is_dataset_tokenized=False,
        d_in=64,
        expansion_factor=2,
        l1_coefficient=2e-3,
        lr=2e-4,
        train_batch_size=512,
        context_size=64,
        feature_sampling_window=50,
        dead_feature_threshold=1e-7,
        n_batches_in_buffer=2,
        total_training_tokens=1_000_000,
        store_batch_size=128,
        log_to_wandb=False,
        wandb_project="test_project",
        wandb_entity="test_entity",
        wandb_log_frequency=10,
        device="cpu",
        seed=24,
        checkpoint_path="test/checkpoints",
        dtype=torch.float32,
        use_cached_activations=False,
        hook_point_head_index=None,
    )

    return cfg


def test_LMSparseAutoencoderSessionloader_init(cfg: Any):
    loader = LMSparseAutoencoderSessionloader(cfg)
    assert loader.cfg == cfg


def test_LMSparseAutoencoderSessionloader_load_session(cfg: Any):
    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sae_group, activations_loader = loader.load_sae_training_group_session()

    assert isinstance(model, HookedTransformer)
    assert isinstance(sae_group.autoencoders[0], SparseAutoencoder)
    assert isinstance(activations_loader, ActivationsStore)


def test_LMSparseAutoencoderSessionloader_load_sae_session_from_pretrained(
    cfg: Any,
):
    # make a
    loader = LMSparseAutoencoderSessionloader(cfg)
    _, sae_group, _ = loader.load_sae_training_group_session()
    old_sparse_autoencoder = sae_group.autoencoders[0]
    with tempfile.TemporaryDirectory() as tmpdirname:
        sae_group.save_model(tmpdirname)
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
