import tempfile
from typing import Any

import pytest
import torch
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.sae_group import SAEGroup
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


def test_load_pretrained_sae_from_huggingface():
    layer = 8  # pick a layer you want.
    REPO_ID = "jbloom/GPT2-Small-SAEs"
    FILENAME = (
        f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"
    )
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model, sae_group, activation_store = (
        LMSparseAutoencoderSessionloader.load_session_from_pretrained(path=path)
    )
    assert isinstance(model, HookedTransformer)
    assert isinstance(sae_group, SAEGroup)
    assert isinstance(activation_store, ActivationsStore)
    assert len(sae_group) == 1
    assert sae_group.cfg.hook_point_layer == layer
    assert sae_group.cfg.model_name == "gpt2-small"
