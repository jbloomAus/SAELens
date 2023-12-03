import tempfile
from types import SimpleNamespace

import pytest
import torch
from transformer_lens import HookedTransformer

from sae_training.activations_store import ActivationsStore
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import LMSparseAutoencoderSessionloader

TEST_MODEL = "tiny-stories-1M"
TEST_DATASET = "roneneldan/TinyStories"

@pytest.fixture
def cfg():
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    # Create a mock object with the necessary attributes
    mock_config = SimpleNamespace()
    mock_config.model_name = TEST_MODEL
    mock_config.model_name = TEST_MODEL
    mock_config.hook_point = "blocks.0.hook_mlp_out"
    mock_config.hook_point_layer = 1
    mock_config.dataset_path = TEST_DATASET
    mock_config.is_dataset_tokenized = False
    mock_config.d_in = 256
    mock_config.expansion_factor = 2
    mock_config.d_sae = mock_config.d_in * mock_config.expansion_factor
    mock_config.l1_coefficient = 2e-3
    mock_config.lr = 2e-4
    mock_config.train_batch_size = 2048
    mock_config.context_size = 64
    mock_config.feature_sampling_method = None
    mock_config.feature_sampling_window = 50
    mock_config.feature_reinit_scale = 0.1
    mock_config.dead_feature_threshold = 1e-7
    mock_config.n_batches_in_buffer = 10
    mock_config.total_training_tokens = 1_000_000
    mock_config.store_batch_size = 2048
    mock_config.log_to_wandb = False
    mock_config.wandb_project = "test_project"
    mock_config.wandb_entity = "test_entity"
    mock_config.wandb_log_frequency = 10
    mock_config.device = "cuda"
    mock_config.seed = 24
    mock_config.checkpoint_path = "test/checkpoints"
    mock_config.dtype = torch.float32 

    return mock_config


def test_LMSparseAutoencoderSessionloader_init(cfg):
    
    loader = LMSparseAutoencoderSessionloader(cfg)
    assert loader.cfg == cfg
    
def test_LMSparseAutoencoderSessionloader_load_session(cfg):
    
    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sparse_autoencoder, activations_loader = loader.load_session()
    
    assert isinstance(model, HookedTransformer)
    assert isinstance(sparse_autoencoder, SparseAutoencoder)
    assert isinstance(activations_loader, ActivationsStore)


def test_LMSparseAutoencoderSessionloader_load_session_from_trained(cfg):
    
    loader = LMSparseAutoencoderSessionloader(cfg)
    _, sparse_autoencoder, _ = loader.load_session()
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        tempfile_path = f"{tmpdirname}/test.pt"
        sparse_autoencoder.save_model(tempfile_path)
        
        _, new_sparse_autoencoder, _  = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
            tempfile_path
        )

        
    assert new_sparse_autoencoder.cfg == sparse_autoencoder.cfg
    # assert weights are the same
    new_parameters = dict(new_sparse_autoencoder.named_parameters())
    for name, param in sparse_autoencoder.named_parameters():
        assert torch.allclose(param, new_parameters[name])