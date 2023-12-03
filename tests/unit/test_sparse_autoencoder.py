import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from sae_training.sparse_autoencoder import SparseAutoencoder

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
    mock_config.device = "cpu"
    mock_config.seed = 24
    mock_config.checkpoint_path = "test/checkpoints"
    mock_config.dtype = torch.float32 

    return mock_config

def test_sparse_autoencoder_init(cfg):
    
    sparse_autoencoder = SparseAutoencoder(cfg)
    
    assert isinstance(sparse_autoencoder, SparseAutoencoder)
    
    assert sparse_autoencoder.W_enc.shape == (cfg.d_in, cfg.d_sae) 
    assert sparse_autoencoder.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sparse_autoencoder.b_enc.shape == (cfg.d_sae,)
    assert sparse_autoencoder.b_dec.shape == (cfg.d_in,)
    
    # assert decoder columns have unit norm
    assert torch.allclose(
        torch.norm(sparse_autoencoder.W_dec, dim=1), 
        torch.ones(cfg.d_sae)
    )

def test_save_model(cfg):
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        # assert file does not exist
        assert os.path.exists(tmpdirname + "/test.pt") == False
        
        sparse_autoencoder = SparseAutoencoder(cfg)
        sparse_autoencoder.save_model(tmpdirname + "/test.pt")
        
        assert os.path.exists(tmpdirname + "/test.pt")
        
        state_dict_original = sparse_autoencoder.state_dict()
        state_dict_loaded = torch.load(tmpdirname + "/test.pt")
        
        # check for cfg and state_dict keys
        assert "cfg" in state_dict_loaded
        assert "state_dict" in state_dict_loaded
        
        # check cfg matches the original
        assert state_dict_loaded["cfg"] == cfg
        
        # check state_dict matches the original
        for key in sparse_autoencoder.state_dict().keys():
            assert torch.allclose(
                state_dict_original[key],  # pylint: disable=unsubscriptable-object
                state_dict_loaded["state_dict"][key]
            )

def test_load_from_pretrained(cfg):
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        # assert file does not exist
        assert os.path.exists(tmpdirname + "/test.pt") == False
        
        sparse_autoencoder = SparseAutoencoder(cfg)
        sparse_autoencoder_state_dict = sparse_autoencoder.state_dict()
        sparse_autoencoder.save_model(tmpdirname + "/test.pt")
        
        assert os.path.exists(tmpdirname + "/test.pt")
        
        sparse_autoencoder_loaded = SparseAutoencoder.load_from_pretrained(tmpdirname + "/test.pt")
        sparse_autoencoder_loaded_state_dict = sparse_autoencoder_loaded.state_dict()
        # check cfg matches the original
        assert sparse_autoencoder_loaded.cfg == cfg
        
        # check state_dict matches the original
        for key in sparse_autoencoder.state_dict().keys():
            assert torch.allclose(
                sparse_autoencoder_state_dict[key],  # pylint: disable=unsubscriptable-object
                sparse_autoencoder_loaded_state_dict[key] # pylint: disable=unsubscriptable-object
            )
        
    
@pytest.mark.skip("TODO")
def test_sparse_autoencoder_forward(cfg):
    pass # TODO

@pytest.mark.skip("TODO")
def test_sparse_autoencoder_resample_neurons(cfg):
    pass # TODO

@pytest.mark.skip("TODO")
def test_sparse_eautoencoder_remove_gradient_parallel_to_decoder_directions(cfg):
    pass # TODO

