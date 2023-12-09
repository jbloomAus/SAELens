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

@pytest.fixture
def sparse_autoencoder(cfg):
    """
    Pytest fixture to create a mock instance of SparseAutoencoder.
    """
    return SparseAutoencoder(cfg)

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
        sparse_autoencoder_loaded.cfg.device = "cpu" # might autoload onto mps
        sparse_autoencoder_loaded = sparse_autoencoder_loaded.to("cpu")
        sparse_autoencoder_loaded_state_dict = sparse_autoencoder_loaded.state_dict()
        # check cfg matches the original
        assert sparse_autoencoder_loaded.cfg == cfg
        
        # check state_dict matches the original
        for key in sparse_autoencoder.state_dict().keys():
            assert torch.allclose(
                sparse_autoencoder_state_dict[key],  # pylint: disable=unsubscriptable-object
                sparse_autoencoder_loaded_state_dict[key] # pylint: disable=unsubscriptable-object
            )
        
    
def test_sparse_autoencoder_forward(sparse_autoencoder):
    
    batch_size = 32
    d_in =sparse_autoencoder.d_in
    d_sae = sparse_autoencoder.d_sae
    
    x = torch.randn(batch_size, d_in)
    sae_out, feature_acts, loss, mse_loss, l1_loss = sparse_autoencoder.forward(
        x,
    )
    
    assert sae_out.shape == (batch_size, d_in)
    assert feature_acts.shape == (batch_size, d_sae)
    assert loss.shape == ()
    assert mse_loss.shape == ()
    assert l1_loss.shape == ()
    assert torch.allclose(loss, mse_loss + l1_loss)
    
    assert torch.allclose(mse_loss, (sae_out.float() - x.float()).pow(2).sum(-1).mean(0))
    assert torch.allclose(l1_loss, sparse_autoencoder.l1_coefficient * torch.abs(feature_acts).sum())
    
    # check everything has the right dtype
    assert sae_out.dtype == sparse_autoencoder.dtype
    assert feature_acts.dtype == sparse_autoencoder.dtype
    assert loss.dtype == sparse_autoencoder.dtype
    assert mse_loss.dtype == sparse_autoencoder.dtype
    assert l1_loss.dtype == sparse_autoencoder.dtype

def test_sparse_autoencoder_resample_neurons(sparse_autoencoder):
    
    batch_size = 32
    d_in =sparse_autoencoder.d_in
    d_sae = sparse_autoencoder.d_sae

    x = torch.randn(batch_size, d_in)
    feature_sparsity = torch.exp((torch.randn(d_sae) - 17))
    neuron_resample_scale = 0.2
    optimizer = torch.optim.Adam(sparse_autoencoder.parameters(), lr=1e-4)
    
    # set weight of the sparse autoencoder to be non-zero (and not have unit norm)
    sparse_autoencoder.W_enc.data = torch.randn(d_in, d_sae)*10
    sparse_autoencoder.W_dec.data = torch.randn(d_sae, d_in)*10
    sparse_autoencoder.b_enc.data = torch.randn(d_sae)*10
    sparse_autoencoder.b_dec.data = torch.randn(d_in)*10
    
    # Set optimizer state so we can tell when it is reset:
    dummy_value = 5.0
    for dict_idx, (k, v) in enumerate(optimizer.state.items()):
            for v_key in ["exp_avg", "exp_avg_sq"]:
                if dict_idx == 0: # W_enc
                    assert k.data.shape == (d_in, d_sae)
                    v[v_key] = dummy_value
                elif dict_idx == 1: # b_enc
                    assert k.data.shape == (d_sae,)
                    v[v_key] = dummy_value
                elif dict_idx == 2: # W_dec
                    assert k.data.shape == (d_sae, d_in)
                    v[v_key]= dummy_value
                elif dict_idx == 3: # b_dec
                    assert k.data.shape == (d_in,)
    is_dead = feature_sparsity < sparse_autoencoder.cfg.dead_feature_threshold
    alive_neurons = feature_sparsity >= sparse_autoencoder.cfg.dead_feature_threshold
    
    
    n_resampled_neurons = sparse_autoencoder.resample_neurons(x, feature_sparsity, neuron_resample_scale, optimizer)
    
    # want to check the following:
    # 1. that the number of neurons reset is equal to the number of neurons that should be reset
    assert n_resampled_neurons == is_dead.sum().item()
    
    # 2. for each neuron we reset:
    #   a. the bias is zero
    assert torch.allclose(
        sparse_autoencoder.b_enc.data[is_dead],
        torch.zeros_like(sparse_autoencoder.b_enc.data[is_dead]))
    #   b. the encoder weights have norm 0.2 * average of other weights. 
    mean_decoder_norm = sparse_autoencoder.W_enc[:, alive_neurons].norm(dim=0).mean().item()
    assert torch.allclose(
        sparse_autoencoder.W_enc[:, is_dead].norm(dim=0),
        torch.ones(n_resampled_neurons) * 0.2 * mean_decoder_norm
    )
    # c. the decoder weights have unit norm
    assert torch.allclose(
        sparse_autoencoder.W_dec[is_dead, :].norm(dim=1),
        torch.ones(n_resampled_neurons)
    )
    
    # d. the Adam parameters are reset
    for dict_idx, (k, v) in enumerate(optimizer.state.items()):
        for v_key in ["exp_avg", "exp_avg_sq"]:
            if dict_idx == 0:
                if k.data.shape != (d_in, d_sae):
                    print(
                        "Warning: it does not seem as if resetting the Adam parameters worked, there are shapes mismatches"
                    )
                if v[v_key][:, is_dead].abs().max().item() > 1e-6:
                    print(
                        "Warning: it does not seem as if resetting the Adam parameters worked"
                    )
        
    # e. check that the decoder weights for reset neurons match the encoder weights for reset neurons
    # (given both are normalized)
    assert torch.allclose(
        (sparse_autoencoder.W_enc[:, is_dead] / sparse_autoencoder.W_enc[:, is_dead].norm(dim=0)).T,
        sparse_autoencoder.W_dec[is_dead, :] / sparse_autoencoder.W_dec[is_dead, :].norm(dim=1).unsqueeze(1)
    )

@pytest.mark.skip("TODO")
def test_sparse_eautoencoder_remove_gradient_parallel_to_decoder_directions(cfg):
    pass # TODO



