import os
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformer_lens import HookedTransformer

from sae_training.activations_store import ActivationsStore
from sae_training.sparse_autoencoder import SparseAutoencoder


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
        "tiny-stories-1M-resid-pre-pretokenized",
        "tiny-stories-1M-attn-out",
    ],
)
def cfg(request: pytest.FixtureRequest) -> SimpleNamespace:
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    params = request.param
    # Create a mock object with the necessary attributes
    mock_config = SimpleNamespace()
    mock_config.model_name = params["model_name"]
    mock_config.dataset_path = params["dataset_path"]
    mock_config.is_dataset_tokenized = params["tokenized"]
    mock_config.hook_point = params["hook_point"]
    mock_config.hook_point_layer = params["hook_point_layer"]
    mock_config.d_in = params["d_in"]
    mock_config.hook_point_head_index = None
    mock_config.use_cached_activations = False
    mock_config.use_ghost_grads = False
    mock_config.expansion_factor = 2
    mock_config.d_sae = mock_config.d_in * mock_config.expansion_factor
    mock_config.l1_coefficient = 2e-3
    mock_config.lp_norm = 1
    mock_config.lr = 2e-4
    mock_config.train_batch_size = 2048
    mock_config.context_size = 64
    mock_config.feature_sampling_method = None
    mock_config.feature_sampling_window = 50
    mock_config.feature_reinit_scale = 0.1
    mock_config.dead_feature_threshold = 1e-7
    mock_config.n_batches_in_buffer = 10
    mock_config.total_training_tokens = 1_000_000
    mock_config.store_batch_size = 32
    mock_config.log_to_wandb = False
    mock_config.wandb_project = "test_project"
    mock_config.wandb_entity = "test_entity"
    mock_config.wandb_log_frequency = 10
    mock_config.device = "cpu"
    mock_config.seed = 24
    mock_config.checkpoint_path = "test/checkpoints"
    # mock_config.dtype = torch.bfloat16
    mock_config.dtype = torch.float32

    return mock_config


@pytest.fixture
def sparse_autoencoder(cfg: Any):
    """
    Pytest fixture to create a mock instance of SparseAutoencoder.
    """
    return SparseAutoencoder(cfg)


@pytest.fixture
def model(cfg: SimpleNamespace):
    return HookedTransformer.from_pretrained(cfg.model_name, device="cpu")


@pytest.fixture
def activation_store(cfg: SimpleNamespace, model: HookedTransformer):
    return ActivationsStore(cfg, model)


def test_sparse_autoencoder_init(cfg: Any):
    sparse_autoencoder = SparseAutoencoder(cfg)

    assert isinstance(sparse_autoencoder, SparseAutoencoder)

    assert sparse_autoencoder.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sparse_autoencoder.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sparse_autoencoder.b_enc.shape == (cfg.d_sae,)
    assert sparse_autoencoder.b_dec.shape == (cfg.d_in,)

    # assert decoder columns have unit norm
    assert torch.allclose(
        torch.norm(sparse_autoencoder.W_dec, dim=1), torch.ones(cfg.d_sae)
    )


def test_save_model(cfg: Any):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # assert file does not exist
        assert not os.path.exists(tmpdirname + "/test.pt")

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
                state_dict_loaded["state_dict"][key],
            )


def test_load_from_pretrained_pt(cfg: Any):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # assert file does not exist
        assert not os.path.exists(tmpdirname + "/test.pt")

        sparse_autoencoder = SparseAutoencoder(cfg)
        sparse_autoencoder_state_dict = sparse_autoencoder.state_dict()
        sparse_autoencoder.save_model(tmpdirname + "/test.pt")

        assert os.path.exists(tmpdirname + "/test.pt")

        sparse_autoencoder_loaded = SparseAutoencoder.load_from_pretrained(
            tmpdirname + "/test.pt"
        )
        sparse_autoencoder_loaded.cfg.device = "cpu"  # might autoload onto mps
        sparse_autoencoder_loaded = sparse_autoencoder_loaded.to("cpu")
        sparse_autoencoder_loaded_state_dict = sparse_autoencoder_loaded.state_dict()
        # check cfg matches the original
        assert sparse_autoencoder_loaded.cfg == cfg

        # check state_dict matches the original
        for key in sparse_autoencoder.state_dict().keys():
            assert torch.allclose(
                sparse_autoencoder_state_dict[
                    key
                ],  # pylint: disable=unsubscriptable-object
                sparse_autoencoder_loaded_state_dict[
                    key
                ],  # pylint: disable=unsubscriptable-object
            )


def test_load_from_pretrained_pkl_gz(cfg: Any):
    with tempfile.TemporaryDirectory() as tmpdirname:
        # assert file does not exist
        assert not os.path.exists(tmpdirname + "/test.pkl.gz")

        sparse_autoencoder = SparseAutoencoder(cfg)
        sparse_autoencoder_state_dict = sparse_autoencoder.state_dict()
        sparse_autoencoder.save_model(tmpdirname + "/test.pkl.gz")

        assert os.path.exists(tmpdirname + "/test.pkl.gz")

        sparse_autoencoder_loaded = SparseAutoencoder.load_from_pretrained(
            tmpdirname + "/test.pkl.gz"
        )
        sparse_autoencoder_loaded.cfg.device = "cpu"  # might autoload onto mps
        sparse_autoencoder_loaded = sparse_autoencoder_loaded.to("cpu")
        sparse_autoencoder_loaded_state_dict = sparse_autoencoder_loaded.state_dict()
        # check cfg matches the original
        assert sparse_autoencoder_loaded.cfg == cfg

        # check state_dict matches the original
        for key in sparse_autoencoder.state_dict().keys():
            assert torch.allclose(
                sparse_autoencoder_state_dict[
                    key
                ],  # pylint: disable=unsubscriptable-object
                sparse_autoencoder_loaded_state_dict[
                    key
                ],  # pylint: disable=unsubscriptable-object
            )


def test_sparse_autoencoder_forward(sparse_autoencoder: SparseAutoencoder):
    batch_size = 32
    d_in = sparse_autoencoder.d_in
    d_sae = sparse_autoencoder.d_sae

    x = torch.randn(batch_size, d_in)
    (
        sae_out,
        feature_acts,
        loss,
        mse_loss,
        l1_loss,
        _ghost_grad_loss,
    ) = sparse_autoencoder.forward(
        x,
    )

    assert sae_out.shape == (batch_size, d_in)
    assert feature_acts.shape == (batch_size, d_sae)
    assert loss.shape == ()
    assert mse_loss.shape == ()
    assert l1_loss.shape == ()
    assert torch.allclose(loss, mse_loss + l1_loss)

    x_centred = x - x.mean(dim=0, keepdim=True)
    expected_mse_loss = (
        torch.pow((sae_out - x.float()), 2)
        / (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
    ).mean()
    assert torch.allclose(mse_loss, expected_mse_loss)
    expected_l1_loss = torch.abs(feature_acts).sum(dim=1).mean(dim=(0,))
    assert torch.allclose(l1_loss, sparse_autoencoder.l1_coefficient * expected_l1_loss)

    # check everything has the right dtype
    assert sae_out.dtype == sparse_autoencoder.dtype
    assert feature_acts.dtype == sparse_autoencoder.dtype
    assert loss.dtype == sparse_autoencoder.dtype
    assert mse_loss.dtype == sparse_autoencoder.dtype
    assert l1_loss.dtype == sparse_autoencoder.dtype
