from collections.abc import Iterable
from types import SimpleNamespace

import pytest
import torch
from datasets import IterableDataset
from transformer_lens import HookedTransformer

from sae_training.activations_store import ActivationsStore


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
            "model_name": "gelu-2l",
            "dataset_path": "NeelNanda/c4-tokenized-2b",
            "tokenized": True,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 512,
        },
        {
            "model_name": "gpt2",
            "dataset_path": "apollo-research/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
            "tokenized": True,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 768,
        },
        {
            "model_name": "gpt2",
            "dataset_path": "Skylion007/openwebtext",
            "tokenized": False,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 768,
        },
    ],
    ids=["tiny-stories-1M", "gelu-2l-tokenized", "gpt2-tokenized", "gpt2"],
)
def cfg(request: pytest.FixtureRequest) -> SimpleNamespace:
    # This function will be called with each parameter set
    params = request.param
    mock_config = SimpleNamespace()
    mock_config.model_name = params["model_name"]
    mock_config.dataset_path = params["dataset_path"]
    mock_config.is_dataset_tokenized = params["tokenized"]
    mock_config.hook_point = params["hook_point"]
    mock_config.hook_point_layer = params["hook_point_layer"]
    mock_config.d_in = params["d_in"]
    mock_config.expansion_factor = 2
    mock_config.d_sae = mock_config.d_in * mock_config.expansion_factor
    mock_config.l1_coefficient = 2e-3
    mock_config.lr = 2e-4
    mock_config.train_batch_size = 32
    mock_config.context_size = 16
    mock_config.use_cached_activations = False
    mock_config.hook_point_head_index = None

    mock_config.feature_sampling_method = None
    mock_config.feature_sampling_window = 50
    mock_config.feature_reinit_scale = 0.1
    mock_config.dead_feature_threshold = 1e-7

    mock_config.n_batches_in_buffer = 4
    mock_config.total_training_tokens = 1_000_000
    mock_config.store_batch_size = 32

    mock_config.log_to_wandb = False
    mock_config.wandb_project = "test_project"
    mock_config.wandb_entity = "test_entity"
    mock_config.wandb_log_frequency = 10
    mock_config.device = torch.device("cpu")
    mock_config.seed = 24
    mock_config.checkpoint_path = "test/checkpoints"
    mock_config.dtype = torch.float32

    return mock_config


@pytest.fixture
def model(cfg: SimpleNamespace):
    return HookedTransformer.from_pretrained(cfg.model_name, device="cpu")


@pytest.fixture
def activation_store(cfg: SimpleNamespace, model: HookedTransformer):
    return ActivationsStore(cfg, model)


@pytest.fixture
def activation_store_head_hook(
    cfg_head_hook: SimpleNamespace, model: HookedTransformer
):
    return ActivationsStore(cfg_head_hook, model)


def test_activations_store__init__(cfg: SimpleNamespace, model: HookedTransformer):
    store = ActivationsStore(cfg, model)

    assert store.cfg == cfg
    assert store.model == model

    assert isinstance(store.dataset, IterableDataset)
    assert isinstance(store.iterable_dataset, Iterable)

    # I expect the dataloader to be initialised
    assert hasattr(store, "dataloader")

    # I expect the buffer to be initialised
    assert hasattr(store, "storage_buffer")

    # the rest is in the dataloader.
    expected_size = (
        cfg.store_batch_size * cfg.context_size * cfg.n_batches_in_buffer // 2
    )
    assert store.storage_buffer.shape == (expected_size, 1, cfg.d_in)


def test_activations_store__get_batch_tokens(activation_store: ActivationsStore):
    batch = activation_store.get_batch_tokens()

    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (
        activation_store.cfg.store_batch_size,
        activation_store.cfg.context_size,
    )
    assert batch.device == activation_store.cfg.device


def test_activations_score_get_next_batch(
    model: HookedTransformer, activation_store: ActivationsStore
):

    batch = activation_store.get_batch_tokens()
    assert batch.shape == (
        activation_store.cfg.store_batch_size,
        activation_store.cfg.context_size,
    )

    # if model.tokenizer.bos_token_id is not None:
    #     torch.testing.assert_close(
    #         batch[:, 0], torch.ones_like(batch[:, 0]) * model.tokenizer.bos_token_id
    #     )


def test_activations_store__get_activations(activation_store: ActivationsStore):
    batch = activation_store.get_batch_tokens()
    activations = activation_store.get_activations(batch)

    cfg = activation_store.cfg
    assert isinstance(activations, torch.Tensor)
    assert activations.shape == (cfg.store_batch_size, cfg.context_size, 1, cfg.d_in)
    assert activations.device == cfg.device


def test_activations_store__get_buffer(activation_store: ActivationsStore):
    n_batches_in_buffer = 3
    buffer = activation_store.get_buffer(n_batches_in_buffer)

    cfg = activation_store.cfg
    assert isinstance(buffer, torch.Tensor)
    buffer_size_expected = cfg.store_batch_size * cfg.context_size * n_batches_in_buffer

    assert buffer.shape == (buffer_size_expected, 1, cfg.d_in)
    assert buffer.device == cfg.device
