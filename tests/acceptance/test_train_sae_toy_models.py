import einops
import pytest
import torch

import wandb
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.toy_models import Config as ToyConfig
from sae_training.toy_models import Model as ToyModel
from sae_training.train_sae_on_toy_model import train_toy_sae


@pytest.fixture
def model():
    
    cfg = ToyConfig(
        n_instances = 1,
        n_features = 5,
        n_hidden = 2,
    )
    model = ToyModel(
        cfg = cfg,
        device = "cpu",
        feature_probability = 0.025,
    )
    model.optimize(steps=5_000)
    return model


def test_train_sae_toy_models(model):
    

    batch = model.generate_batch(25000)
    hidden = einops.einsum(batch, model.W, "batch_size instances features, instances hidden features -> batch_size instances hidden")

    toy_config = {
        "d_in": 2,
        "d_sae": 5,
        "dtype": torch.float32,
        "device": "cpu",
    }

    sae = SparseAutoencoder(toy_config)
    # wandb.init(project="sae-training-test", config=toy_config)
    sae = train_toy_sae(
        model, sae, hidden.detach().squeeze(), use_wandb=False, batch_size=32, n_epochs=10)
    # wandb.finish()


    batch = model.generate_batch(2500)
    hidden = einops.einsum(batch, model.W, "batch_size instances features, instances hidden features -> batch_size instances hidden")
    sae_out, hidden_post = sae.forward(hidden)

    batch_size = batch.shape[0]
    mse_loss = ((sae_out - hidden)**2).mean().item()
    l0 =  ((hidden_post != 0) / batch_size).sum().item()
    # expected_l0 = model.feature_probability.sum().item()

    assert mse_loss < 1e-3
    # assert l0 < 2