import pytest
import torch

from sae_lens.training.toy_models import ReluOutputModel, ReluOutputModelCE, ToyConfig


def test_toy_sparsity_limits():
    cfg = ToyConfig(feature_probability=0.0)
    model = ReluOutputModel(cfg)
    batch = model.generate_batch(10)
    assert (batch == 0).all()

    cfg.feature_probability = 1.0
    model = ReluOutputModel(cfg)
    batch = model.generate_batch(10)
    assert (batch > 0).all()  # technically rand() can return 0...is this ok?


@pytest.mark.parametrize("n_correlated_pairs", [1, 2, 3])
def test_toy_correlated_features(n_correlated_pairs: int):
    cfg = ToyConfig(
        n_features=6,
        n_hidden=2,
        n_correlated_pairs=n_correlated_pairs,
        feature_probability=0.5,
    )
    model = ReluOutputModel(cfg)

    batch = model.generate_batch(100)
    for i in range(n_correlated_pairs):
        assert torch.eq(batch[:, i * 2] > 0, batch[:, i * 2 + 1] > 0).all()


@pytest.mark.parametrize("n_anticorrelated_pairs", [1, 2, 3])
def test_toy_anticorrelated_features(n_anticorrelated_pairs: int):
    cfg = ToyConfig(
        n_features=6,
        n_hidden=2,
        n_anticorrelated_pairs=n_anticorrelated_pairs,
        feature_probability=0.5,
    )
    model = ReluOutputModel(cfg)

    batch = model.generate_batch(100)
    for i in range(n_anticorrelated_pairs):
        assert torch.eq(batch[:, i * 2] > 0, batch[:, i * 2 + 1] == 0).all()
        assert torch.eq(batch[:, i * 2] == 0, batch[:, i * 2 + 1] > 0).all()


@pytest.mark.parametrize("n_correlated_pairs", [1, 2, 3])
@pytest.mark.parametrize("n_anticorrelated_pairs", [1, 2, 3])
def test_toy_anti_and_corr_features(
    n_correlated_pairs: int, n_anticorrelated_pairs: int
):
    cfg = ToyConfig(
        n_features=12,
        n_hidden=2,
        n_correlated_pairs=n_correlated_pairs,
        n_anticorrelated_pairs=n_anticorrelated_pairs,
        feature_probability=0.5,
    )
    model = ReluOutputModel(cfg)

    batch = model.generate_batch(100)
    for i in range(n_correlated_pairs):
        assert torch.eq(batch[:, i * 2] > 0, batch[:, i * 2 + 1] > 0).all()
    for i in range(n_anticorrelated_pairs):
        assert torch.eq(
            batch[:, n_correlated_pairs * 2 + i * 2] > 0,
            batch[:, n_correlated_pairs * 2 + i * 2 + 1] == 0,
        ).all()
        assert torch.eq(
            batch[:, n_correlated_pairs * 2 + i * 2] == 0,
            batch[:, n_correlated_pairs * 2 + i * 2 + 1] > 0,
        ).all()


def test_reluoutput_forward():
    cfg = ToyConfig(n_features=6, n_hidden=2, feature_probability=0.5)
    model = ReluOutputModel(cfg)
    with torch.inference_mode():
        model.W[0, :] = torch.tensor([1, 0, 0, 0, 0, 0])
        model.W[1, :] = torch.tensor([0, 1, 0, 0, 0, 0])
        model.W[2:, :] = 0
        model.b_final[:] = 0

    batch = model.generate_batch(100)
    expected = batch.clone()
    expected[:, 2:] = 0
    expected_hidden = torch.zeros((batch.shape[0], cfg.n_hidden))
    expected_hidden[:, 0] = batch[:, 0]
    expected_hidden[:, 1] = batch[:, 1]
    output, cache = model.run_with_cache(batch)
    assert torch.allclose(output, expected)
    assert torch.allclose(cache["hook_out_prebias"], expected)
    assert torch.allclose(cache["hook_hidden"], expected_hidden)


def test_reluoutputce_batch_shape():
    cfg = ToyConfig(n_features=5, n_hidden=2, feature_probability=0.5)
    model = ReluOutputModelCE(cfg)
    batch = model.generate_batch(100)
    assert batch.shape == (100, cfg.n_features + 1)


def test_ReluOutputModel_can_overfit_with_full_hidden_layer():
    cfg = ToyConfig(n_features=6, n_hidden=6, feature_probability=0.5)
    model = ReluOutputModel(cfg)
    batch = model.generate_batch(5)

    # before training, the model should be able to reproduce the input
    assert not torch.allclose(model(batch), batch, atol=1e-2)

    model.optimize(steps=10000)

    # after training, the model should be able to reproduce the input
    assert torch.allclose(model(batch), batch, atol=1e-2)


def test_ReluOutputModelCE_can_overfit_with_full_hidden_layer():
    cfg = ToyConfig(n_features=6, n_hidden=6, feature_probability=0.5)
    model = ReluOutputModelCE(cfg)
    batch = model.generate_batch(5)
    top_batch_feats = torch.argmax(batch, dim=-1)

    model.optimize(steps=10000)

    # after training, the model should be able to reproduce the input
    output_top_feats = torch.argmax(model(batch), dim=-1)
    assert torch.allclose(output_top_feats, top_batch_feats)
