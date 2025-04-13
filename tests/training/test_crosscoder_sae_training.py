from typing import Any

import einops
import pytest
import torch
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.training_crosscoder_sae import (
    TrainingCrosscoderSAE,
    TrainingCrosscoderSAEConfig
)
from tests.helpers import build_sae_cfg


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layers": [1,2,3],
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layers": [1,2,3],
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layers": [1,2,3],
            "d_in": 64,
            "normalize_activations": "constant_norm_rescale",
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
    ],
    ids=[
        "tiny-stories-1M-resid-pre",
        "tiny-stories-1M-resid-pre-pretokenized",
        "tiny-stories-1M-resid-pre-pretokenized-norm-rescale",
    ],
)
def cfg(request: pytest.FixtureRequest):
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    params = request.param
    return build_sae_cfg(**params)


@pytest.fixture
def training_crosscoder_sae(cfg: LanguageModelSAERunnerConfig):
    """
    Pytest fixture to create a mock instance of SparseAutoencoder.
    """
    return TrainingCrosscoderSAE(
        TrainingCrosscoderSAEConfig.from_sae_runner_config(cfg),
        use_error_term=True)


@pytest.fixture
def activation_store(model: HookedTransformer, cfg: LanguageModelSAERunnerConfig):
    return ActivationsStore.from_config(
        model, cfg, override_dataset=Dataset.from_list([{"text": "hello world"}] * 2000)
    )


@pytest.fixture
def model(cfg: LanguageModelSAERunnerConfig):
    return HookedTransformer.from_pretrained(cfg.model_name, device="cpu")


# todo: remove the need for this fixture
@pytest.fixture
def trainer(
    cfg: LanguageModelSAERunnerConfig,
    training_crosscoder_sae: TrainingCrosscoderSAE,
    model: HookedTransformer,
    activation_store: ActivationsStore,
):
    return SAETrainer(
        model=model,
        sae=training_crosscoder_sae,
        activation_store=activation_store,
        save_checkpoint_fn=lambda *args, **kwargs: None,  # noqa: ARG005
        cfg=cfg,
    )

def test_sae_forward(training_crosscoder_sae: TrainingCrosscoderSAE):
    batch_size = 32
    d_in = training_crosscoder_sae.cfg.d_in
    n_layers = len(training_crosscoder_sae.cfg.hook_layers)
    d_sae = training_crosscoder_sae.cfg.d_sae

    x = torch.randn(batch_size, n_layers, d_in)
    train_step_output = training_crosscoder_sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=training_crosscoder_sae.cfg.l1_coefficient,
    )

    assert train_step_output.sae_out.shape == (batch_size, n_layers, d_in)
    assert train_step_output.feature_acts.shape == (batch_size, d_sae)
    assert (
        pytest.approx(train_step_output.loss.detach(), rel=1e-3)
        == (
            train_step_output.losses["mse_loss"]
            + train_step_output.losses["l1_loss"]
            + train_step_output.losses.get("ghost_grad_loss", 0.0)
        )
        .detach()  # type: ignore
        .cpu()
        .numpy()
    )

    expected_mse_loss = (
        (torch.pow((train_step_output.sae_out - x.float()), 2))
        .sum(dim=-1)
        .mean()
        .detach()
        .float()
    )

    assert (
        pytest.approx(train_step_output.losses["mse_loss"].item()) == expected_mse_loss  # type: ignore
    )

    expected_l1_loss = (
        (train_step_output.feature_acts
         * training_crosscoder_sae.W_dec.norm(dim=2).sum(dim=1))
        .norm(dim=1, p=1)
        .mean()
    )
    assert (
        pytest.approx(train_step_output.losses["l1_loss"].item(), rel=1e-3)  # type: ignore
        == training_crosscoder_sae.cfg.l1_coefficient * expected_l1_loss.detach().float()
    )


def test_sae_forward_with_mse_loss_norm(
    training_crosscoder_sae: TrainingCrosscoderSAE,
):
    # change the confgi and ensure the mse loss is calculated correctly
    training_crosscoder_sae.cfg.mse_loss_normalization = "dense_batch"
    training_crosscoder_sae.mse_loss_fn = training_crosscoder_sae._get_mse_loss_fn()

    batch_size = 32
    d_in = training_crosscoder_sae.cfg.d_in
    n_layers = len(training_crosscoder_sae.cfg.hook_layers)
    d_sae = training_crosscoder_sae.cfg.d_sae

    x = torch.randn(batch_size, n_layers, d_in)
    train_step_output = training_crosscoder_sae.training_forward_pass(
        sae_in=x,
        current_l1_coefficient=training_crosscoder_sae.cfg.l1_coefficient,
    )

    assert train_step_output.sae_out.shape == (batch_size, n_layers, d_in)
    assert train_step_output.feature_acts.shape == (batch_size, d_sae)
    assert "ghost_grad_loss" not in train_step_output.losses

    x_centred = x - x.mean(dim=0, keepdim=True)
    expected_mse_loss = (
        (
            torch.nn.functional.mse_loss(train_step_output.sae_out, x, reduction="none")
            / (1e-6 + x_centred.norm(dim=-1, keepdim=True))
        )
        .sum(dim=-1)
        .mean()
        .detach()
        .item()
    )

    assert (
        pytest.approx(train_step_output.losses["mse_loss"].item()) == expected_mse_loss  # type: ignore
    )

    assert (
        pytest.approx(train_step_output.loss.detach(), rel=1e-3)
        == (
            train_step_output.losses["mse_loss"]
            + train_step_output.losses["l1_loss"]
            + train_step_output.losses.get("ghost_grad_loss", 0.0)
        )
        .detach()  # type: ignore
        .numpy()
    )

    expected_l1_loss = (
        (train_step_output.feature_acts *
         training_crosscoder_sae.W_dec.norm(dim=2).sum(dim=1))
        .norm(dim=1, p=1)
        .mean()
    )
    assert (
        pytest.approx(train_step_output.losses["l1_loss"].item(), rel=1e-3)  # type: ignore
        == training_crosscoder_sae.cfg.l1_coefficient * expected_l1_loss.detach().float()
    )

