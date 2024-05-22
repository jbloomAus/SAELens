from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens import __version__
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.sparse_autoencoder import ForwardOutput, SparseAutoencoder
from sae_lens.training.train_sae_on_language_model import (
    SAETrainer,
    TrainStepOutput,
    _log_feature_sparsity,
    _update_sae_lens_training_version,
)
from tests.unit.helpers import TINYSTORIES_MODEL, build_sae_cfg, load_model_cached


@pytest.fixture
def cfg():
    cfg = build_sae_cfg(d_in=64, d_sae=128, hook_point_layer=0)
    return cfg


@pytest.fixture
def model():
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def activation_store(model: HookedTransformer, cfg: LanguageModelSAERunnerConfig):
    return ActivationsStore.from_config(
        model, cfg, dataset=Dataset.from_list([{"text": "hello world"}] * 2000)
    )


@pytest.fixture
def sae(cfg: LanguageModelSAERunnerConfig):
    return SparseAutoencoder(cfg)


@pytest.fixture
def trainer(
    cfg: LanguageModelSAERunnerConfig,
    sae: SparseAutoencoder,
    model: HookedTransformer,
    activation_store: ActivationsStore,
):

    trainer = SAETrainer(
        model=model,
        sae=sae,
        activation_store=activation_store,
        save_checkpoint_fn=lambda *args, **kwargs: None,
        cfg=cfg,
    )

    return trainer


def modify_sae_output(
    sae: SparseAutoencoder, modifier: Callable[[ForwardOutput], ForwardOutput]
):
    """
    Helper to modify the output of the SAE forward pass for use in patching, for use in patch side_effect.
    We need real grads during training, so we can't just mock the whole forward pass directly.
    """

    def modified_forward(*args: Any, **kwargs: Any):
        output = SparseAutoencoder.forward(sae, *args, **kwargs)
        return modifier(output)

    return modified_forward


def test_train_step__reduces_loss_when_called_repeatedly_on_same_acts(
    trainer: SAETrainer,
) -> None:

    layer_acts = trainer.activation_store.next_batch()

    # intentionally train on the same activations 5 times to ensure loss decreases
    train_outputs = [
        trainer._train_step(
            sparse_autoencoder=trainer.sae,
            sae_in=layer_acts[:, 0, :],
        )
        for _ in range(5)
    ]

    # ensure loss decreases with each training step
    for output, next_output in zip(train_outputs[:-1], train_outputs[1:]):
        assert output.loss > next_output.loss
    assert (
        trainer.n_frac_active_tokens == 20
    )  # should increment each step by batch_size (5*4)


def test_train_step__output_looks_reasonable(trainer: SAETrainer) -> None:

    layer_acts = trainer.activation_store.next_batch()

    output = trainer._train_step(
        sparse_autoencoder=trainer.sae,
        sae_in=layer_acts[:, 0, :],
    )

    assert output.loss > 0
    # only hook_point_layer=0 acts should be passed to the SAE
    assert torch.allclose(output.sae_in, layer_acts[:, 0, :])
    assert output.sae_out.shape == output.sae_in.shape
    assert output.feature_acts.shape == (4, 128)  # batch_size, d_sae
    assert output.ghost_grad_neuron_mask.shape == (128,)
    assert output.loss.shape == ()
    assert output.mse_loss.shape == ()
    assert output.ghost_grad_loss == 0
    # ghots grads shouldn't trigger until dead_feature_window, which hasn't been reached yet
    assert torch.all(output.ghost_grad_neuron_mask == False)  # noqa
    assert output.ghost_grad_loss == 0
    assert trainer.n_frac_active_tokens == 4
    assert trainer.act_freq_scores.sum() > 0  # at least SOME acts should have fired
    assert torch.allclose(
        trainer.act_freq_scores, (output.feature_acts.abs() > 0).float().sum(0)
    )


def test_train_step__ghost_grads_mask(trainer: SAETrainer) -> None:

    layer_acts = trainer.activation_store.next_batch()

    trainer.n_forward_passes_since_fired = (
        torch.ones(trainer.cfg.d_sae).float() * 3 * trainer.cfg.dead_feature_window  # type: ignore
    )

    output = trainer._train_step(
        sparse_autoencoder=trainer.sae,
        sae_in=layer_acts[:, 0, :],
    )

    assert torch.all(
        output.ghost_grad_neuron_mask == torch.ones_like(output.ghost_grad_neuron_mask)
    )


def test_train_step__sparsity_updates_based_on_feature_act_sparsity(
    trainer: SAETrainer,
) -> None:

    layer_acts = trainer.activation_store.next_batch()
    trainer.n_forward_passes_since_fired = (
        torch.ones(trainer.cfg.d_sae).float() * 3 * trainer.cfg.dead_feature_window  # type: ignore
    )
    feature_acts = torch.zeros((4, 128))
    feature_acts[:, :12] = 1

    with patch.object(
        trainer.sae,
        "forward",
        side_effect=modify_sae_output(
            trainer.sae, lambda out: out._replace(feature_acts=feature_acts)
        ),
    ):
        train_output = trainer._train_step(
            sparse_autoencoder=trainer.sae,
            sae_in=layer_acts[:, 0, :],
        )

    # should increase by batch_size
    assert trainer.n_frac_active_tokens == 4
    # add freq scores for all non-zero feature acts
    assert torch.allclose(trainer.act_freq_scores[:12], 4 * torch.ones(12).float())
    assert torch.allclose(trainer.n_forward_passes_since_fired[:12], torch.zeros(12))

    # the outputs from the SAE should be included in the train output
    assert train_output.feature_acts is feature_acts


def test_log_feature_sparsity__handles_zeroes_by_default_fp32() -> None:
    fp32_zeroes = torch.tensor([0], dtype=torch.float32)
    assert _log_feature_sparsity(fp32_zeroes).item() != float("-inf")


# TODO: currently doesn't work for fp16, we should address this
@pytest.mark.skip(reason="Currently doesn't work for fp16")
def test_log_feature_sparsity__handles_zeroes_by_default_fp16() -> None:
    fp16_zeroes = torch.tensor([0], dtype=torch.float16)
    assert _log_feature_sparsity(fp16_zeroes).item() != float("-inf")


def test_build_train_step_log_dict(trainer: SAETrainer) -> None:

    train_output = TrainStepOutput(
        sae_in=torch.tensor([[-1, 0], [0, 2], [1, 1]]).float(),
        sae_out=torch.tensor([[0, 0], [0, 2], [0.5, 1]]).float(),
        feature_acts=torch.tensor([[0, 0, 0, 1], [1, 0, 0, 1], [1, 0, 1, 1]]).float(),
        loss=torch.tensor(0.5),
        mse_loss=torch.tensor(0.25),
        l1_loss=torch.tensor(0.1),
        ghost_grad_loss=torch.tensor(0.15),
        ghost_grad_neuron_mask=torch.tensor([False, True, False, True]),
    )

    # we're relying on the trainer only for some of the metrics here
    # we should more / less try to break this and push
    # everything through the train step output if we can.
    log_dict = trainer._build_train_step_log_dict(
        output=train_output, n_training_tokens=123
    )
    assert log_dict == {
        "losses/mse_loss": 0.25,
        # l1 loss is scaled by l1_coefficient
        "losses/l1_loss": train_output.l1_loss.item() / trainer.sae.l1_coefficient,
        "losses/ghost_grad_loss": pytest.approx(0.15),
        "losses/overall_loss": 0.5,
        "metrics/explained_variance": 0.75,
        "metrics/explained_variance_std": 0.25,
        "metrics/l0": 2.0,
        "sparsity/mean_passes_since_fired": trainer.n_forward_passes_since_fired.mean().item(),
        "sparsity/dead_features": 2,
        "details/current_learning_rate": 2e-4,
        "details/current_l1_coefficient": trainer.cfg.l1_coefficient,
        "details/n_training_tokens": 123,
    }


def test_train_sae_group_on_language_model__runs(
    ts_model: HookedTransformer,
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    cfg = build_sae_cfg(
        checkpoint_path=str(checkpoint_dir),
        train_batch_size=32,
        training_tokens=100,
        context_size=8,
    )
    # just a tiny datast which will run quickly
    dataset = Dataset.from_list([{"text": "hello world"}] * 2000)
    activation_store = ActivationsStore.from_config(ts_model, cfg, dataset=dataset)
    sae = SparseAutoencoder(cfg)
    sae = SAETrainer(
        model=ts_model,
        sae=sae,
        activation_store=activation_store,
        save_checkpoint_fn=lambda *args, **kwargs: None,
        cfg=cfg,
    ).fit()

    assert isinstance(sae, SparseAutoencoder)


def test_update_sae_lens_training_version_sets_the_current_version():
    cfg = build_sae_cfg(sae_lens_training_version="0.1.0")
    sae = SparseAutoencoder(cfg)
    _update_sae_lens_training_version(sae)
    assert sae.cfg.sae_lens_training_version == __version__
