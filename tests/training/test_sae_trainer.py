from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformer_lens import HookedTransformer

from sae_lens import __version__
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sae_trainer import (
    SAETrainer,
    TrainStepOutput,
    _log_feature_sparsity,
    _update_sae_lens_training_version,
)
from sae_lens.training.training_sae import TrainingSAE
from tests.helpers import TINYSTORIES_MODEL, build_sae_cfg, load_model_cached


@pytest.fixture
def cfg():
    return build_sae_cfg(d_in=64, d_sae=128, hook_layer=0)


@pytest.fixture
def model():
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def activation_store(model: HookedTransformer, cfg: LanguageModelSAERunnerConfig):
    return ActivationsStore.from_config(
        model, cfg, override_dataset=Dataset.from_list([{"text": "hello world"}] * 2000)
    )


@pytest.fixture
def training_sae(cfg: LanguageModelSAERunnerConfig):
    return TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())


@pytest.fixture
def trainer(
    cfg: LanguageModelSAERunnerConfig,
    training_sae: TrainingSAE,
    model: HookedTransformer,
    activation_store: ActivationsStore,
):
    return SAETrainer(
        model=model,
        sae=training_sae,
        activation_store=activation_store,
        save_checkpoint_fn=lambda *args, **kwargs: None,  # noqa: ARG005
        cfg=cfg,
    )


def modify_sae_output(sae: TrainingSAE, modifier: Callable[[torch.Tensor], Any]):
    """
    Helper to modify the output of the SAE forward pass for use in patching, for use in patch side_effect.
    We need real grads during training, so we can't just mock the whole forward pass directly.
    """

    def modified_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        output = TrainingSAE.forward(sae, *args, **kwargs)
        return modifier(output)

    return modified_forward


def test_train_step__reduces_loss_when_called_repeatedly_on_same_acts(
    trainer: SAETrainer,
) -> None:
    layer_acts = trainer.activations_store.next_batch()

    # intentionally train on the same activations 5 times to ensure loss decreases
    train_outputs = [
        trainer._train_step(
            sae=trainer.sae,
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
    layer_acts = trainer.activations_store.next_batch()

    output = trainer._train_step(
        sae=trainer.sae,
        sae_in=layer_acts[:, 0, :],
    )

    assert output.loss > 0
    # only hook_point_layer=0 acts should be passed to the SAE
    assert torch.allclose(output.sae_in, layer_acts[:, 0, :])
    assert output.sae_out.shape == output.sae_in.shape
    assert output.feature_acts.shape == (4, 128)  # batch_size, d_sae
    # ghots grads shouldn't trigger until dead_feature_window, which hasn't been reached yet
    assert output.losses.get("ghost_grad_loss", 0) == 0
    assert trainer.n_frac_active_tokens == 4
    assert trainer.act_freq_scores.sum() > 0  # at least SOME acts should have fired
    assert torch.allclose(
        trainer.act_freq_scores, (output.feature_acts.abs() > 0).float().sum(0)
    )


def test_train_step__sparsity_updates_based_on_feature_act_sparsity(
    trainer: SAETrainer,
) -> None:
    trainer._reset_running_sparsity_stats()
    layer_acts = trainer.activations_store.next_batch()

    train_output = trainer._train_step(
        sae=trainer.sae,
        sae_in=layer_acts[:, 0, :],
    )
    feature_acts = train_output.feature_acts

    # should increase by batch_size
    assert trainer.n_frac_active_tokens == 4
    # add freq scores for all non-zero feature acts
    assert torch.allclose(
        trainer.act_freq_scores, (feature_acts > 0).float().sum(dim=0)
    )

    # check that features that just fired have n_forward_passes_since_fired = 0
    assert (
        trainer.n_forward_passes_since_fired[
            ((feature_acts > 0).float()[-1] == 1)
        ].max()
        == 0
    )
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
        hidden_pre=torch.tensor([[-1, 0, 0, 1], [1, -1, 0, 1], [1, -1, 1, 1]]).float(),
        loss=torch.tensor(0.5),
        losses={
            "mse_loss": 0.25,
            "l1_loss": 0.1,
            "ghost_grad_loss": 0.15,
        },
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
        "losses/l1_loss": train_output.losses["l1_loss"] / trainer.cfg.l1_coefficient,
        "losses/raw_l1_loss": train_output.losses["l1_loss"],
        "losses/overall_loss": 0.5,
        "losses/ghost_grad_loss": 0.15,
        "metrics/explained_variance": 0.6875,
        "metrics/explained_variance_legacy": 0.75,
        "metrics/explained_variance_legacy_std": 0.25,
        "metrics/l0": 2.0,
        "sparsity/mean_passes_since_fired": trainer.n_forward_passes_since_fired.mean().item(),
        "sparsity/dead_features": trainer.dead_neurons.sum().item(),
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
        training_tokens=20,
        context_size=8,
    )
    # just a tiny datast which will run quickly
    dataset = Dataset.from_list([{"text": "hello world"}] * 100)
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())
    sae = SAETrainer(
        model=ts_model,
        sae=sae,
        activation_store=activation_store,
        save_checkpoint_fn=lambda *args, **kwargs: None,  # noqa: ARG005
        cfg=cfg,
    ).fit()

    assert isinstance(sae, TrainingSAE)


def test_update_sae_lens_training_version_sets_the_current_version():
    cfg = build_sae_cfg(sae_lens_training_version="0.1.0")
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())
    _update_sae_lens_training_version(sae)
    assert sae.cfg.sae_lens_training_version == str(__version__)


def test_estimated_norm_scaling_factor_persistence(
    ts_model: HookedTransformer,
    tmp_path: Path,
):
    """Test that estimated_norm_scaling_factor is correctly persisted in intermediate checkpoints
    but not in the final checkpoint."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    cfg = build_sae_cfg(
        checkpoint_path=str(checkpoint_dir),
        training_tokens=100,  # Increased to ensure we hit checkpoints
        context_size=8,
        normalize_activations="expected_average_only_in",
        n_checkpoints=2,  # Explicitly request 2 checkpoints during training
    )

    # Create a small dataset
    dataset = Dataset.from_list([{"text": "hello world"}] * 100)
    activation_store = ActivationsStore.from_config(
        ts_model, cfg, override_dataset=dataset
    )
    sae = TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())

    trainer = SAETrainer(
        model=ts_model,
        sae=sae,
        activation_store=activation_store,
        save_checkpoint_fn=SAETrainingRunner.save_checkpoint,
        cfg=cfg,
    )

    # Train the model - this should create checkpoints
    trainer.fit()
    checkpoint_paths = list(
        checkpoint_dir.glob("**/activations_store_state.safetensors")
    )
    # We should have exactly 2 checkpoints:
    assert (
        len(checkpoint_paths) == 2
    ), f"Expected 2 checkpoints but got {len(checkpoint_paths)}"
    during_checkpoints = [
        load_file(path) for path in checkpoint_paths if "final" not in path.parent.name
    ]
    final_checkpoints = [
        load_file(path) for path in checkpoint_paths if "final" in path.parent.name
    ]
    assert (
        len(during_checkpoints) == 1
    ), f"Expected 1 other checkpoint but got {len(during_checkpoints)}"
    assert (
        len(final_checkpoints) == 1
    ), f"Expected 1 final checkpoint but got {len(final_checkpoints)}"
    during_checkpoint = during_checkpoints[0]
    final_checkpoint = final_checkpoints[0]

    # Check intermediate checkpoints have the scaling factor
    assert "estimated_norm_scaling_factor" in during_checkpoint
    assert during_checkpoint["estimated_norm_scaling_factor"] is not None

    # Final checkpoint should NOT have the scaling factor as it's been folded into the weights
    assert "estimated_norm_scaling_factor" not in final_checkpoint
