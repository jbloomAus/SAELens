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
from sae_lens.training.crosscoder_sae_trainer import CrosscoderSAETrainer
from sae_lens.training.sae_trainer import (
    TrainStepOutput,
    _log_feature_sparsity,
    _update_sae_lens_training_version,
)
from sae_lens.training.training_crosscoder_sae import TrainingCrosscoderSAE
from tests.helpers import TINYSTORIES_MODEL, build_sae_cfg, load_model_cached


@pytest.fixture
def cfg():
    return build_sae_cfg(
        d_in=64,
        d_sae=128,
        hook_name="blocks.{}.hook_mlp_out",
        hook_layers=[1,2,3],
        normalize_sae_decoder=False,
        scale_sparsity_penalty_by_decoder_norm=True,
    )


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
    return TrainingCrosscoderSAE.from_dict(cfg.get_training_sae_cfg_dict(),
                                           use_error_term=True)


@pytest.fixture
def trainer(
    cfg: LanguageModelSAERunnerConfig,
    training_sae: TrainingCrosscoderSAE,
    model: HookedTransformer,
    activation_store: ActivationsStore,
):
    return CrosscoderSAETrainer(
        model=model,
        sae=training_sae,
        activation_store=activation_store,
        save_checkpoint_fn=lambda *args, **kwargs: None,  # noqa: ARG005
        cfg=cfg,
    )


def modify_sae_output(sae: TrainingCrosscoderSAE, modifier: Callable[[torch.Tensor], Any]):
    """
    Helper to modify the output of the SAE forward pass for use in patching, for use in patch side_effect.
    We need real grads during training, so we can't just mock the whole forward pass directly.
    """

    def modified_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        output = TrainingCrosscoderSAE.forward(sae, *args, **kwargs)
        return modifier(output)

    return modified_forward


def test_train_step__reduces_loss_when_called_repeatedly_on_same_acts(
    trainer: CrosscoderSAETrainer,
) -> None:
    layer_acts = trainer.activations_store.next_batch()

    # intentionally train on the same activations 5 times to ensure loss decreases
    train_outputs = [
        trainer._train_step(
            sae=trainer.sae,
            sae_in=layer_acts,
        )
        for _ in range(5)
    ]

    # ensure loss decreases with each training step
    for output, next_output in zip(train_outputs[:-1], train_outputs[1:]):
        assert output.loss > next_output.loss
    assert (
        trainer.n_frac_active_tokens == 20
    )  # should increment each step by batch_size (5*4)


def test_train_step__output_looks_reasonable(trainer: CrosscoderSAETrainer) -> None:
    layer_acts = trainer.activations_store.next_batch()

    output = trainer._train_step(
        sae=trainer.sae,
        sae_in=layer_acts,
    )

    assert output.loss > 0
    # only hook_point_layer=0 acts should be passed to the SAE
    assert torch.allclose(output.sae_in, layer_acts)
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
    trainer: CrosscoderSAETrainer,
) -> None:
    trainer._reset_running_sparsity_stats()
    layer_acts = trainer.activations_store.next_batch()

    train_output = trainer._train_step(
        sae=trainer.sae,
        sae_in=layer_acts,
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
