import os
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest
import torch
import wandb
from datasets import Dataset
from torch import Tensor
from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.optim import get_scheduler
from sae_lens.training.sae_group import SparseAutoencoderDictionary
from sae_lens.training.sparse_autoencoder import ForwardOutput, SparseAutoencoder
from sae_lens.training.train_sae_on_language_model import (
    SAETrainContext,
    TrainStepOutput,
    _build_train_step_log_dict,
    _log_feature_sparsity,
    _save_checkpoint,
    _train_step,
    train_sae_group_on_language_model,
)
from tests.unit.helpers import build_sae_cfg


# TODO: Address why we have this code here rather than importing it.
def build_train_ctx(
    sae: SparseAutoencoder,
    act_freq_scores: Tensor | None = None,
    n_forward_passes_since_fired: Tensor | None = None,
    n_frac_active_tokens: int = 0,
) -> SAETrainContext:
    """
    Factory helper to build a default SAETrainContext object.
    """
    assert sae.cfg.d_sae is not None
    assert not isinstance(sae.cfg.lr, list)
    optimizer = torch.optim.Adam(sae.parameters(), lr=sae.cfg.lr)
    return SAETrainContext(
        act_freq_scores=(
            torch.zeros(sae.cfg.d_sae) if act_freq_scores is None else act_freq_scores
        ),
        n_forward_passes_since_fired=(
            torch.zeros(sae.cfg.d_sae)
            if n_forward_passes_since_fired is None
            else n_forward_passes_since_fired
        ),
        n_frac_active_tokens=n_frac_active_tokens,
        optimizer=optimizer,
        scheduler=get_scheduler(
            "constant",
            lr=sae.cfg.lr,
            optimizer=optimizer,
            training_steps=1000,
            lr_end=0,
            warm_up_steps=0,
            decay_steps=0,
            num_cycles=1,
        ),
    )


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


def test_train_step__reduces_loss_when_called_repeatedly_on_same_acts() -> None:
    cfg = build_sae_cfg(d_in=64, d_sae=128, hook_point_layer=0)
    sae = SparseAutoencoder(cfg)
    ctx = build_train_ctx(sae)

    layer_acts = torch.randn(10, 1, 64)

    # intentionally train on the same activations 5 times to ensure loss decreases
    train_outputs = [
        _train_step(
            sparse_autoencoder=sae,
            ctx=ctx,
            layer_acts=layer_acts,
            all_layers=[0],
            feature_sampling_window=1000,
            use_wandb=False,
            n_training_steps=10,
            batch_size=10,
            wandb_suffix="",
        )
        for _ in range(5)
    ]

    # ensure loss decreases with each training step
    for output, next_output in zip(train_outputs[:-1], train_outputs[1:]):
        assert output.loss > next_output.loss
    assert ctx.n_frac_active_tokens == 50  # should increment each step by batch_size


def test_train_step__output_looks_reasonable() -> None:
    cfg = build_sae_cfg(d_in=64, d_sae=128, hook_point_layer=0)
    sae = SparseAutoencoder(cfg)
    ctx = build_train_ctx(sae)

    layer_acts = torch.randn(10, 2, 64)

    output = _train_step(
        sparse_autoencoder=sae,
        ctx=ctx,
        layer_acts=layer_acts,
        all_layers=[0],
        feature_sampling_window=1000,
        use_wandb=False,
        n_training_steps=10,
        batch_size=10,
        wandb_suffix="",
    )

    assert output.loss > 0
    # only hook_point_layer=0 acts should be passed to the SAE
    assert torch.allclose(output.sae_in, layer_acts[:, 0, :])
    assert output.sae_out.shape == output.sae_in.shape
    assert output.feature_acts.shape == (10, 128)  # batch_size, d_sae
    assert output.ghost_grad_neuron_mask.shape == (128,)
    assert output.loss.shape == ()
    assert output.mse_loss.shape == ()
    assert output.ghost_grad_loss.shape == ()
    # ghots grads shouldn't trigger until dead_feature_window, which hasn't been reached yet
    assert torch.all(output.ghost_grad_neuron_mask == False)  # noqa
    assert output.ghost_grad_loss == 0
    assert ctx.n_frac_active_tokens == 10
    assert ctx.act_freq_scores.sum() > 0  # at least SOME acts should have fired
    assert torch.allclose(
        ctx.act_freq_scores, (output.feature_acts.abs() > 0).float().sum(0)
    )


def test_train_step__ghost_grads_mask() -> None:
    cfg = build_sae_cfg(d_in=2, d_sae=4, dead_feature_window=5)
    sae = SparseAutoencoder(cfg)
    ctx = build_train_ctx(
        sae, n_forward_passes_since_fired=torch.tensor([0, 4, 7, 9]).float()
    )

    output = _train_step(
        sparse_autoencoder=sae,
        ctx=ctx,
        layer_acts=torch.randn(10, 1, 2),
        all_layers=[0],
        feature_sampling_window=1000,
        use_wandb=False,
        n_training_steps=10,
        batch_size=10,
        wandb_suffix="",
    )
    assert torch.all(
        output.ghost_grad_neuron_mask == torch.Tensor([False, False, True, True])
    )


def test_train_step__sparsity_updates_based_on_feature_act_sparsity() -> None:
    cfg = build_sae_cfg(d_in=2, d_sae=4, hook_point_layer=0)
    sae = SparseAutoencoder(cfg)

    feature_acts = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 1]]).float()
    layer_acts = torch.randn(3, 1, 2)

    ctx = build_train_ctx(
        sae,
        n_frac_active_tokens=9,
        act_freq_scores=torch.tensor([0, 3, 7, 1]).float(),
        n_forward_passes_since_fired=torch.tensor([8, 2, 0, 0]).float(),
    )
    with patch.object(
        sae,
        "forward",
        side_effect=modify_sae_output(
            sae, lambda out: out._replace(feature_acts=feature_acts)
        ),
    ):
        train_output = _train_step(
            sparse_autoencoder=sae,
            ctx=ctx,
            layer_acts=layer_acts,
            all_layers=[0],
            feature_sampling_window=1000,
            use_wandb=False,
            n_training_steps=10,
            batch_size=3,
            wandb_suffix="",
        )

    # should increase by batch_size
    assert ctx.n_frac_active_tokens == 12
    # add freq scores for all non-zero feature acts
    assert torch.allclose(
        ctx.act_freq_scores,
        torch.tensor([2, 3, 8, 3]).float(),
    )
    assert torch.allclose(
        ctx.n_forward_passes_since_fired,
        torch.tensor([0, 3, 0, 0]).float(),
    )

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


def test_build_train_step_log_dict() -> None:
    cfg = build_sae_cfg(
        d_in=2, d_sae=4, hook_point_layer=0, lr=2e-4, l1_coefficient=1e-2
    )
    sae = SparseAutoencoder(cfg)
    ctx = build_train_ctx(
        sae,
        act_freq_scores=torch.tensor([0, 3, 1, 0]).float(),
        n_frac_active_tokens=10,
        n_forward_passes_since_fired=torch.tensor([4, 0, 0, 0]).float(),
    )
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

    log_dict = _build_train_step_log_dict(
        sae, train_output, ctx, wandb_suffix="-wandbftw", n_training_tokens=123
    )
    assert log_dict == {
        "losses/mse_loss-wandbftw": 0.25,
        # l1 loss is scaled by l1_coefficient
        "losses/l1_loss-wandbftw": pytest.approx(10),
        "losses/ghost_grad_loss-wandbftw": pytest.approx(0.15),
        "losses/overall_loss-wandbftw": 0.5,
        "metrics/explained_variance-wandbftw": 0.75,
        "metrics/explained_variance_std-wandbftw": 0.25,
        "metrics/l0-wandbftw": 2.0,
        "sparsity/mean_passes_since_fired-wandbftw": 1.0,
        "sparsity/dead_features-wandbftw": 2,
        "details/current_learning_rate-wandbftw": 2e-4,
        "details/n_training_tokens": 123,
    }


def test_save_checkpoint(tmp_path: Path) -> None:

    # set wandb mode to offline
    os.environ["WANDB_MODE"] = "offline"

    wandb.init()
    checkpoint_dir = tmp_path / "checkpoint"
    cfg = build_sae_cfg(
        checkpoint_path=checkpoint_dir, d_in=25, d_sae=100, log_to_wandb=True
    )
    sae_group = SparseAutoencoderDictionary(cfg)
    assert len(sae_group.autoencoders) == 1
    ctx = build_train_ctx(
        next(iter(sae_group))[1],
        act_freq_scores=torch.randint(0, 100, (100,)),
        n_forward_passes_since_fired=torch.randint(0, 100, (100,)),
        n_frac_active_tokens=123,
    )
    name = next(iter(sae_group.autoencoders.keys()))

    res = _save_checkpoint(sae_group, {name: ctx}, "test_checkpoint")
    assert res == str(checkpoint_dir / "test_checkpoint")

    subfolder = os.listdir(res)[0]
    assert subfolder == name

    # list contents of subfolder
    contents = os.listdir(f"{res}/{subfolder}")
    assert len(contents) == 3
    assert "cfg.json" in contents
    assert "sparsity.safetensors" in contents
    assert "sae_weights.safetensors" in contents


def test_train_sae_group_on_language_model__runs(
    ts_model: HookedTransformer,
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    cfg = build_sae_cfg(
        checkpoint_path=checkpoint_dir,
        train_batch_size=32,
        training_tokens=100,
        context_size=8,
    )
    # just a tiny datast which will run quickly
    dataset = Dataset.from_list([{"text": "hello world"}] * 2000)
    activation_store = ActivationsStore.from_config(ts_model, cfg, dataset=dataset)
    sae_group = SparseAutoencoderDictionary(cfg)
    res = train_sae_group_on_language_model(
        model=ts_model,
        sae_group=sae_group,
        activation_store=activation_store,
        batch_size=32,
    )
    assert res.checkpoint_paths == [str(checkpoint_dir / "final")]
    assert len(res.log_feature_sparsities) == 1

    name = next(iter(res.sae_group))[0]
    assert res.log_feature_sparsities[name].shape == (cfg.d_sae,)
    assert res.sae_group is sae_group
