import torch
from torch import Tensor

from sae_training.optim import get_scheduler
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.train_sae_on_language_model import SAETrainContext, _train_step
from tests.unit.helpers import build_sae_cfg


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
        scheduler=get_scheduler(None, optimizer=optimizer),
    )


def test_train_step_reduces_loss_when_called_repeatedly_on_same_acts() -> None:
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


def test_train_step_output_looks_reasonable() -> None:
    cfg = build_sae_cfg(d_in=64, d_sae=128, hook_point_layer=0, dead_feature_window=100)
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
    # ghots grads shouldn't trigger until dead_feature_window, which hasn't been reached yet
    assert torch.all(output.ghost_grad_neuron_mask == False)  # noqa
    assert output.ghost_grad_loss == 0
    assert ctx.n_frac_active_tokens == 10
    assert ctx.act_freq_scores.sum() > 0  # at least SOME acts should have fired
    assert torch.allclose(
        ctx.act_freq_scores, (output.feature_acts.abs() > 0).float().sum(0)
    )
