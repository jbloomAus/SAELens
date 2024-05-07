from functools import partial
from typing import Any, Mapping, cast

import pandas as pd
import torch
import wandb
from transformer_lens.hook_points import HookedRootModule

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sparse_autoencoder import SparseAutoencoder


@torch.no_grad()
def run_evals(
    sparse_autoencoder: SparseAutoencoder,
    activation_store: ActivationsStore,
    model: HookedRootModule,
    n_training_steps: int,
    suffix: str = "",
) -> Mapping[str, Any]:
    hook_point = sparse_autoencoder.cfg.hook_point
    hook_point_layer = sparse_autoencoder.hook_point_layer
    hook_point_head_index = sparse_autoencoder.cfg.hook_point_head_index
    hook_point_eval = sparse_autoencoder.cfg.hook_point_eval.format(
        layer=hook_point_layer
    )
    ### Evals
    eval_tokens = activation_store.get_batch_tokens()

    # Get Reconstruction Score
    losses_df = recons_loss_batched(
        sparse_autoencoder,
        model,
        activation_store,
        n_batches=10,
    )

    recons_score = losses_df["score"].mean()
    ntp_loss = losses_df["loss"].mean()
    recons_loss = losses_df["recons_loss"].mean()
    zero_abl_loss = losses_df["zero_abl_loss"].mean()
    d_kl = losses_df["d_kl"].mean()

    # get cache
    _, cache = model.run_with_cache(
        eval_tokens,
        prepend_bos=False,
        names_filter=[hook_point_eval, hook_point],
        **sparse_autoencoder.cfg.model_kwargs,
    )

    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if hook_point_head_index is not None:
        original_act = cache[hook_point][:, :, hook_point_head_index]
    elif any(substring in hook_point for substring in has_head_dim_key_substrings):
        original_act = cache[hook_point].flatten(-2, -1)
    else:
        original_act = cache[hook_point]

    sae_out, _, _, _, _, _ = sparse_autoencoder(original_act)
    del cache

    if "cuda" in str(model.cfg.device):
        torch.cuda.empty_cache()

    l2_norm_in = torch.norm(original_act, dim=-1)
    l2_norm_out = torch.norm(sae_out, dim=-1)
    l2_norm_in_for_div = l2_norm_in.clone()
    l2_norm_in_for_div[torch.abs(l2_norm_in_for_div) < 0.0001] = 1
    l2_norm_ratio = l2_norm_out / l2_norm_in_for_div

    metrics = {
        # l2 norms
        f"metrics/l2_norm{suffix}": l2_norm_out.mean().item(),
        f"metrics/l2_ratio{suffix}": l2_norm_ratio.mean().item(),
        # CE Loss
        f"metrics/CE_loss_score{suffix}": recons_score,
        f"metrics/ce_loss_without_sae{suffix}": ntp_loss,
        f"metrics/ce_loss_with_sae{suffix}": recons_loss,
        f"metrics/ce_loss_with_ablation{suffix}": zero_abl_loss,
        # KL divergence against intact model
        f"metrics/kl_div{suffix}": d_kl,
    }

    if wandb.run is not None:
        wandb.log(
            metrics,
            step=n_training_steps,
        )

    return metrics


def recons_loss_batched(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    n_batches: int = 100,
):
    losses = []
    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens()
        score, loss, recons_loss, zero_abl_loss, d_kl = get_recons_loss(
            sparse_autoencoder, model, batch_tokens
        )
        losses.append(
            (
                score.mean().item(),
                loss.mean().item(),
                recons_loss.mean().item(),
                zero_abl_loss.mean().item(),
                d_kl.mean().item(),
            )
        )

    losses = pd.DataFrame(
        losses,
        columns=cast(
            Any, ["score", "loss", "recons_loss", "zero_abl_loss", "d_kl"]
            )
    )

    return losses


@torch.no_grad()
def get_recons_loss(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedRootModule,
    batch_tokens: torch.Tensor,
):
    hook_point = sparse_autoencoder.cfg.hook_point
    model_outs = model(
        batch_tokens, return_type="both", **sparse_autoencoder.cfg.model_kwargs
    )
    head_index = sparse_autoencoder.cfg.hook_point_head_index
    loss = model_outs.loss

    def standard_replacement_hook(activations: torch.Tensor, hook: Any):
        activations = sparse_autoencoder.forward(activations).sae_out.to(
            activations.dtype
        )
        return activations

    def all_head_replacement_hook(activations: torch.Tensor, hook: Any):
        new_activations = sparse_autoencoder.forward(
            activations.flatten(-2, -1)
        ).sae_out.to(activations.dtype)
        new_activations = new_activations.reshape(
            activations.shape
        )  # reshape to match original shape
        return new_activations

    def single_head_replacement_hook(activations: torch.Tensor, hook: Any):
        new_activations = sparse_autoencoder.forward(
            activations[:, :, head_index]
        ).sae_out.to(activations.dtype)
        activations[:, :, head_index] = new_activations
        return activations

    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if any(substring in hook_point for substring in has_head_dim_key_substrings):
        if head_index is None:
            replacement_hook = all_head_replacement_hook
        else:
            replacement_hook = single_head_replacement_hook
    else:
        replacement_hook = standard_replacement_hook

    recons_outs = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_point, partial(replacement_hook))],
        **sparse_autoencoder.cfg.model_kwargs,
    )
    recons_loss = recons_outs.loss

    zero_abl_loss = model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(hook_point, zero_ablate_hook)],
        **sparse_autoencoder.cfg.model_kwargs,
    )

    div_val = zero_abl_loss - loss
    div_val[torch.abs(div_val) < 0.0001] = 1.0

    score = (zero_abl_loss - recons_loss) / div_val

    # KL divergence
    model_logits = model_outs.logits  # [batch, pos, d_vocab]
    model_logprobs = torch.nn.functional.log_softmax(
        model_logits, dim=-1)
    recons_logits = recons_outs.logits
    recons_logprobs = torch.nn.functional.log_softmax(
        recons_logits, dim=-1)
    # Note: PyTorch KL is backwards compared to the mathematical definition
    # target distribution comes second, see
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html
    d_kl = torch.nn.functional.kl_div(
        recons_logprobs,
        model_logprobs,
        reduction='batchmean',
        log_target=True,  # for numerics
    )

    return score, loss, recons_loss, zero_abl_loss, d_kl


def zero_ablate_hook(activations: torch.Tensor, hook: Any):
    activations = torch.zeros_like(activations)
    return activations


def kl_divergence_attention(y_true: torch.Tensor, y_pred: torch.Tensor):
    # Compute log probabilities for KL divergence
    log_y_true = torch.log2(y_true + 1e-10)
    log_y_pred = torch.log2(y_pred + 1e-10)

    return y_true * (log_y_true - log_y_pred)
