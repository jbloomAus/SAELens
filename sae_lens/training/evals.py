from functools import partial
from typing import Any, Mapping, cast

import pandas as pd
import torch
import torch.nn.functional as F
from transformer_lens.hook_points import HookedRootModule

import wandb
from transformer_lens.hook_points import HookedRootModule

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sparse_autoencoder import SparseAutoencoder

from dataclasses import dataclass, asdict


@dataclass
class Metrics:
    l0_norm: float
    l0_ratio: float
    l1_norm: float
    l1_ratio: float
    l2_norm: float
    l2_ratio: float
    CE_loss_score: float
    ce_loss_without_sae: float
    ce_loss_with_sae: float
    ce_loss_with_ablation: float
    kl_div: float
    l0_logits: float
    l1_logits: float
    sparsity: float
    percent_alive: float

    def as_dict(self, suffix: str) -> dict:
        output_dict = {}
        for key, value in asdict(self).items():
            output_dict[f"metrics/{key}{suffix}"] = value
        return output_dict


"""
Runs evals for a given SparseAutoencoder. You must also provide an ActivationsStore and a Hooked Model.

"""


@torch.no_grad()
def run_evals(
    sparse_autoencoder: SparseAutoencoder,
    activation_store: ActivationsStore,
    model: HookedRootModule,
    ctx: dict[str, str]
) -> Metrics:
    recons_score, ntp_loss, recons_loss, zero_abl_loss, kl_div = get_recons_loss(
        sparse_autoencoder,
        model,
        activation_store,
        n_batches=10,
    )

    original_act, sae_out = get_activations(model, activation_store, sparse_autoencoder)

    l2_norm_in = torch.norm(original_act, dim=-1)
    l2_norm_out = torch.norm(sae_out, dim=-1)
    l2_norm_in_for_div = l2_norm_in.clone()
    l2_norm_in_for_div[torch.abs(l2_norm_in_for_div) < 0.0001] = 1
    l2_norm_ratio = l2_norm_out / l2_norm_in_for_div

    l1_norm_in = torch.norm(original_act, p=1, dim=-1)
    l1_norm_out = torch.norm(sae_out, p=1, dim=-1)
    l1_norm_in_for_div = l1_norm_in.clone()
    l1_norm_in_for_div[torch.abs(l1_norm_in_for_div) < 0.0001] = 1
    l1_norm_ratio = l1_norm_out / l1_norm_in_for_div

    l0_norm_in = torch.sum(original_act != 0, dim=-1, dtype=original_act.dtype)
    l0_norm_out = torch.sum(sae_out != 0, dim=-1, dtype=sae_out.dtype)
    l0_norm_ratio = l0_norm_out / l0_norm_in

    l0_logits, l1_logits, sparsity, percent_alive = get_sparsity_metrics(sparse_autoencoder, activation_store)

    metrics = Metrics(
        l0_norm=l0_norm_out.mean().item(),
        l0_ratio=l0_norm_ratio.mean().item(),
        l1_norm=l1_norm_out.mean().item(),
        l1_ratio=l1_norm_ratio.mean().item(),
        l2_norm=l2_norm_out.mean().item(),
        l2_ratio=l2_norm_ratio.mean().item(),
        CE_loss_score=recons_score,
        ce_loss_without_sae=ntp_loss,
        ce_loss_with_sae=recons_loss,
        ce_loss_with_ablation=zero_abl_loss,
        kl_div=kl_div.item(),
        l0_logits=l0_logits,
        l1_logits=l1_logits,
        sparsity=sparsity.mean().item(),
        percent_alive=percent_alive,
    )

    if wandb.run is not None:
        wandb.log(
            metrics.as_dict(suffix=ctx['suffix']),
            step=ctx['n_training_steps'],
        )

    return metrics


def get_recons_loss(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    n_batches: int = 100,
):
    losses = []
    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens()
        score, loss, recons_loss, zero_abl_loss, kl_div = get_recons_loss_for_batch(
            sparse_autoencoder, model, batch_tokens
        )
        losses.append(
            (
                score.mean().item(),
                loss.mean().item(),
                recons_loss.mean().item(),
                zero_abl_loss.mean().item(),
                kl_div.item(),
            )
        )

    losses_df = pd.DataFrame(
        losses, columns=cast(Any, ["score", "loss", "recons_loss", "zero_abl_loss", "kl_div"])
    )

    recons_score = losses_df["score"].mean()
    ntp_loss = losses_df["loss"].mean()
    recons_loss = losses_df["recons_loss"].mean()
    zero_abl_loss = losses_df["zero_abl_loss"].mean()
    kl_div = losses_df["kl_div"].mean()

    return recons_score, ntp_loss, recons_loss, zero_abl_loss, kl_div


def get_replacement_hook(sparse_autoencoder: SparseAutoencoder):
    hook_point = sparse_autoencoder.cfg.hook_point
    head_index = sparse_autoencoder.cfg.hook_point_head_index

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

    head_dim_key_names = ["hook_q", "hook_k", "hook_v", "hook_z"]

    if any(name in hook_point for name in head_dim_key_names):
        if head_index is None:
            replacement_hook = all_head_replacement_hook
        else:
            replacement_hook = single_head_replacement_hook
    else:
        replacement_hook = standard_replacement_hook

    return replacement_hook


@torch.no_grad()
def get_recons_loss_for_batch(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedRootModule,
    batch_tokens: torch.Tensor,
):
    hook_point = sparse_autoencoder.cfg.hook_point
    loss = model(
        batch_tokens, return_type="loss", **sparse_autoencoder.cfg.model_kwargs
    )

    logits_without_sae = model(  # 4, 64, 50257
        batch_tokens,
        **sparse_autoencoder.cfg.model_kwargs,
    )

    replacement_hook = get_replacement_hook(sparse_autoencoder)

    reconstructed = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_point, partial(replacement_hook))],
        **sparse_autoencoder.cfg.model_kwargs,
    )

    zero_abl = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_point, zero_ablate_hook)],
        **sparse_autoencoder.cfg.model_kwargs,
    )

    div_val = zero_abl.loss - loss
    div_val[torch.abs(div_val) < 0.0001] = 1.0

    score = (zero_abl.loss - reconstructed.loss) / div_val

    kl_div = F.kl_div(
        F.log_softmax(reconstructed.logits, dim=-1),
        F.softmax(logits_without_sae, dim=-1),
        reduction='batchmean'
    )

    return score, loss, reconstructed.loss, zero_abl.loss, kl_div


def zero_ablate_hook(activations: torch.Tensor, hook: Any):
    activations = torch.zeros_like(activations)
    return activations


@torch.no_grad()
def get_sparsity_metrics(
    sparse_autoencoder: SparseAutoencoder,
    activation_store: ActivationsStore,
    n_batches: int = 50,
) -> tuple[float, float, torch.Tensor, float]:
    assert activation_store.d_in == sparse_autoencoder.cfg.d_in  # TODO this should be checked on initialization
    batch_size = sparse_autoencoder.cfg.train_batch_size

    assert isinstance(sparse_autoencoder.cfg.d_sae, int)
    total_feature_acts = torch.zeros(sparse_autoencoder.cfg.d_sae)
    l0s_list = []
    l1s_list = []
    for i in range(n_batches):
        batch_activations = activation_store.next_batch()
        feature_acts = sparse_autoencoder(batch_activations).feature_acts.squeeze()
        l0s = (feature_acts > 0).float().squeeze().sum(dim=1)
        l1s = feature_acts.abs().sum(dim=1)
        total_feature_acts += (feature_acts > 0).squeeze().sum(dim=0).cpu()
        l0s_list.append(l0s)
        l1s_list.append(l1s)

    l0 = torch.concat(l0s_list).mean().item()
    l1 = torch.concat(l1s_list).mean().item()

    sparsity = total_feature_acts / (n_batches * batch_size)
    log_feature_sparsity = torch.log10(sparsity + 1e-10)
    percent_alive = (log_feature_sparsity > -5).float().mean().item()

    return l0, l1, sparsity, percent_alive


def get_activations(model: HookedRootModule, activation_store: ActivationsStore, sparse_autoencoder: SparseAutoencoder):
    if "cuda" in str(model.cfg.device):
        torch.cuda.empty_cache()
    eval_tokens = activation_store.get_batch_tokens()
    hook_point = sparse_autoencoder.cfg.hook_point
    hook_point_layer = sparse_autoencoder.hook_point_layer
    hook_point_head_index = sparse_autoencoder.cfg.hook_point_head_index
    hook_point_eval = sparse_autoencoder.cfg.hook_point_eval.format(
        layer=hook_point_layer
    )
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

    return original_act, sae_out
