from functools import partial
from typing import Any, cast

import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

import wandb
from sae_training.activations_store import ActivationsStore
from sae_training.sparse_autoencoder import SparseAutoencoder


@torch.no_grad()
def run_evals(
    sparse_autoencoder: SparseAutoencoder,
    activation_store: ActivationsStore,
    model: HookedTransformer,
    n_training_steps: int,
    suffix: str = "",
):
    hook_point = sparse_autoencoder.cfg.hook_point
    hook_point_layer = sparse_autoencoder.cfg.hook_point_layer
    hook_point_head_index = sparse_autoencoder.cfg.hook_point_head_index

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

    # get cache
    _, cache = model.run_with_cache(
        eval_tokens,
        prepend_bos=False,
        names_filter=[get_act_name("pattern", hook_point_layer), hook_point],
    )

    # get act
    if hook_point_head_index is not None:
        original_act = cache[hook_point][:, :, hook_point_head_index]
    elif "attn" in hook_point:
        original_act = cache[hook_point].flatten(-2, -1)
    else:
        original_act = cache[hook_point]

    sae_out, _, _, _, _, _ = sparse_autoencoder(original_act)
    # patterns_original = (
    #     cache[get_act_name("pattern", hook_point_layer)][:, hook_point_head_index]
    #     .detach()
    #     .cpu()
    # )
    del cache

    if "cuda" in str(model.cfg.device):
        torch.cuda.empty_cache()

    l2_norm_in = torch.norm(original_act, dim=-1)
    l2_norm_out = torch.norm(sae_out, dim=-1)
    l2_norm_ratio = l2_norm_out / l2_norm_in

    metrics = {
            # l2 norms
            f"metrics/l2_norm{suffix}": l2_norm_out.mean().item(),
            f"metrics/l2_ratio{suffix}": l2_norm_ratio.mean().item(),
            # CE Loss
            f"metrics/CE_loss_score{suffix}": recons_score,
            f"metrics/ce_loss_without_sae{suffix}": ntp_loss,
            f"metrics/ce_loss_with_sae{suffix}": recons_loss,
            f"metrics/ce_loss_with_ablation{suffix}": zero_abl_loss,
        }
    
    if wandb.run is not None:
        wandb.log(
            metrics,
            step=n_training_steps
        )

    return metrics

    # head_index = sparse_autoencoder.cfg.hook_point_head_index

    # def standard_replacement_hook(activations: torch.Tensor, hook: Any):
    #     activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
    #     return activations

    # def all_head_replacement_hook(activations: torch.Tensor, hook: Any):
    #     new_activations = sparse_autoencoder.forward(activations)[0].to(
    #         activations.dtype
    #     )
    #     activations = new_activations.reshape(
    #         activations.shape
    #     )  # reshape to match original shape
    #     return activations

    # def single_head_replacement_hook(activations: torch.Tensor, hook: Any):
    #     new_activations = sparse_autoencoder.forward(activations[:, :, head_index])[
    #         0
    #     ].to(activations.dtype)
    #     activations[:, :, head_index] = new_activations
    #     return activations

    # if "attn" in hook_point:
    #     if hook_point_head_index is None:
    #         replacement_hook = all_head_replacement_hook
    #     else:
    #         replacement_hook = single_head_replacement_hook
    # else:
    #     replacement_hook = standard_replacement_hook

    # get attn when using reconstructed activations
    # with model.hooks(fwd_hooks=[(hook_point, partial(replacement_hook))]):
    #     _, new_cache = model.run_with_cache(
    #         eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)]
    #     )
    #     patterns_reconstructed = (
    #         new_cache[get_act_name("pattern", hook_point_layer)][
    #             :, hook_point_head_index
    #         ]
    #         .detach()
    #         .cpu()
    #     )
    #     del new_cache

    # # get attn when using reconstructed activations
    # with model.hooks(fwd_hooks=[(hook_point, partial(zero_ablate_hook))]):
    #     _, zero_ablation_cache = model.run_with_cache(
    #         eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)]
    #     )
    #     patterns_ablation = (
    #         zero_ablation_cache[get_act_name("pattern", hook_point_layer)][
    #             :, hook_point_head_index
    #         ]
    #         .detach()
    #         .cpu()
    #     )
    #     del zero_ablation_cache

    # if sparse_autoencoder.cfg.hook_point_head_index:
    #     kl_result_reconstructed = kl_divergence_attention(
    #         patterns_original, patterns_reconstructed
    #     )
    #     kl_result_reconstructed = kl_result_reconstructed.sum(dim=-1).numpy()

    #     kl_result_ablation = kl_divergence_attention(
    #         patterns_original, patterns_ablation
    #     )
    #     kl_result_ablation = kl_result_ablation.sum(dim=-1).numpy()

    #     if wandb.run is not None:
    #         wandb.log(
    #             {
    #                 f"metrics/kldiv_reconstructed{suffix}": kl_result_reconstructed.mean().item(),
    #                 f"metrics/kldiv_ablation{suffix}": kl_result_ablation.mean().item(),
    #             },
    #             step=n_training_steps,
    #         )


def recons_loss_batched(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedTransformer,
    activation_store: ActivationsStore,
    n_batches: int = 100,
):
    losses = []
    for _ in tqdm(range(n_batches)):
        batch_tokens = activation_store.get_batch_tokens()
        score, loss, recons_loss, zero_abl_loss = get_recons_loss(
            sparse_autoencoder, model, batch_tokens
        )
        losses.append(
            (
                score.mean().item(),
                loss.mean().item(),
                recons_loss.mean().item(),
                zero_abl_loss.mean().item(),
            )
        )

    losses = pd.DataFrame(
        losses, columns=cast(Any, ["score", "loss", "recons_loss", "zero_abl_loss"])
    )

    return losses


@torch.no_grad()
def get_recons_loss(
    sparse_autoencoder: SparseAutoencoder,
    model: HookedTransformer,
    batch_tokens: torch.Tensor,
):
    hook_point = sparse_autoencoder.cfg.hook_point
    loss = model(batch_tokens, return_type="loss")
    hook_point_head_index = sparse_autoencoder.cfg.hook_point_head_index

    def standard_replacement_hook(activations: torch.Tensor, hook: Any):
        activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
        return activations

    def all_head_replacement_hook(activations: torch.Tensor, hook: Any):
        new_activations = sparse_autoencoder.forward(activations.flatten(-2, -1))[0].to(
            activations.dtype
        )
        new_activations = new_activations.reshape(
            activations.shape
        )  # reshape to match original shape
        return new_activations

    def single_head_replacement_hook(activations: torch.Tensor, hook: Any):
        new_activations = sparse_autoencoder.forward(
            activations[:, :, hook_point_head_index]
        )[0].to(activations.dtype)
        activations[:, :, hook_point_head_index] = new_activations
        return activations

    if "attn" in hook_point:
        if hook_point_head_index is None:
            replacement_hook = all_head_replacement_hook
        else:
            replacement_hook = single_head_replacement_hook
    else:
        replacement_hook = standard_replacement_hook

    recons_loss = model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(hook_point, partial(replacement_hook))],
    )

    zero_abl_loss = model.run_with_hooks(
        batch_tokens, return_type="loss", fwd_hooks=[(hook_point, zero_ablate_hook)]
    )

    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    return score, loss, recons_loss, zero_abl_loss


def zero_ablate_hook(activations: torch.Tensor, hook: Any):
    activations = torch.zeros_like(activations)
    return activations


def kl_divergence_attention(y_true: torch.Tensor, y_pred: torch.Tensor):
    # Compute log probabilities for KL divergence
    log_y_true = torch.log2(y_true + 1e-10)
    log_y_pred = torch.log2(y_pred + 1e-10)

    return y_true * (log_y_true - log_y_pred)
