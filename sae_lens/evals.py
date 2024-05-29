from functools import partial
from typing import Any, Mapping, cast

import pandas as pd
import torch
from transformer_lens.hook_points import HookedRootModule

from sae_lens.sae import SAE
from sae_lens.training.activations_store import ActivationsStore


@torch.no_grad()
def run_evals(
    sae: SAE,
    activation_store: ActivationsStore,
    model: HookedRootModule,
    n_eval_batches: int = 10,
    eval_batch_size_prompts: int | None = None,
    model_kwargs: Mapping[str, Any] = {},
) -> Mapping[str, Any]:

    hook_name = sae.cfg.hook_name
    hook_head_index = sae.cfg.hook_head_index
    ### Evals
    eval_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)

    # TODO: Come up with a cleaner long term strategy here for SAEs that do reshaping.
    # turn off hook_z reshaping mode if it's on, and restore it after evals
    if "hook_z" in hook_name:
        previous_hook_z_reshaping_mode = sae.hook_z_reshaping_mode
        sae.turn_off_forward_pass_hook_z_reshaping()
    else:
        previous_hook_z_reshaping_mode = None

    # Get Reconstruction Score
    losses_df = recons_loss_batched(
        sae,
        model,
        activation_store,
        n_batches=n_eval_batches,
        eval_batch_size_prompts=eval_batch_size_prompts,
    )

    recons_score = losses_df["score"].mean()
    ntp_loss = losses_df["loss"].mean()
    recons_loss = losses_df["recons_loss"].mean()
    zero_abl_loss = losses_df["zero_abl_loss"].mean()

    # get cache
    _, cache = model.run_with_cache(
        eval_tokens,
        prepend_bos=False,
        names_filter=[hook_name],
        **model_kwargs,
    )

    # we would include hook z, except that we now have base SAE's
    # which will do their own reshaping for hook z.
    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if hook_head_index is not None:
        original_act = cache[hook_name][:, :, hook_head_index]
    elif any(substring in hook_name for substring in has_head_dim_key_substrings):
        original_act = cache[hook_name].flatten(-2, -1)
    else:
        original_act = cache[hook_name]

    # normalise if necessary
    if activation_store.normalize_activations:
        original_act = activation_store.apply_norm_scaling_factor(original_act)

    # send the (maybe normalised) activations into the SAE
    sae_out = sae.decode(sae.encode(original_act))
    del cache

    l2_norm_in = torch.norm(original_act, dim=-1)
    l2_norm_out = torch.norm(sae_out, dim=-1)
    l2_norm_in_for_div = l2_norm_in.clone()
    l2_norm_in_for_div[torch.abs(l2_norm_in_for_div) < 0.0001] = 1
    l2_norm_ratio = l2_norm_out / l2_norm_in_for_div

    metrics = {
        # l2 norms
        "metrics/l2_norm": l2_norm_out.mean().item(),
        "metrics/l2_ratio": l2_norm_ratio.mean().item(),
        "metrics/l2_norm_in": l2_norm_in.mean().item(),
        # CE Loss
        "metrics/CE_loss_score": recons_score,
        "metrics/ce_loss_without_sae": ntp_loss,
        "metrics/ce_loss_with_sae": recons_loss,
        "metrics/ce_loss_with_ablation": zero_abl_loss,
    }

    # restore previous hook z reshaping mode if necessary
    if "hook_z" in hook_name:
        if previous_hook_z_reshaping_mode and not sae.hook_z_reshaping_mode:
            sae.turn_on_forward_pass_hook_z_reshaping()
        elif not previous_hook_z_reshaping_mode and sae.hook_z_reshaping_mode:
            sae.turn_off_forward_pass_hook_z_reshaping()

    return metrics


def recons_loss_batched(
    sae: SAE,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    n_batches: int = 100,
    eval_batch_size_prompts: int | None = None,
):
    losses = []
    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)
        score, loss, recons_loss, zero_abl_loss = get_recons_loss(
            sae,
            model,
            batch_tokens,
            activation_store,
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
    sae: SAE,
    model: HookedRootModule,
    batch_tokens: torch.Tensor,
    activation_store: ActivationsStore,
    model_kwargs: Mapping[str, Any] = {},
):
    hook_name = sae.cfg.hook_name
    head_index = sae.cfg.hook_head_index

    loss = model(batch_tokens, return_type="loss", **model_kwargs)

    # TODO(tomMcGrath): the rescaling below is a bit of a hack and could probably be tidied up
    def standard_replacement_hook(activations: torch.Tensor, hook: Any):
        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations:
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        activations = sae.decode(sae.encode(activations)).to(activations.dtype)

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations:
            activations = activation_store.unscale(activations)
        return activations

    def all_head_replacement_hook(activations: torch.Tensor, hook: Any):
        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations:
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        new_activations = sae.decode(sae.encode(activations.flatten(-2, -1))).to(
            activations.dtype
        )

        new_activations = new_activations.reshape(
            activations.shape
        )  # reshape to match original shape

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations:
            new_activations = activation_store.unscale(new_activations)

        return new_activations

    def single_head_replacement_hook(activations: torch.Tensor, hook: Any):
        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations:
            activations = activation_store.apply_norm_scaling_factor(activations)

        new_activations = sae.decoder(sae.encode(activations[:, :, head_index])).to(
            activations.dtype
        )
        activations[:, :, head_index] = new_activations

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations:
            activations = activation_store.unscale(activations)
        return activations

    def zero_ablate_hook(activations: torch.Tensor, hook: Any):
        activations = torch.zeros_like(activations)
        return activations

    # we would include hook z, except that we now have base SAE's
    # which will do their own reshaping for hook z.
    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if any(substring in hook_name for substring in has_head_dim_key_substrings):
        if head_index is None:
            replacement_hook = all_head_replacement_hook
        else:
            replacement_hook = single_head_replacement_hook
    else:
        replacement_hook = standard_replacement_hook

    recons_loss = model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(hook_name, partial(replacement_hook))],
        **model_kwargs,
    )

    zero_abl_loss = model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(hook_name, zero_ablate_hook)],
        **model_kwargs,
    )

    div_val = zero_abl_loss - loss
    div_val[torch.abs(div_val) < 0.0001] = 1.0

    score = (zero_abl_loss - recons_loss) / div_val

    return score, loss, recons_loss, zero_abl_loss
