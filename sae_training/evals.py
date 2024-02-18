from functools import partial

import torch
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
):
    hook_point = sparse_autoencoder.cfg.hook_point
    hook_point_layer = sparse_autoencoder.cfg.hook_point_layer
    hook_point_head_index = sparse_autoencoder.cfg.hook_point_head_index

    ### Evals
    eval_tokens = activation_store.get_batch_tokens()

    # Get Reconstruction Score
    recons_score, ntp_loss, recons_loss, zero_abl_loss = get_recons_loss(
        sparse_autoencoder, model, activation_store, eval_tokens
    )

    # get cache
    _, cache = model.run_with_cache(
        eval_tokens,
        prepend_bos=False,
        names_filter=[get_act_name("pattern", hook_point_layer), hook_point],
    )

    # get act
    if sparse_autoencoder.cfg.hook_point_head_index is not None:
        original_act = cache[sparse_autoencoder.cfg.hook_point][
            :, :, sparse_autoencoder.cfg.hook_point_head_index
        ]
    else:
        original_act = cache[sparse_autoencoder.cfg.hook_point]

    sae_out, feature_acts, _, _, _, _ = sparse_autoencoder(original_act)
    patterns_original = (
        cache[get_act_name("pattern", hook_point_layer)][:, hook_point_head_index]
        .detach()
        .cpu()
    )
    del cache

    if "cuda" in str(model.cfg.device):
        torch.cuda.empty_cache()

    l2_norm_in = torch.norm(original_act, dim=-1)
    l2_norm_out = torch.norm(sae_out, dim=-1)
    l2_norm_ratio = l2_norm_out / l2_norm_in

    wandb.log(
        {
            # l2 norms
            "metrics/l2_norm": l2_norm_out.mean().item(),
            "metrics/l2_ratio": l2_norm_ratio.mean().item(),
            # CE Loss
            "metrics/CE_loss_score": recons_score,
            "metrics/ce_loss_without_sae": ntp_loss,
            "metrics/ce_loss_with_sae": recons_loss,
            "metrics/ce_loss_with_ablation": zero_abl_loss,
        },
        step=n_training_steps,
    )

    head_index = sparse_autoencoder.cfg.hook_point_head_index

    def standard_replacement_hook(activations, hook):
        activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
        return activations

    def head_replacement_hook(activations, hook):
        new_actions = sparse_autoencoder.forward(activations[:, :, head_index])[0].to(
            activations.dtype
        )
        activations[:, :, head_index] = new_actions
        return activations

    head_index = sparse_autoencoder.cfg.hook_point_head_index
    replacement_hook = (
        standard_replacement_hook if head_index is None else head_replacement_hook
    )

    # get attn when using reconstructed activations
    with model.hooks(fwd_hooks=[(hook_point, partial(replacement_hook))]):
        _, new_cache = model.run_with_cache(
            eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)]
        )
        patterns_reconstructed = (
            new_cache[get_act_name("pattern", hook_point_layer)][
                :, hook_point_head_index
            ]
            .detach()
            .cpu()
        )
        del new_cache

    # get attn when using reconstructed activations
    with model.hooks(fwd_hooks=[(hook_point, partial(zero_ablate_hook))]):
        _, zero_ablation_cache = model.run_with_cache(
            eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)]
        )
        patterns_ablation = (
            zero_ablation_cache[get_act_name("pattern", hook_point_layer)][
                :, hook_point_head_index
            ]
            .detach()
            .cpu()
        )
        del zero_ablation_cache

    if sparse_autoencoder.cfg.hook_point_head_index:
        kl_result_reconstructed = kl_divergence_attention(
            patterns_original, patterns_reconstructed
        )
        kl_result_reconstructed = kl_result_reconstructed.sum(dim=-1).numpy()

        kl_result_ablation = kl_divergence_attention(
            patterns_original, patterns_ablation
        )
        kl_result_ablation = kl_result_ablation.sum(dim=-1).numpy()

        wandb.log(
            {
                "metrics/kldiv_reconstructed": kl_result_reconstructed.mean().item(),
                "metrics/kldiv_ablation": kl_result_ablation.mean().item(),
            },
            step=n_training_steps,
        )


@torch.no_grad()
def get_recons_loss(sparse_autoencoder, model, activation_store, batch_tokens):
    def logits_to_interval_loss(logits):
        clps = model.loss_fn(logits, batch_tokens, per_token=True)
        loss = clps[:, sparse_autoencoder.cfg.start_pos_offset:sparse_autoencoder.cfg.end_pos_offset]
        return loss.mean()

    hook_point = activation_store.cfg.hook_point
    loss = logits_to_interval_loss(model(batch_tokens, return_type="logits"))
    

    head_index = sparse_autoencoder.cfg.hook_point_head_index

    def standard_replacement_hook(activations, hook):
        activations = sparse_autoencoder.forward(activations)[0].to(activations.dtype)
        return activations

    def head_replacement_hook(activations, hook):
        new_actions = sparse_autoencoder.forward(activations[:, :, head_index])[0].to(
            activations.dtype
        )
        activations[:, :, head_index] = new_actions
        return activations

    replacement_hook = (
        standard_replacement_hook if head_index is None else head_replacement_hook
    )
    recons_loss = logits_to_interval_loss(model.run_with_hooks(
        batch_tokens,
        return_type="logits",
        fwd_hooks=[(hook_point, partial(replacement_hook))],
    ))

    zero_abl_loss = logits_to_interval_loss(model.run_with_hooks(
        batch_tokens, return_type="logits", fwd_hooks=[(hook_point, zero_ablate_hook)]
    ))

    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    return score, loss, recons_loss, zero_abl_loss


def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1]).to(mlp_post.dtype)
    return mlp_post


def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.0
    return mlp_post


def kl_divergence_attention(y_true, y_pred):
    # Compute log probabilities for KL divergence
    log_y_true = torch.log2(y_true + 1e-10)
    log_y_pred = torch.log2(y_pred + 1e-10)

    return y_true * (log_y_true - log_y_pred)
