import argparse
import re
from functools import partial
from typing import Any, Mapping, Tuple

import einops
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule

from sae_lens.sae import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.training.activations_store import ActivationsStore


@torch.no_grad()
def run_evals(
    sae: SAE,
    activation_store: ActivationsStore,
    model: HookedRootModule,
    n_eval_batches: int = 10,
    eval_batch_size_prompts: int | None = None,
    model_kwargs: Mapping[str, Any] = {},
) -> dict[str, Any]:

    hook_name = sae.cfg.hook_name
    actual_batch_size = (
        eval_batch_size_prompts or activation_store.store_batch_size_prompts
    )

    # TODO: Come up with a cleaner long term strategy here for SAEs that do reshaping.
    # turn off hook_z reshaping mode if it's on, and restore it after evals
    if "hook_z" in hook_name:
        previous_hook_z_reshaping_mode = sae.hook_z_reshaping_mode
        sae.turn_off_forward_pass_hook_z_reshaping()
    else:
        previous_hook_z_reshaping_mode = None

    metrics = get_downstream_reconstruction_metrics(
        sae,
        model,
        activation_store,
        n_batches=n_eval_batches,
        eval_batch_size_prompts=actual_batch_size,
    )

    activation_store.reset_input_dataset()

    metrics |= get_sparsity_and_variance_metrics(
        sae,
        model,
        activation_store,
        n_batches=n_eval_batches,
        eval_batch_size_prompts=actual_batch_size,
        model_kwargs=model_kwargs,
    )

    # restore previous hook z reshaping mode if necessary
    if "hook_z" in hook_name:
        if previous_hook_z_reshaping_mode and not sae.hook_z_reshaping_mode:
            sae.turn_on_forward_pass_hook_z_reshaping()
        elif not previous_hook_z_reshaping_mode and sae.hook_z_reshaping_mode:
            sae.turn_off_forward_pass_hook_z_reshaping()

    total_tokens_evaluated = (
        activation_store.context_size * n_eval_batches * actual_batch_size
    )
    metrics["metrics/total_tokens_evaluated"] = total_tokens_evaluated

    return metrics


def get_downstream_reconstruction_metrics(
    sae: SAE,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    n_batches: int,
    eval_batch_size_prompts: int,
):
    metrics = []
    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)
        (
            recons_kl_div,
            zero_abl_kl_div,
            original_ce_loss,
            recons_ce_loss,
            zero_abl_ce_loss,
        ) = get_recons_loss(
            sae,
            model,
            batch_tokens,
            activation_store,
        )

        metrics.append(
            (
                recons_kl_div,
                zero_abl_kl_div,
                original_ce_loss,
                recons_ce_loss,
                zero_abl_ce_loss,
            )
        )

    recons_kl_div = torch.stack([metric[0] for metric in metrics]).mean()
    zero_abl_kl_div = torch.stack([metric[1] for metric in metrics]).mean()
    kl_div_score = (zero_abl_kl_div - recons_kl_div) / zero_abl_kl_div

    zero_abl_ce_loss = torch.stack([metric[4] for metric in metrics]).mean()
    recons_ce_loss = torch.stack([metric[3] for metric in metrics]).mean()
    original_ce_loss = torch.stack([metric[2] for metric in metrics]).mean()
    ce_loss_score = (zero_abl_ce_loss - recons_ce_loss) / (
        zero_abl_ce_loss - original_ce_loss
    )

    metrics = {
        "metrics/ce_loss_score": ce_loss_score.item(),
        "metrics/ce_loss_without_sae": original_ce_loss.item(),
        "metrics/ce_loss_with_sae": recons_ce_loss.item(),
        "metrics/ce_loss_with_ablation": zero_abl_ce_loss.item(),
        "metrics/kl_div_score": kl_div_score.item(),
        "metrics/kl_div_without_sae": 0,
        "metrics/kl_div_with_sae": recons_kl_div.item(),
        "metrics/kl_div_with_ablation": zero_abl_kl_div.item(),
    }

    return metrics


def get_sparsity_and_variance_metrics(
    sae: SAE,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    n_batches: int,
    eval_batch_size_prompts: int,
    model_kwargs: Mapping[str, Any],
):

    metrics_list = []

    hook_name = sae.cfg.hook_name
    hook_head_index = sae.cfg.hook_head_index

    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)

        # get cache
        _, cache = model.run_with_cache(
            batch_tokens,
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
        if activation_store.normalize_activations == "expected_average_only_in":
            original_act = activation_store.apply_norm_scaling_factor(original_act)

        # send the (maybe normalised) activations into the SAE
        sae_feature_activations = sae.encode(original_act.to(sae.device))
        sae_out = sae.decode(sae_feature_activations).to(original_act.device)
        del cache

        flattened_sae_input = einops.rearrange(original_act, "b ctx d -> (b ctx) d")
        flattened_sae_feature_acts = einops.rearrange(
            sae_feature_activations, "b ctx d -> (b ctx) d"
        )
        flattened_sae_out = einops.rearrange(sae_out, "b ctx d -> (b ctx) d")

        l2_norm_in = torch.norm(flattened_sae_input, dim=-1)
        l2_norm_out = torch.norm(flattened_sae_out, dim=-1)
        l2_norm_in_for_div = l2_norm_in.clone()
        l2_norm_in_for_div[torch.abs(l2_norm_in_for_div) < 0.0001] = 1
        l2_norm_ratio = l2_norm_out / l2_norm_in_for_div

        l0 = (flattened_sae_feature_acts > 0).sum(dim=-1)
        l1 = flattened_sae_feature_acts.sum(dim=-1)
        resid_sum_of_squares = (
            (flattened_sae_input - flattened_sae_out).pow(2).sum(dim=-1)
        )
        total_sum_of_squares = (
            (flattened_sae_input - flattened_sae_input.mean(dim=0)).pow(2).sum(-1)
        )
        explained_variance = 1 - resid_sum_of_squares / total_sum_of_squares

        metrics_list.append(
            (
                l2_norm_in,
                l2_norm_out,
                l2_norm_ratio,
                explained_variance,
                l0.float(),
                l1,
                resid_sum_of_squares,
            )
        )

    metrics: dict[str, float] = {}
    for i, metric_name in enumerate(
        [
            "l2_norm_in",
            "l2_norm_out",
            "l2_ratio",
            "explained_variance",
            "l0",
            "l1",
            "mse",
        ]
    ):
        metrics[f"metrics/{metric_name}"] = (
            torch.stack([m[i] for m in metrics_list]).mean().item()
        )

    return metrics


@torch.no_grad()
def get_recons_loss(
    sae: SAE,
    model: HookedRootModule,
    batch_tokens: torch.Tensor,
    activation_store: ActivationsStore,
    model_kwargs: Mapping[str, Any] = {},
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hook_name = sae.cfg.hook_name
    head_index = sae.cfg.hook_head_index

    original_logits, original_ce_loss = model(
        batch_tokens, return_type="both", **model_kwargs
    )

    # TODO(tomMcGrath): the rescaling below is a bit of a hack and could probably be tidied up
    def standard_replacement_hook(activations: torch.Tensor, hook: Any):

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        activations = sae.decode(sae.encode(activations)).to(activations.dtype)

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.unscale(activations)

        return activations.to(original_device)

    def all_head_replacement_hook(activations: torch.Tensor, hook: Any):

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        new_activations = sae.decode(sae.encode(activations.flatten(-2, -1))).to(
            activations.dtype
        )

        new_activations = new_activations.reshape(
            activations.shape
        )  # reshape to match original shape

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            new_activations = activation_store.unscale(new_activations)

        return new_activations.to(original_device)

    def single_head_replacement_hook(activations: torch.Tensor, hook: Any):

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        new_activations = sae.decoder(sae.encode(activations[:, :, head_index])).to(
            activations.dtype
        )
        activations[:, :, head_index] = new_activations

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.unscale(activations)
        return activations.to(original_device)

    def zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations = torch.zeros_like(activations)
        return activations.to(original_device)

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

    recons_logits, recons_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, partial(replacement_hook))],
        **model_kwargs,
    )

    zero_abl_logits, zero_abl_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, zero_ablate_hook)],
        **model_kwargs,
    )

    def compute_kl(original_logits: torch.Tensor, new_logits: torch.Tensor):
        original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
        log_original_probs = torch.log(original_probs)
        new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
        log_new_probs = torch.log(new_probs)
        kl_div = original_probs * (log_original_probs - log_new_probs)
        kl_div = kl_div.sum(dim=-1)
        return kl_div

    recons_kl_div = compute_kl(original_logits, recons_logits)
    zero_abl_kl_div = compute_kl(original_logits, zero_abl_logits)

    return (
        recons_kl_div,
        zero_abl_kl_div,
        original_ce_loss,
        recons_ce_loss,
        zero_abl_ce_loss,
    )


def all_loadable_saes() -> list[tuple[str, str, float, float]]:
    all_loadable_saes = []
    saes_directory = get_pretrained_saes_directory()
    for release, lookup in saes_directory.items():
        for sae_name in lookup.saes_map.keys():
            expected_var_explained = lookup.expected_var_explained[sae_name]
            expected_l0 = lookup.expected_l0[sae_name]
            all_loadable_saes.append(
                (release, sae_name, expected_var_explained, expected_l0)
            )

    return all_loadable_saes


def multiple_evals(
    sae_regex_pattern: str,
    sae_block_pattern: str,
    num_eval_batches: int = 10,
    eval_batch_size_prompts: int = 8,
    datasets: list[str] = ["Skylion007/openwebtext", "lighteval/MATH"],
    ctx_lens: list[int] = [64, 128, 256, 512],
) -> pd.DataFrame:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sae_regex_compiled = re.compile(sae_regex_pattern)
    sae_block_compiled = re.compile(sae_block_pattern)
    all_saes = all_loadable_saes()
    filtered_saes = [
        sae
        for sae in all_saes
        if sae_regex_compiled.fullmatch(sae[0]) and sae_block_compiled.fullmatch(sae[1])
    ]

    assert len(filtered_saes) > 0, "No SAEs matched the given regex patterns"

    eval_results = []

    current_model = None
    current_model_str = None
    print(filtered_saes)
    for sae_name, sae_block, _, _ in tqdm(filtered_saes):

        sae = SAE.from_pretrained(
            release=sae_name,  # see other options in sae_lens/pretrained_saes.yaml
            sae_id=sae_block,  # won't always be a hook point
            device=device,
        )[0]

        if current_model_str != sae.cfg.model_name:
            del current_model  # potentially saves GPU memory
            current_model_str = sae.cfg.model_name
            current_model = HookedTransformer.from_pretrained(
                current_model_str, device=device
            )
        assert current_model is not None

        for ctx_len in ctx_lens:
            for dataset in datasets:

                activation_store = ActivationsStore.from_sae(
                    current_model, sae, context_size=ctx_len, dataset=dataset
                )
                activation_store.shuffle_input_dataset(seed=42)

                eval_metrics = {}
                eval_metrics["sae_id"] = f"{sae_name}-{sae_block}"
                eval_metrics["context_size"] = ctx_len
                eval_metrics["dataset"] = dataset

                eval_metrics |= run_evals(
                    sae=sae,
                    activation_store=activation_store,
                    model=current_model,
                    n_eval_batches=num_eval_batches,
                    eval_batch_size_prompts=eval_batch_size_prompts,
                )

                eval_results.append(eval_metrics)

    return pd.DataFrame(eval_results)


if __name__ == "__main__":

    # Example commands:
    # python sae_lens/evals.py "gpt2-small-res-jb.*" "blocks.8.hook_resid_pre" --save_path "gpt2_small_jb_layer8_resid_pre_eval_results.csv"
    # python sae_lens/evals.py "gpt2-small.*" "blocks.8.hook_resid_pre" --save_path "gpt2_small_layer8_resid_pre_eval_results.csv"
    # python sae_lens/evals.py "gpt2-small.*" ".*" --save_path "gpt2_small_eval_results.csv"
    # python sae_lens/evals.py "mistral.*" ".*" --save_path "mistral_eval_results.csv"

    arg_parser = argparse.ArgumentParser(description="Run evaluations on SAEs")
    arg_parser.add_argument(
        "sae_regex_pattern",
        type=str,
        help="Regex pattern to match SAE names. Can be an entire SAE name to match a specific SAE.",
    )
    arg_parser.add_argument(
        "sae_block_pattern",
        type=str,
        help="Regex pattern to match SAE block names. Can be an entire block name to match a specific block.",
    )
    arg_parser.add_argument(
        "--num_eval_batches",
        type=int,
        default=10,
        help="Number of evaluation batches to run.",
    )
    arg_parser.add_argument(
        "--eval_batch_size_prompts",
        type=int,
        default=8,
        help="Batch size for evaluation prompts.",
    )
    arg_parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Skylion007/openwebtext", "lighteval/MATH"],
        help="Datasets to evaluate on.",
    )
    arg_parser.add_argument(
        "--ctx_lens",
        nargs="+",
        default=[64, 128, 256, 512],
        help="Context lengths to evaluate on.",
    )
    arg_parser.add_argument(
        "--save_path",
        type=str,
        default="eval_results.csv",
        help="Path to save evaluation results to.",
    )

    args = arg_parser.parse_args()

    eval_results = multiple_evals(
        sae_regex_pattern=args.sae_regex_pattern,
        sae_block_pattern=args.sae_block_pattern,
        num_eval_batches=args.num_eval_batches,
        eval_batch_size_prompts=args.eval_batch_size_prompts,
        datasets=args.datasets,
        ctx_lens=args.ctx_lens,
    )

    eval_results.to_csv(args.save_path, index=False)
