import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import einops
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule

from sae_lens.sae import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.training.activations_store import ActivationsStore


# Everything by default is false so the user can just set the ones they want to true
@dataclass
class EvalConfig:
    batch_size_prompts: int | None = None

    # Reconstruction metrics
    n_eval_reconstruction_batches: int = 10
    compute_kl: bool = False
    compute_ce_loss: bool = False

    # Sparsity and variance metrics
    n_eval_sparsity_variance_batches: int = 1
    compute_l2_norms: bool = False
    compute_sparsity_metrics: bool = False
    compute_variance_metrics: bool = False


def get_eval_everything_config(
    batch_size_prompts: int | None = None,
    n_eval_reconstruction_batches: int = 10,
    n_eval_sparsity_variance_batches: int = 1,
) -> EvalConfig:
    """
    Returns an EvalConfig object with all metrics set to True, so that when passed to run_evals all available metrics will be run.
    """
    return EvalConfig(
        batch_size_prompts=batch_size_prompts,
        n_eval_reconstruction_batches=n_eval_reconstruction_batches,
        compute_kl=True,
        compute_ce_loss=True,
        compute_l2_norms=True,
        n_eval_sparsity_variance_batches=n_eval_sparsity_variance_batches,
        compute_sparsity_metrics=True,
        compute_variance_metrics=True,
    )


@torch.no_grad()
def run_evals(
    sae: SAE,
    activation_store: ActivationsStore,
    model: HookedRootModule,
    eval_config: EvalConfig = EvalConfig(),
    model_kwargs: Mapping[str, Any] = {},
    ignore_tokens: set[int | None] = set(),
) -> dict[str, Any]:

    hook_name = sae.cfg.hook_name
    actual_batch_size = (
        eval_config.batch_size_prompts or activation_store.store_batch_size_prompts
    )

    # TODO: Come up with a cleaner long term strategy here for SAEs that do reshaping.
    # turn off hook_z reshaping mode if it's on, and restore it after evals
    if "hook_z" in hook_name:
        previous_hook_z_reshaping_mode = sae.hook_z_reshaping_mode
        sae.turn_off_forward_pass_hook_z_reshaping()
    else:
        previous_hook_z_reshaping_mode = None

    metrics = {}

    if eval_config.compute_kl or eval_config.compute_ce_loss:
        assert eval_config.n_eval_reconstruction_batches > 0
        metrics |= get_downstream_reconstruction_metrics(
            sae,
            model,
            activation_store,
            compute_kl=eval_config.compute_kl,
            compute_ce_loss=eval_config.compute_ce_loss,
            n_batches=eval_config.n_eval_reconstruction_batches,
            eval_batch_size_prompts=actual_batch_size,
            ignore_tokens=ignore_tokens,
        )

        activation_store.reset_input_dataset()

    if (
        eval_config.compute_l2_norms
        or eval_config.compute_sparsity_metrics
        or eval_config.compute_variance_metrics
    ):
        assert eval_config.n_eval_sparsity_variance_batches > 0
        metrics |= get_sparsity_and_variance_metrics(
            sae,
            model,
            activation_store,
            compute_l2_norms=eval_config.compute_l2_norms,
            compute_sparsity_metrics=eval_config.compute_sparsity_metrics,
            compute_variance_metrics=eval_config.compute_variance_metrics,
            n_batches=eval_config.n_eval_sparsity_variance_batches,
            eval_batch_size_prompts=actual_batch_size,
            model_kwargs=model_kwargs,
            ignore_tokens=ignore_tokens,
        )

    if len(metrics) == 0:
        raise ValueError(
            "No metrics were computed, please set at least one metric to True."
        )

    # restore previous hook z reshaping mode if necessary
    if "hook_z" in hook_name:
        if previous_hook_z_reshaping_mode and not sae.hook_z_reshaping_mode:
            sae.turn_on_forward_pass_hook_z_reshaping()
        elif not previous_hook_z_reshaping_mode and sae.hook_z_reshaping_mode:
            sae.turn_off_forward_pass_hook_z_reshaping()

    total_tokens_evaluated = (
        activation_store.context_size
        * eval_config.n_eval_reconstruction_batches
        * actual_batch_size
    )
    metrics["metrics/total_tokens_evaluated"] = total_tokens_evaluated

    return metrics


def get_downstream_reconstruction_metrics(
    sae: SAE,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    compute_kl: bool,
    compute_ce_loss: bool,
    n_batches: int,
    eval_batch_size_prompts: int,
    ignore_tokens: set[int | None] = set(),
):
    metrics_dict = {}
    if compute_kl:
        metrics_dict["kl_div_with_sae"] = []
        metrics_dict["kl_div_with_ablation"] = []
    if compute_ce_loss:
        metrics_dict["ce_loss_with_sae"] = []
        metrics_dict["ce_loss_without_sae"] = []
        metrics_dict["ce_loss_with_ablation"] = []

    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)
        for metric_name, metric_value in get_recons_loss(
            sae,
            model,
            batch_tokens,
            activation_store,
            compute_kl=compute_kl,
            compute_ce_loss=compute_ce_loss,
        ).items():

            if len(ignore_tokens) > 0:
                mask = torch.logical_not(
                    torch.any(
                        torch.stack(
                            [batch_tokens == token for token in ignore_tokens], dim=0
                        ),
                        dim=0,
                    )
                )
                if metric_value.shape[1] != mask.shape[1]:
                    # ce loss will be missing the last value
                    mask = mask[:, :-1]
                metric_value = metric_value[mask]

            metrics_dict[metric_name].append(metric_value)

    metrics: dict[str, float] = {}
    for metric_name, metric_values in metrics_dict.items():
        metrics[f"metrics/{metric_name}"] = torch.cat(metric_values).mean().item()

    if compute_kl:
        metrics["metrics/kl_div_score"] = (
            metrics["metrics/kl_div_with_ablation"] - metrics["metrics/kl_div_with_sae"]
        ) / metrics["metrics/kl_div_with_ablation"]

    if compute_ce_loss:
        metrics["metrics/ce_loss_score"] = (
            metrics["metrics/ce_loss_with_ablation"]
            - metrics["metrics/ce_loss_with_sae"]
        ) / (
            metrics["metrics/ce_loss_with_ablation"]
            - metrics["metrics/ce_loss_without_sae"]
        )

    return metrics


def get_sparsity_and_variance_metrics(
    sae: SAE,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    n_batches: int,
    compute_l2_norms: bool,
    compute_sparsity_metrics: bool,
    compute_variance_metrics: bool,
    eval_batch_size_prompts: int,
    model_kwargs: Mapping[str, Any],
    ignore_tokens: set[int | None] = set(),
):

    hook_name = sae.cfg.hook_name
    hook_head_index = sae.cfg.hook_head_index

    metric_dict = {}
    if compute_l2_norms:
        metric_dict["l2_norm_in"] = []
        metric_dict["l2_norm_out"] = []
        metric_dict["l2_ratio"] = []
    if compute_sparsity_metrics:
        metric_dict["l0"] = []
        metric_dict["l1"] = []
    if compute_variance_metrics:
        metric_dict["explained_variance"] = []
        metric_dict["mse"] = []

    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)

        if len(ignore_tokens) > 0:
            mask = torch.logical_not(
                torch.any(
                    torch.stack(
                        [batch_tokens == token for token in ignore_tokens], dim=0
                    ),
                    dim=0,
                )
            )
        else:
            mask = torch.ones_like(batch_tokens, dtype=torch.bool)
        flattened_mask = mask.flatten()

        # get cache
        _, cache = model.run_with_cache(
            batch_tokens,
            prepend_bos=False,
            names_filter=[hook_name],
            stop_at_layer=sae.cfg.hook_layer + 1,
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

        # normalise if necessary (necessary in training only, otherwise we should fold the scaling in)
        if activation_store.normalize_activations == "expected_average_only_in":
            original_act = activation_store.apply_norm_scaling_factor(original_act)

        # send the (maybe normalised) activations into the SAE
        sae_feature_activations = sae.encode(original_act.to(sae.device))
        sae_out = sae.decode(sae_feature_activations).to(original_act.device)
        del cache

        if activation_store.normalize_activations == "expected_average_only_in":
            sae_out = activation_store.unscale(sae_out)

        flattened_sae_input = einops.rearrange(original_act, "b ctx d -> (b ctx) d")
        flattened_sae_feature_acts = einops.rearrange(
            sae_feature_activations, "b ctx d -> (b ctx) d"
        )
        flattened_sae_out = einops.rearrange(sae_out, "b ctx d -> (b ctx) d")

        # apply mask
        flattened_sae_input = flattened_sae_input[flattened_mask]
        flattened_sae_feature_acts = flattened_sae_feature_acts[flattened_mask]
        flattened_sae_out = flattened_sae_out[flattened_mask]

        if compute_l2_norms:
            l2_norm_in = torch.norm(flattened_sae_input, dim=-1)
            l2_norm_out = torch.norm(flattened_sae_out, dim=-1)
            l2_norm_in_for_div = l2_norm_in.clone()
            l2_norm_in_for_div[torch.abs(l2_norm_in_for_div) < 0.0001] = 1
            l2_norm_ratio = l2_norm_out / l2_norm_in_for_div
            metric_dict["l2_norm_in"].append(l2_norm_in)
            metric_dict["l2_norm_out"].append(l2_norm_out)
            metric_dict["l2_ratio"].append(l2_norm_ratio)

        if compute_sparsity_metrics:
            l0 = (flattened_sae_feature_acts > 0).sum(dim=-1).float()
            l1 = flattened_sae_feature_acts.sum(dim=-1)
            metric_dict["l0"].append(l0)
            metric_dict["l1"].append(l1)

        if compute_variance_metrics:
            resid_sum_of_squares = (
                (flattened_sae_input - flattened_sae_out).pow(2).sum(dim=-1)
            )
            total_sum_of_squares = (
                (flattened_sae_input - flattened_sae_input.mean(dim=0)).pow(2).sum(-1)
            )
            mse = resid_sum_of_squares / flattened_mask.sum()
            explained_variance = 1 - resid_sum_of_squares / total_sum_of_squares
            metric_dict["explained_variance"].append(explained_variance)
            metric_dict["mse"].append(mse)

    metrics: dict[str, float] = {}
    for metric_name, metric_values in metric_dict.items():
        # since we're masking, we need to flatten but may not have n_ctx for all metrics
        # in all batches.
        metrics[f"metrics/{metric_name}"] = torch.cat(metric_values).mean().item()

    return metrics


@torch.no_grad()
def get_recons_loss(
    sae: SAE,
    model: HookedRootModule,
    batch_tokens: torch.Tensor,
    activation_store: ActivationsStore,
    compute_kl: bool,
    compute_ce_loss: bool,
    model_kwargs: Mapping[str, Any] = {},
) -> dict[str, Any]:
    hook_name = sae.cfg.hook_name
    head_index = sae.cfg.hook_head_index

    original_logits, original_ce_loss = model(
        batch_tokens, return_type="both", loss_per_token=True, **model_kwargs
    )

    metrics = {}

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

        new_activations = sae.decode(sae.encode(activations[:, :, head_index])).to(
            activations.dtype
        )
        activations[:, :, head_index] = new_activations

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.unscale(activations)
        return activations.to(original_device)

    def standard_zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations = torch.zeros_like(activations)
        return activations.to(original_device)

    def single_head_zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations[:, :, head_index] = torch.zeros_like(activations[:, :, head_index])
        return activations.to(original_device)

    # we would include hook z, except that we now have base SAE's
    # which will do their own reshaping for hook z.
    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if any(substring in hook_name for substring in has_head_dim_key_substrings):
        if head_index is None:
            replacement_hook = all_head_replacement_hook
            zero_ablate_hook = standard_zero_ablate_hook
        else:
            replacement_hook = single_head_replacement_hook
            zero_ablate_hook = single_head_zero_ablate_hook
    else:
        replacement_hook = standard_replacement_hook
        zero_ablate_hook = standard_zero_ablate_hook

    recons_logits, recons_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, partial(replacement_hook))],
        loss_per_token=True,
        **model_kwargs,
    )

    zero_abl_logits, zero_abl_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, zero_ablate_hook)],
        loss_per_token=True,
        **model_kwargs,
    )

    def kl(original_logits: torch.Tensor, new_logits: torch.Tensor):
        original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
        log_original_probs = torch.log(original_probs)
        new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
        log_new_probs = torch.log(new_probs)
        kl_div = original_probs * (log_original_probs - log_new_probs)
        kl_div = kl_div.sum(dim=-1)
        return kl_div

    if compute_kl:
        recons_kl_div = kl(original_logits, recons_logits)
        zero_abl_kl_div = kl(original_logits, zero_abl_logits)
        metrics["kl_div_with_sae"] = recons_kl_div
        metrics["kl_div_with_ablation"] = zero_abl_kl_div

    if compute_ce_loss:
        metrics["ce_loss_with_sae"] = recons_ce_loss
        metrics["ce_loss_without_sae"] = original_ce_loss
        metrics["ce_loss_with_ablation"] = zero_abl_ce_loss

    return metrics


def all_loadable_saes() -> list[tuple[str, str, float, float]]:
    all_loadable_saes = []
    saes_directory = get_pretrained_saes_directory()
    for release, lookup in tqdm(saes_directory.items()):
        for sae_name in lookup.saes_map.keys():
            expected_var_explained = lookup.expected_var_explained[sae_name]
            expected_l0 = lookup.expected_l0[sae_name]
            all_loadable_saes.append(
                (release, sae_name, expected_var_explained, expected_l0)
            )

    return all_loadable_saes


def get_saes_from_regex(
    sae_regex_pattern: str, sae_block_pattern: str
) -> list[tuple[str, str, float, float]]:
    sae_regex_compiled = re.compile(sae_regex_pattern)
    sae_block_compiled = re.compile(sae_block_pattern)
    all_saes = all_loadable_saes()
    filtered_saes = [
        sae
        for sae in all_saes
        if sae_regex_compiled.fullmatch(sae[0]) and sae_block_compiled.fullmatch(sae[1])
    ]
    return filtered_saes


def nested_dict() -> defaultdict[Any, Any]:
    return defaultdict(nested_dict)


def dict_to_nested(flat_dict: dict[str, Any]) -> defaultdict[Any, Any]:
    nested = nested_dict()
    for key, value in flat_dict.items():
        parts = key.split("/")
        d = nested
        for part in parts[:-1]:
            d = d[part]
        d[parts[-1]] = value
    return nested


def multiple_evals(
    sae_regex_pattern: str,
    sae_block_pattern: str,
    num_eval_batches: int = 10,
    eval_batch_size_prompts: int = 8,
    datasets: list[str] = ["Skylion007/openwebtext", "lighteval/MATH"],
    ctx_lens: list[int] = [128],
    output_dir: str = "eval_results",
) -> list[defaultdict[Any, Any]]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    filtered_saes = get_saes_from_regex(sae_regex_pattern, sae_block_pattern)

    assert len(filtered_saes) > 0, "No SAEs matched the given regex patterns"

    eval_results = []
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    eval_config = get_eval_everything_config(
        batch_size_prompts=eval_batch_size_prompts,
        n_eval_reconstruction_batches=num_eval_batches,
        n_eval_sparsity_variance_batches=num_eval_batches,
    )

    current_model = None
    current_model_str = None
    print(filtered_saes)
    for sae_release_name, sae_id, _, _ in tqdm(filtered_saes):

        sae = SAE.from_pretrained(
            release=sae_release_name,  # see other options in sae_lens/pretrained_saes.yaml
            sae_id=sae_id,  # won't always be a hook point
            device=device,
        )[0]

        if current_model_str != sae.cfg.model_name:
            del current_model  # potentially saves GPU memory
            current_model_str = sae.cfg.model_name
            current_model = HookedTransformer.from_pretrained_no_processing(
                current_model_str, device=device, **sae.cfg.model_from_pretrained_kwargs
            )
        assert current_model is not None

        for ctx_len in ctx_lens:
            for dataset in datasets:

                activation_store = ActivationsStore.from_sae(
                    current_model, sae, context_size=ctx_len, dataset=dataset
                )
                activation_store.shuffle_input_dataset(seed=42)

                eval_metrics = nested_dict()
                eval_metrics["unique_id"] = f"{sae_release_name}-{sae_id}"
                eval_metrics["sae_set"] = f"{sae_release_name}"
                eval_metrics["sae_id"] = f"{sae_id}"
                eval_metrics["eval_cfg"]["context_size"] = ctx_len
                eval_metrics["eval_cfg"]["dataset"] = dataset

                run_eval_metrics = run_evals(
                    sae=sae,
                    activation_store=activation_store,
                    model=current_model,
                    eval_config=eval_config,
                    ignore_tokens={
                        current_model.tokenizer.pad_token_id,  # type: ignore
                        current_model.tokenizer.eos_token_id,  # type: ignore
                        current_model.tokenizer.bos_token_id,  # type: ignore
                    },
                )
                eval_metrics["metrics"] = run_eval_metrics

                # Add SAE config
                eval_metrics["sae_cfg"] = sae.cfg.to_dict()

                # Add eval config
                eval_metrics["eval_cfg"].update(eval_config.__dict__)

                eval_results.append(eval_metrics)

    return eval_results


def run_evaluations(args: argparse.Namespace) -> list[defaultdict[Any, Any]]:
    # Filter SAEs based on regex patterns
    filtered_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)

    num_sae_sets = len(set(sae_set for sae_set, _, _, _ in filtered_saes))
    num_all_sae_ids = len(filtered_saes)

    print("Filtered SAEs based on provided patterns:")
    print(f"Number of SAE sets: {num_sae_sets}")
    print(f"Total number of SAE IDs: {num_all_sae_ids}")

    eval_results = multiple_evals(
        sae_regex_pattern=args.sae_regex_pattern,
        sae_block_pattern=args.sae_block_pattern,
        num_eval_batches=args.num_eval_batches,
        eval_batch_size_prompts=args.eval_batch_size_prompts,
        datasets=args.datasets,
        ctx_lens=args.ctx_lens,
        output_dir=args.output_dir,
    )

    return eval_results


def process_results(eval_results: list[defaultdict[Any, Any]], output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save individual JSON files
    for result in eval_results:
        json_filename = f"{result['unique_id']}_{result['eval_cfg']['context_size']}_{result['eval_cfg']['dataset'].replace('/', '_')}.json"
        json_path = output_path / json_filename
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

    # Save all results in a single JSON file
    with open(output_path / "all_eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # Convert to DataFrame and save as CSV
    df = pd.json_normalize(eval_results)  # type: ignore
    df.to_csv(output_path / "all_eval_results.csv", index=False)

    return {
        "individual_jsons": list(output_path.glob("*.json")),
        "combined_json": output_path / "all_eval_results.json",
        "csv": output_path / "all_eval_results.csv",
    }


if __name__ == "__main__":
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
        default=["Skylion007/openwebtext"],
        help="Datasets to evaluate on, such as 'Skylion007/openwebtext' or 'lighteval/MATH'.",
    )
    arg_parser.add_argument(
        "--ctx_lens",
        nargs="+",
        default=[128],
        help="Context lengths to evaluate on.",
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results",
    )

    args = arg_parser.parse_args()

    eval_results = run_evaluations(args)
    output_files = process_results(eval_results, args.output_dir)

    print("Evaluation complete. Output files:")
    print(f"Individual JSONs: {len(output_files['individual_jsons'])}")  # type: ignore
    print(f"Combined JSON: {output_files['combined_json']}")
    print(f"CSV: {output_files['csv']}")
