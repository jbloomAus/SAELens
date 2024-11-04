from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, NamedTuple

import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from sae_lens import SAE, HookedSAETransformer


def list_flatten(nested_list):
    return [x for y in nested_list for x in y]


# A very handy function Neel wrote to get context around a feature activation
def make_token_df(tokens, len_prefix=5, len_suffix=3, model=None):
    str_tokens = [model.to_str_tokens(t) for t in tokens]
    unique_token = [
        [f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens
    ]

    context = []
    prompt = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p - len_prefix) : p])
            if p == tokens.shape[1] - 1:
                suffix = ""
            else:
                suffix = "".join(
                    str_tokens[b][p + 1 : min(tokens.shape[1] - 1, p + 1 + len_suffix)]
                )
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            prompt.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(
        dict(
            str_tokens=list_flatten(str_tokens),
            unique_token=list_flatten(unique_token),
            context=context,
            prompt=prompt,
            pos=pos,
            label=label,
        )
    )


# Display token probabilities
def get_prediction_df(
    prompt: str,
    model: HookedTransformer,
    prepend_bos: bool = True,
    top_k: int = 10,
) -> pd.DataFrame:
    """Test model predictions for a prompt.
    Returns results as a pandas DataFrame containing token predictions and their metrics.
    """

    prompt_tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    logits = model(prompt_tokens)

    if logits.shape[0] == 1:
        logits = logits.squeeze(0)

    probs = logits.softmax(dim=-1)

    # Get predictions for last token
    token_probs = probs[-1]
    sorted_token_probs, sorted_token_values = token_probs.sort(descending=True)

    results = []
    # Add top k predictions
    for i in range(top_k):
        token_id = sorted_token_values[i]
        results.append(
            {
                "token": model.to_string(token_id),
                "rank": i,
                "logit": logits[-1, token_id].item(),
                "probability": sorted_token_probs[i].item()
                * 100,  # Convert to percentage
            }
        )

    df = pd.DataFrame(results)
    df["token"] = df["token"].astype(str)
    df.sort_values(by="probability", ascending=False, inplace=True)

    return df


class SaeReconstructionCache(NamedTuple):
    sae_in: torch.Tensor
    feature_acts: torch.Tensor
    sae_out: torch.Tensor
    sae_error: torch.Tensor


def track_grad(tensor: torch.Tensor) -> None:
    """wrapper around requires_grad and retain_grad"""
    tensor.requires_grad_(True)
    tensor.retain_grad()


@dataclass
class ApplySaesAndRunOutput:
    model_output: torch.Tensor
    model_activations: dict[str, torch.Tensor]
    sae_activations: dict[str, SaeReconstructionCache]

    def zero_grad(self) -> None:
        """Helper to zero grad all tensors in this object."""
        self.model_output.grad = None
        for act in self.model_activations.values():
            act.grad = None
        for cache in self.sae_activations.values():
            cache.sae_in.grad = None
            cache.feature_acts.grad = None
            cache.sae_out.grad = None
            cache.sae_error.grad = None


def apply_saes_and_run(
    model: HookedTransformer,
    saes: dict[str, SAE],
    input: Any,
    include_error_term: bool = True,
    track_model_hooks: list[str] | None = None,
    return_type: Literal["logits", "loss"] = "logits",
    track_grads: bool = False,
) -> ApplySaesAndRunOutput:
    """
    Apply the SAEs to the model at the specific hook points, and run the model.
    By default, this will include a SAE error term which guarantees that the SAE
    will not affect model output. This function is designed to work correctly with
    backprop as well, so it can be used for gradient-based feature attribution.

    Args:
        model: the model to run
        saes: the SAEs to apply
        input: the input to the model
        include_error_term: whether to include the SAE error term to ensure the SAE doesn't affect model output. Default True
        track_model_hooks: a list of hook points to record the activations and gradients. Default None
        return_type: this is passed to the model.run_with_hooks function. Default "logits"
        track_grads: whether to track gradients. Default False
    """

    fwd_hooks = []
    bwd_hooks = []

    sae_activations: dict[str, SaeReconstructionCache] = {}
    model_activations: dict[str, torch.Tensor] = {}

    # this hook just track the SAE input, output, features, and error. If `track_grads=True`, it also ensures
    # that requires_grad is set to True and retain_grad is called for intermediate values.
    def reconstruction_hook(
        sae_in: torch.Tensor, hook: HookPoint, hook_point: str
    ):  # noqa: ARG001
        sae = saes[hook_point]
        feature_acts = sae.encode(sae_in)
        sae_out = sae.decode(feature_acts)
        sae_error = (sae_in - sae_out).detach().clone()
        if track_grads:
            track_grad(sae_error)
            track_grad(sae_out)
            track_grad(feature_acts)
            track_grad(sae_in)
        sae_activations[hook_point] = SaeReconstructionCache(
            sae_in=sae_in,
            feature_acts=feature_acts,
            sae_out=sae_out,
            sae_error=sae_error,
        )

        if include_error_term:
            return sae_out + sae_error
        return sae_out

    def sae_bwd_hook(output_grads: torch.Tensor, hook: HookPoint):  # noqa: ARG001
        # this just passes the output grads to the input, so the SAE gets the same grads despite the error term hackery
        return (output_grads,)

    # this hook just records model activations, and ensures that intermediate activations have gradient tracking turned on if needed
    def tracking_hook(
        hook_input: torch.Tensor, hook: HookPoint, hook_point: str
    ):  # noqa: ARG001
        model_activations[hook_point] = hook_input
        if track_grads:
            track_grad(hook_input)
        return hook_input

    for hook_point in saes.keys():
        fwd_hooks.append(
            (hook_point, partial(reconstruction_hook, hook_point=hook_point))
        )
        bwd_hooks.append((hook_point, sae_bwd_hook))
    for hook_point in track_model_hooks or []:
        fwd_hooks.append((hook_point, partial(tracking_hook, hook_point=hook_point)))

    # now, just run the model while applying the hooks
    with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
        model_output = model(input, return_type=return_type)

    return ApplySaesAndRunOutput(
        model_output=model_output,
        model_activations=model_activations,
        sae_activations=sae_activations,
    )


from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, NamedTuple

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from sae_lens import SAE

EPS = 1e-8

torch.set_grad_enabled(True)


@dataclass
class AttributionGrads:
    metric: torch.Tensor
    model_output: torch.Tensor
    model_activations: dict[str, torch.Tensor]
    sae_activations: dict[str, SaeReconstructionCache]


@dataclass
class Attribution:
    model_attributions: dict[str, torch.Tensor]
    model_activations: dict[str, torch.Tensor]
    model_grads: dict[str, torch.Tensor]
    sae_feature_attributions: dict[str, torch.Tensor]
    sae_feature_activations: dict[str, torch.Tensor]
    sae_feature_grads: dict[str, torch.Tensor]
    sae_errors_attribution_proportion: dict[str, float]


def calculate_attribution_grads(
    model: HookedSAETransformer,
    prompt: str,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    track_hook_points: list[str] | None = None,
    include_saes: dict[str, SAE] | None = None,
    return_logits: bool = True,
    include_error_term: bool = True,
) -> AttributionGrads:
    """
    Wrapper around apply_saes_and_run that calculates gradients wrt to the metric_fn.
    Tracks grads for both SAE feature and model neurons, and returns them in a structured format.
    """
    output = apply_saes_and_run(
        model,
        saes=include_saes or {},
        input=prompt,
        return_type="logits" if return_logits else "loss",
        track_model_hooks=track_hook_points,
        include_error_term=include_error_term,
        track_grads=True,
    )
    metric = metric_fn(output.model_output)
    output.zero_grad()
    metric.backward()
    return AttributionGrads(
        metric=metric,
        model_output=output.model_output,
        model_activations=output.model_activations,
        sae_activations=output.sae_activations,
    )


def calculate_feature_attribution(
    model: HookedSAETransformer,
    input: Any,
    metric_fn: Callable[[torch.Tensor], torch.Tensor],
    track_hook_points: list[str] | None = None,
    include_saes: dict[str, SAE] | None = None,
    return_logits: bool = True,
    include_error_term: bool = True,
) -> Attribution:
    """
    Calculate feature attribution for SAE features and model neurons following
    the procedure in https://transformer-circuits.pub/2024/march-update/index.html#feature-heads.
    This include the SAE error term by default, so inserting the SAE into the calculation is
    guaranteed to not affect the model output. This can be disabled by setting `include_error_term=False`.

    Args:
        model: The model to calculate feature attribution for.
        input: The input to the model.
        metric_fn: A function that takes the model output and returns a scalar metric.
        track_hook_points: A list of model hook points to track activations for, if desired
        include_saes: A dictionary of SAEs to include in the calculation. The key is the hook point to apply the SAE to.
        return_logits: Whether to return the model logits or loss. This is passed to TLens, so should match whatever the metric_fn expects (probably logits)
        include_error_term: Whether to include the SAE error term in the calculation. This is recommended, as it ensures that the SAE will not affecting the model output.
    """
    # first, calculate gradients wrt to the metric_fn.
    # these will be multiplied with the activation values to get the attributions
    outputs_with_grads = calculate_attribution_grads(
        model,
        input,
        metric_fn,
        track_hook_points,
        include_saes=include_saes,
        return_logits=return_logits,
        include_error_term=include_error_term,
    )
    model_attributions = {}
    model_activations = {}
    model_grads = {}
    sae_feature_attributions = {}
    sae_feature_activations = {}
    sae_feature_grads = {}
    sae_error_proportions = {}
    # this code is long, but all it's doing is multiplying the grads by the activations
    # and recording grads, acts, and attributions in dictionaries to return to the user
    with torch.no_grad():
        for name, act in outputs_with_grads.model_activations.items():
            assert act.grad is not None
            raw_activation = act.detach().clone()
            model_attributions[name] = (act.grad * raw_activation).detach().clone()
            model_activations[name] = raw_activation
            model_grads[name] = act.grad.detach().clone()
        for name, act in outputs_with_grads.sae_activations.items():
            assert act.feature_acts.grad is not None
            assert act.sae_out.grad is not None
            raw_activation = act.feature_acts.detach().clone()
            sae_feature_attributions[name] = (
                (act.feature_acts.grad * raw_activation).detach().clone()
            )
            sae_feature_activations[name] = raw_activation
            sae_feature_grads[name] = act.feature_acts.grad.detach().clone()
            if include_error_term:
                assert act.sae_error.grad is not None
                error_grad_norm = act.sae_error.grad.norm().item()
            else:
                error_grad_norm = 0
            sae_out_norm = act.sae_out.grad.norm().item()
            sae_error_proportions[name] = error_grad_norm / (
                sae_out_norm + error_grad_norm + EPS
            )
        return Attribution(
            model_attributions=model_attributions,
            model_activations=model_activations,
            model_grads=model_grads,
            sae_feature_attributions=sae_feature_attributions,
            sae_feature_activations=sae_feature_activations,
            sae_feature_grads=sae_feature_grads,
            sae_errors_attribution_proportion=sae_error_proportions,
        )


def convert_sparse_feature_to_long_df(sparse_tensor: torch.Tensor) -> pd.DataFrame:
    """
    Convert a sparse tensor to a long format pandas DataFrame.
    """
    df = pd.DataFrame(sparse_tensor.detach().cpu().numpy())
    df_long = df.melt(ignore_index=False, var_name="column", value_name="value")
    df_long.columns = ["feature", "attribution"]
    df_long_nonzero = df_long[df_long["attribution"] != 0]
    df_long_nonzero = df_long_nonzero.reset_index().rename(
        columns={"index": "position"}
    )
    return df_long_nonzero
