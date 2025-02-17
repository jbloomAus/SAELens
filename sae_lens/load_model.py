import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Callable, Literal, cast

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens.HookedTransformer import Loss, Output
from transformer_lens.utils import (
    USE_DEFAULT_VALUE,
    get_tokens_with_bos_removed,
    lm_cross_entropy_loss,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from sae_lens import logger


def load_model(
    model_class_name: str,
    model_name: str,
    device: str | torch.device | None = None,
    model_from_pretrained_kwargs: dict[str, Any] | None = None,
) -> HookedRootModule:
    model_from_pretrained_kwargs = model_from_pretrained_kwargs or {}

    if "n_devices" in model_from_pretrained_kwargs:
        n_devices = model_from_pretrained_kwargs["n_devices"]
        if n_devices > 1:
            logger.info("MODEL LOADING:")
            logger.info("Setting model device to cuda for d_devices")
            logger.info(f"Will use cuda:0 to cuda:{n_devices - 1}")
            device = "cuda"
            logger.info("-------------")

    if model_class_name == "HookedTransformer":
        return HookedTransformer.from_pretrained_no_processing(
            model_name=model_name, device=device, **model_from_pretrained_kwargs
        )
    if model_class_name == "HookedMamba":
        try:
            from mamba_lens import HookedMamba
        except ImportError:  # pragma: no cover
            raise ValueError(
                "mamba-lens must be installed to work with mamba models. This can be added with `pip install sae-lens[mamba]`"
            )
        # HookedMamba has incorrect typing information, so we need to cast the type here
        return cast(
            HookedRootModule,
            HookedMamba.from_pretrained(
                model_name, device=cast(Any, device), **model_from_pretrained_kwargs
            ),
        )
    if model_class_name == "AutoModelForCausalLM":
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_from_pretrained_kwargs
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return HookedProxyLM(hf_model, tokenizer)

    # pragma: no cover
    raise ValueError(f"Unknown model class: {model_class_name}")


class HookedProxyLM(HookedRootModule):
    """
    A HookedRootModule that wraps a Huggingface AutoModelForCausalLM.
    """

    tokenizer: PreTrainedTokenizerBase
    model: torch.nn.Module

    def __init__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.decoder_block_matcher = guess_decoder_block_matcher(model)
        self.setup()

    # copied and modified from base HookedRootModule
    def setup(self):
        self.mod_dict = {}
        self.hook_dict: dict[str, HookPoint] = {}
        for name, module in self.model.named_modules():
            if name == "":
                continue

            hook_point = HookPoint()
            hook_point.name = name  # type: ignore

            module.register_forward_hook(get_hook_fn(hook_point))

            self.hook_dict[name] = hook_point
            self.mod_dict[name] = hook_point

    def forward(
        self,
        tokens: torch.Tensor,
        return_type: Literal["both", "logits"] = "logits",
        loss_per_token: bool = False,
        stop_at_layer: int | None = None,
        **kwargs: Any,
    ) -> Output | Loss:
        # This is just what's needed for evals, not everything that HookedTransformer has
        if return_type not in ("both", "logits"):
            raise NotImplementedError(
                "Only return_type supported is 'both' or 'logits' to match what's in evals.py and ActivationsStore"
            )

        stop_hook = None
        if stop_at_layer is not None:
            if return_type != "logits":
                raise NotImplementedError(
                    "stop_at_layer is not supported for return_type='both'"
                )
            if self.decoder_block_matcher is not None:
                layer_name = self.decoder_block_matcher.format(num=stop_at_layer)
                layer = get_layer_by_name(self.model, layer_name)
                stop_hook = layer.register_forward_hook(stop_hook_fn)

        try:
            output = self.model(tokens)
            logits = _extract_logits_from_output(output)
        except StopForward:
            # If we stop early, we don't care about the return output
            return None  # type: ignore
        finally:
            if stop_hook is not None:
                stop_hook.remove()

        if return_type == "logits":
            return logits

        if tokens.device != logits.device:
            tokens = tokens.to(logits.device)
        loss = lm_cross_entropy_loss(logits, tokens, per_token=loss_per_token)
        return Output(logits, loss)

    def to_tokens(
        self,
        input: str | list[str],
        prepend_bos: bool | None = USE_DEFAULT_VALUE,
        padding_side: Literal["left", "right"] | None = USE_DEFAULT_VALUE,
        move_to_device: bool = True,
        truncate: bool = True,
    ) -> torch.Tensor:
        # Hackily modified version of HookedTransformer.to_tokens to work with ActivationsStore
        # Assumes that prepend_bos is always False, move_to_device is always False, and truncate is always False
        # copied from HookedTransformer.to_tokens

        if prepend_bos is not False:
            raise ValueError(
                "Only works with prepend_bos=False, to match ActivationsStore usage"
            )

        if padding_side is not None:
            raise ValueError(
                "Only works with padding_side=None, to match ActivationsStore usage"
            )

        if truncate is not False:
            raise ValueError(
                "Only works with truncate=False, to match ActivationsStore usage"
            )

        if move_to_device is not False:
            raise ValueError(
                "Only works with move_to_device=False, to match ActivationsStore usage"
            )

        tokens = self.tokenizer(
            input,
            return_tensors="pt",
            truncation=False,
            max_length=None,
        )["input_ids"]

        # We don't want to prepend bos but the tokenizer does it automatically, so we remove it manually
        if hasattr(self.tokenizer, "add_bos_token") and self.tokenizer.add_bos_token:  # type: ignore
            tokens = get_tokens_with_bos_removed(self.tokenizer, tokens)
        return tokens  # type: ignore


def _extract_logits_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    raise ValueError(f"Unknown output type: {type(output)}")


def get_hook_fn(hook_point: HookPoint):
    def hook_fn(module: Any, input: Any, output: Any) -> Any:  # noqa: ARG001
        if isinstance(output, torch.Tensor):
            return hook_point(output)
        if isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            return (hook_point(output[0]), *output[1:])
        # if this isn't a tensor, just skip the hook entirely as this will break otherwise
        return output

    return hook_fn


class StopForward(Exception):
    pass


def stop_hook_fn(module: Any, input: Any, output: Any) -> Any:  # noqa: ARG001
    raise StopForward()


LAYER_GUESS_RE = r"^([^\d]+)\.([\d]+)(.*)$"


def get_layer_by_name(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    return dict(model.named_modules())[layer_name]


# broken into a separate function for easier testing
def _guess_block_matcher_from_layers(
    layers: Iterable[str], filter: Callable[[str], bool] | None = None
) -> str | None:
    """
    Guess the block matcher for a given model. The block matcher is a template string like "model.layers.{num}"
    that can be used to match the hidden layers of the model.
    """
    counts_by_guess: dict[str, int] = defaultdict(int)

    for layer in layers:
        if re.match(LAYER_GUESS_RE, layer):
            guess = re.sub(LAYER_GUESS_RE, r"\1.{num}\3", layer)
            if filter is None or filter(guess):
                counts_by_guess[guess] += 1
    if len(counts_by_guess) == 0:
        return None

    # score is higher for guesses that match more often, are and shorter in length
    guess_scores = [
        (guess, count + 1 / len(guess)) for guess, count in counts_by_guess.items()
    ]
    return max(guess_scores, key=lambda x: x[1])[0]


def guess_decoder_block_matcher(model: torch.nn.Module) -> str | None:
    """
    Guess the hidden layer matcher for a given model.
    """
    return _guess_block_matcher_from_layers(dict(model.named_modules()).keys())
