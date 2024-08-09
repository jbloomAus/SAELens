from typing import Any, cast

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


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
            print("MODEL LOADING:")
            print("Setting model device to cuda for d_devices")
            print(f"Will use cuda:0 to cuda:{n_devices-1}")
            device = "cuda"
            print("-------------")

    if model_class_name == "HookedTransformer":
        return HookedTransformer.from_pretrained(
            model_name=model_name, device=device, **model_from_pretrained_kwargs
        )
    elif model_class_name == "HookedMamba":
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
    elif model_class_name == "AutoModelForCausalLM":
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_from_pretrained_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return HookedProxyLM(hf_model, tokenizer)

    else:  # pragma: no cover
        raise ValueError(f"Unknown model class: {model_class_name}")


class HookedProxyLM(HookedRootModule):
    tokenizer: PreTrainedTokenizerBase
    model: torch.nn.Module

    def __init__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizerBase):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
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

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)


def get_hook_fn(hook_point: HookPoint):

    def hook_fn(module: Any, input: Any, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return hook_point(output)
        elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            return (hook_point(output[0]), *output[1:])
        else:
            # if this isn't a tensor, just skip the hook entirely as this will break otherwise
            return output

    return hook_fn
