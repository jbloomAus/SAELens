from typing import Any, cast

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule


def load_model(
    model_class_name: str, model_name: str, device: str | torch.device | None = None
) -> HookedRootModule:
    if model_class_name == "HookedTransformer":
        return HookedTransformer.from_pretrained(model_name=model_name, device=device)
    elif model_class_name == "HookedMamba":
        try:
            from mamba_lens import HookedMamba
        except ImportError:
            raise ValueError(
                "mamba-lens must be installed to work with mamba models. This can be added with `pip install sae-lens[mamba]`"
            )
        # HookedMamba has incorrect typing information, so we need to cast the type here
        return cast(
            HookedRootModule,
            HookedMamba.from_pretrained(model_name, device=cast(Any, device)),
        )
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")
