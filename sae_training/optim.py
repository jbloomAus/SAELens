"""
Took the LR scheduler from my previous work: https://github.com/jbloomAus/DecisionTransformerInterpretability/blob/ee55df35cdb92e81d689c72fb9dd5a7252893363/src/decision_transformer/utils.py#L425
"""

import math
from typing import Optional

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


#  None
#  Linear Warmup and decay
#  Cosine Annealing with Warmup
#  Cosine Annealing with Warmup / Restarts
def get_scheduler(
    scheduler_name: Optional[str],
    optimizer: optim.Optimizer,
    warm_up_steps: int = 0,
    training_steps: int | None = None,
    num_cycles: int = 1,
    lr_end: float = 0.0,
):
    """
    Loosely based on this, seemed simpler write this than import
    transformers: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules

    Args:
        scheduler_name (Optional[str]): Name of the scheduler to use. If None, returns a constant scheduler
        optimizer (optim.Optimizer): Optimizer to use
        **kwargs: Additional arguments to pass to the scheduler including warm_up_steps,
            training_steps, num_cycles, lr_end.
    """

    def get_warmup_lambda(warm_up_steps: int, training_steps: int):

        def lr_lambda(steps: int):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                return (training_steps - steps) / (training_steps - warm_up_steps)

        return lr_lambda

    # heavily derived from hugging face although copilot helped.
    def get_warmup_cosine_lambda(
        warm_up_steps: int, training_steps: int, lr_end: float
    ):

        def lr_lambda(steps: int):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                progress = (steps - warm_up_steps) / (training_steps - warm_up_steps)
                return lr_end + 0.5 * (1 - lr_end) * (1 + math.cos(math.pi * progress))

        return lr_lambda

    if scheduler_name is None or scheduler_name.lower() == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)
    elif scheduler_name.lower() == "constantwithwarmup":
        return lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda steps: min(1.0, (steps + 1) / warm_up_steps),
        )
    elif scheduler_name.lower() == "linearwarmupdecay":
        assert training_steps is not None, "training_steps must be provided"
        lr_lambda = get_warmup_lambda(warm_up_steps, training_steps)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name.lower() == "cosineannealing":
        assert training_steps is not None, "training_steps must be provided"
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=training_steps, eta_min=lr_end
        )
    elif scheduler_name.lower() == "cosineannealingwarmup":
        assert training_steps is not None, "training_steps must be provided"
        lr_lambda = get_warmup_cosine_lambda(
            warm_up_steps, training_steps, lr_end=lr_end
        )
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_name.lower() == "cosineannealingwarmrestarts":
        assert training_steps is not None, "training_steps must be provided"
        T_0 = training_steps // num_cycles
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, eta_min=lr_end
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
