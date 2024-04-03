"""
Took the LR scheduler from my previous work: https://github.com/jbloomAus/DecisionTransformerInterpretability/blob/ee55df35cdb92e81d689c72fb9dd5a7252893363/src/decision_transformer/utils.py#L425
"""

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


#  Constant
#  Cosine Annealing with Warmup
#  Cosine Annealing with Warmup / Restarts
#  No default values specified so the type-checker can verify we don't forget any arguments.
def get_scheduler(
    scheduler_name: str,
    optimizer: optim.Optimizer,
    training_steps: int,
    lr: float,
    warm_up_steps: int,
    decay_steps: int,
    lr_end: float,
    num_cycles: int,
) -> lr_scheduler.LRScheduler:
    """
    Loosely based on this, seemed simpler write this than import
    transformers: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules

    Args:
        scheduler_name (str): Name of the scheduler to use, one of "constant", "cosineannealing", "cosineannealingwarmrestarts"
        optimizer (optim.Optimizer): Optimizer to use
        training_steps (int): Total number of training steps
        warm_up_steps (int, optional): Number of linear warm up steps. Defaults to 0.
        decay_steps (int, optional): Number of linear decay steps to 0. Defaults to 0.
        num_cycles (int, optional): Number of cycles for cosine annealing with warm restarts. Defaults to 1.
        lr_end (float, optional): Final learning rate multiplier before decay. Defaults to 0.0.
    """
    base_scheduler_steps = training_steps - warm_up_steps - decay_steps
    norm_scheduler_name = scheduler_name.lower()
    main_scheduler = _get_main_scheduler(
        norm_scheduler_name,
        optimizer,
        steps=base_scheduler_steps,
        lr_end=lr_end,
        num_cycles=num_cycles,
    )
    if norm_scheduler_name == "constant":
        # constant scheduler ignores lr_end, so decay needs to start at lr
        lr_end = lr
    schedulers: list[lr_scheduler.LRScheduler] = []
    milestones: list[int] = []
    if warm_up_steps > 0:
        schedulers.append(
            lr_scheduler.LinearLR(
                optimizer,
                start_factor=1 / warm_up_steps,
                end_factor=1.0,
                total_iters=warm_up_steps - 1,
            ),
        )
        milestones.append(warm_up_steps)
    schedulers.append(main_scheduler)
    if decay_steps > 0:
        if lr_end == 0.0:
            raise ValueError(
                "Cannot have decay_steps with lr_end=0.0, this would decay from 0 to 0 and be a waste."
            )
        schedulers.append(
            lr_scheduler.LinearLR(
                optimizer,
                start_factor=lr_end / lr,
                end_factor=0.0,
                total_iters=decay_steps,
            ),
        )
        milestones.append(training_steps - decay_steps)
    return lr_scheduler.SequentialLR(
        schedulers=schedulers,
        optimizer=optimizer,
        milestones=milestones,
    )


def _get_main_scheduler(
    scheduler_name: str,
    optimizer: optim.Optimizer,
    steps: int,
    lr_end: float,
    num_cycles: int,
) -> lr_scheduler.LRScheduler:
    if scheduler_name == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)
    elif scheduler_name == "cosineannealing":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr_end)
    elif scheduler_name == "cosineannealingwarmrestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=steps // num_cycles, eta_min=lr_end
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
