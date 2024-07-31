"""
Took the LR scheduler from my previous work: https://github.com/jbloomAus/DecisionTransformerInterpretability/blob/ee55df35cdb92e81d689c72fb9dd5a7252893363/src/decision_transformer/utils.py#L425
"""

from typing import Any

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


#  Constant
#  Cosine Annealing with Warmup
#  Cosine Annealing with Warmup / Restarts
#  No default values specified so the type-checker can verify we don't forget any arguments.
def get_lr_scheduler(
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
    main_scheduler = _get_main_lr_scheduler(
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


def _get_main_lr_scheduler(
    scheduler_name: str,
    optimizer: optim.Optimizer,
    steps: int,
    lr_end: float,
    num_cycles: int,
) -> lr_scheduler.LRScheduler:
    if scheduler_name == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)
    elif scheduler_name == "cosineannealing":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr_end)  # type: ignore
    elif scheduler_name == "cosineannealingwarmrestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=steps // num_cycles, eta_min=lr_end  # type: ignore
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


class L1Scheduler:

    def __init__(
        self,
        l1_warm_up_steps: float,
        total_steps: int,
        final_l1_coefficient: float,
    ):

        self.l1_warmup_steps = l1_warm_up_steps
        # assume using warm-up
        if self.l1_warmup_steps != 0:
            self.current_l1_coefficient = 0.0
        else:
            self.current_l1_coefficient = final_l1_coefficient

        self.final_l1_coefficient = final_l1_coefficient

        self.current_step = 0
        self.total_steps = total_steps
        assert isinstance(self.final_l1_coefficient, float | int)

    def __repr__(self) -> str:
        return (
            f"L1Scheduler(final_l1_value={self.final_l1_coefficient}, "
            f"l1_warmup_steps={self.l1_warmup_steps}, "
            f"total_steps={self.total_steps})"
        )

    def step(self):
        """
        Updates the l1 coefficient of the sparse autoencoder.
        """
        step = self.current_step
        if step < self.l1_warmup_steps:
            self.current_l1_coefficient = self.final_l1_coefficient * (
                (1 + step) / self.l1_warmup_steps
            )  # type: ignore
        else:
            self.current_l1_coefficient = self.final_l1_coefficient  # type: ignore

        self.current_step += 1

    def state_dict(self):
        """State dict for serializing as part of an SAETrainContext."""
        return {
            "l1_warmup_steps": self.l1_warmup_steps,
            "total_steps": self.total_steps,
            "current_l1_coefficient": self.current_l1_coefficient,
            "final_l1_coefficient": self.final_l1_coefficient,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Loads all state apart from attached SAE."""
        for k in state_dict:
            setattr(self, k, state_dict[k])
