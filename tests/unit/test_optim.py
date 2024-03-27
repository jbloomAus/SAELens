import pytest
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LRScheduler,
)

from sae_training.optim import get_scheduler


@pytest.fixture
def optimizer():
    return Adam([torch.tensor(1.0)], lr=0.1)


def step_times(num: int, optimizer: Adam, scheduler: LRScheduler):
    for _ in range(num):
        step(optimizer, scheduler)


def step(optimizer: Adam, scheduler: LRScheduler):
    optimizer.step()
    scheduler.step()


@pytest.mark.parametrize(
    "scheduler_name",
    [
        "linearwarmupdecay",
        "cosineannealing",
        "cosineannealingwarmup",
        "cosineannealingwarmrestarts",
    ],
)
def test_get_scheduler_requires_training_steps(scheduler_name: str, optimizer: Adam):
    with pytest.raises(AssertionError, match="training_steps must be provided"):
        get_scheduler(scheduler_name, optimizer, 10)


def test_get_scheduler_errors_on_uknown_scheduler(optimizer: Adam):
    with pytest.raises(ValueError, match="Unsupported scheduler: unknown"):
        get_scheduler("unknown", optimizer)


def test_get_scheduler_constant(optimizer: Adam):
    scheduler = get_scheduler("constant", optimizer)
    assert scheduler.get_last_lr() == [0.1]
    step_times(3, optimizer, scheduler)
    assert scheduler.get_last_lr() == [0.1]


def test_get_scheduler_constantwithwarmup(optimizer: Adam):
    scheduler = get_scheduler("constantwithwarmup", optimizer, warm_up_steps=2)
    assert scheduler.get_last_lr() == [0.05]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [0.1]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [0.1]
    step_times(3, optimizer, scheduler)
    assert scheduler.get_last_lr() == [0.1]


def test_get_scheduler_linearwarmupdecay(optimizer: Adam):
    scheduler = get_scheduler(
        "linearwarmupdecay", optimizer, warm_up_steps=2, training_steps=6
    )
    # first, ramp up for 2 steps
    assert scheduler.get_last_lr() == [0.05]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [0.1]
    step(optimizer, scheduler)
    # next, ramp down for 4 steps
    assert scheduler.get_last_lr() == [0.1]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [pytest.approx(0.075)]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [pytest.approx(0.05)]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [pytest.approx(0.025)]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [0.0]
    # NOTE: the LR goes negative if you go beyond the training steps


def test_get_scheduler_cosineannealing(optimizer: Adam):
    scheduler = get_scheduler(
        "cosineannealing", optimizer, training_steps=4, lr_end=0.05
    )
    assert isinstance(scheduler, CosineAnnealingLR)
    assert scheduler.T_max == 4
    assert scheduler.eta_min == 0.05


def test_get_scheduler_cosineannealingwarmup():
    # NOTE: if the lr_end is not 0.0, this test will not pass.
    # If eta_min = lr_end * lr, then the test will pass.
    # We should be careful about the difference between our lr_end and eta_min.
    lr_end = 0.0
    optimizer = Adam([torch.tensor(1.0)], lr=0.1)
    scheduler = get_scheduler(
        "cosineannealingwarmup",
        optimizer,
        warm_up_steps=2,
        training_steps=6,
        lr_end=lr_end,
    )
    # first, ramp up for 2 steps
    assert scheduler.get_last_lr() == [0.05]
    step(optimizer, scheduler)
    assert scheduler.get_last_lr() == [0.1]
    step(optimizer, scheduler)

    # From here on, it should match CosineAnnealingLR
    new_optimizer = Adam([torch.tensor(1.0)], lr=0.1)
    cos_scheduler = CosineAnnealingLR(new_optimizer, T_max=4, eta_min=lr_end)

    step(optimizer, scheduler)
    step(new_optimizer, cos_scheduler)
    assert scheduler.get_last_lr() == pytest.approx(cos_scheduler.get_last_lr())
    step(optimizer, scheduler)
    step(new_optimizer, cos_scheduler)
    assert scheduler.get_last_lr() == pytest.approx(cos_scheduler.get_last_lr())
    step(optimizer, scheduler)
    step(new_optimizer, cos_scheduler)
    assert scheduler.get_last_lr() == pytest.approx(cos_scheduler.get_last_lr())


def test_get_scheduler_cosineannealingwarmrestarts(optimizer: Adam):
    scheduler = get_scheduler(
        "cosineannealingwarmrestarts",
        optimizer,
        training_steps=8,
        lr_end=0.05,
        num_cycles=2,
    )
    assert isinstance(scheduler, CosineAnnealingWarmRestarts)
    assert scheduler.T_0 == 4
    assert scheduler.eta_min == 0.05
