from sae_lens.training.optim import L1Scheduler
from tests.unit.helpers import build_sae_cfg


def test_l1_scheduler_initialization():
    cfg = build_sae_cfg(
        l1_coefficient=5,
        training_tokens=100 * 4,  # train batch size (so 100 steps)
        l1_warm_up_steps=10,
    )

    l1_scheduler = L1Scheduler(
        l1_warm_up_steps=cfg.l1_warm_up_steps,  # type: ignore
        total_steps=cfg.training_tokens // cfg.train_batch_size_tokens,
        final_l1_coefficient=cfg.l1_coefficient,
    )

    assert cfg.l1_coefficient == 5
    assert (
        l1_scheduler.current_l1_coefficient == 0
    )  # the l1 coefficient is set to 0, to begin warm up.

    # over 10 steps, we should get to the final value of 5
    for i in range(10):
        l1_scheduler.step()
        assert l1_scheduler.current_l1_coefficient == 5 * (1 + i) / 10


def test_l1_scheduler_initialization_no_warmup():
    cfg = build_sae_cfg(
        l1_coefficient=5,
        training_tokens=100 * 4,  # train batch size (so 100 steps)
        l1_warm_up_steps=0,
    )

    l1_scheduler = L1Scheduler(
        l1_warm_up_steps=cfg.l1_warm_up_steps,  # type: ignore
        total_steps=cfg.training_tokens // cfg.train_batch_size_tokens,
        final_l1_coefficient=cfg.l1_coefficient,
    )

    assert cfg.l1_coefficient == 5
    assert (
        l1_scheduler.current_l1_coefficient == 5
    )  # the l1 coefficient is set to 0, to begin warm up.

    # over 10 steps, we should get to the final value of 5
    for _ in range(10):
        l1_scheduler.step()
        assert l1_scheduler.current_l1_coefficient == l1_scheduler.final_l1_coefficient
