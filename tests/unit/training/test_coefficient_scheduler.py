from sae_lens.training.optim import CoefficientScheduler
from tests.unit.helpers import build_sae_cfg


def test_coefficient_scheduler_initialization():
    cfg = build_sae_cfg(
        sparsity_coefficient=5,
        training_tokens=100 * 4,  # train batch size (so 100 steps)
        coefficient_warm_up_steps=10,
    )

    coefficient_scheduler = CoefficientScheduler(
        coefficient_warm_up_steps=cfg.coefficient_warm_up_steps,  # type: ignore
        total_steps=cfg.training_tokens // cfg.train_batch_size_tokens,
        final_sparsity_coefficient=cfg.sparsity_coefficient,
    )

    assert cfg.sparsity_coefficient == 5
    assert (
        coefficient_scheduler.current_sparsity_coefficient == 0
    )  # the l1 coefficient is set to 0, to begin warm up.

    # over 10 steps, we should get to the final value of 5
    for i in range(10):
        coefficient_scheduler.step()
        assert coefficient_scheduler.current_sparsity_coefficient == 5 * (1 + i) / 10


def test_coefficient_scheduler_initialization_no_warmup():
    cfg = build_sae_cfg(
        sparsity_coefficient=5,
        training_tokens=100 * 4,  # train batch size (so 100 steps)
        coefficient_warm_up_steps=10,
    )

    coefficient_scheduler = CoefficientScheduler(
        coefficient_warm_up_steps=cfg.coefficient_warm_up_steps,  # type: ignore
        total_steps=cfg.training_tokens // cfg.train_batch_size_tokens,
        final_sparsity_coefficient=cfg.sparsity_coefficient,
    )

    assert cfg.sparsity_coefficient == 5
    assert (
        coefficient_scheduler.current_sparsity_coefficient == 5
    )  # the l1 coefficient is set to 0, to begin warm up.

    # over 10 steps, we should get to the final value of 5
    for _ in range(10):
        coefficient_scheduler.step()
        assert coefficient_scheduler.current_sparsity_coefficient == coefficient_scheduler.final_sparsity_coefficient
