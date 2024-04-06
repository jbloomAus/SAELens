from sae_lens.training.sae_group import SAEGroup
from tests.unit.helpers import build_sae_cfg


def test_SAEGroup_initializes_all_permutations_of_list_params():
    cfg = build_sae_cfg(
        d_in=5,
        lr=[0.01, 0.001],
        expansion_factor=[2, 4],
    )
    sae_group = SAEGroup(cfg)
    assert len(sae_group) == 4
    lr_sae_combos = [(ae.cfg.lr, ae.cfg.d_sae) for ae in sae_group]
    assert (0.01, 10) in lr_sae_combos
    assert (0.01, 20) in lr_sae_combos
    assert (0.001, 10) in lr_sae_combos
    assert (0.001, 20) in lr_sae_combos


def test_SAEGroup_replaces_layer_with_actual_layer():
    cfg = build_sae_cfg(
        hook_point="blocks.{layer}.attn.hook_q",
        hook_point_layer=5,
    )
    sae_group = SAEGroup(cfg)
    assert len(sae_group) == 1
    assert sae_group.autoencoders[0].cfg.hook_point == "blocks.5.attn.hook_q"


def test_SAEGroup_train_and_eval():
    cfg = build_sae_cfg(
        lr=[0.01, 0.001],
        expansion_factor=[2, 4],
    )
    sae_group = SAEGroup(cfg)
    sae_group.train()
    for sae in sae_group:
        assert sae.training is True
    sae_group.eval()
    for sae in sae_group:
        assert sae.training is False
    sae_group.train()
    for sae in sae_group:
        assert sae.training is True
