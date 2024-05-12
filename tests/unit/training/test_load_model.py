from mamba_lens import HookedMamba
from transformer_lens import HookedTransformer

from sae_lens.training.load_model import load_model


def test_load_model_works_with_mamba():
    model = load_model(
        model_class_name="HookedMamba",
        model_name="state-spaces/mamba-370m",
        device="cpu",
    )
    assert model is not None
    assert isinstance(model, HookedMamba)


def test_load_model_works_without_model_kwargs():
    model = load_model(
        model_class_name="HookedTransformer",
        model_name="pythia-14m",
        device="cpu",
    )
    assert model is not None
    assert isinstance(model, HookedTransformer)
    assert model.cfg.checkpoint_index is None


def test_load_model_works_with_model_kwargs():
    model = load_model(
        model_class_name="HookedTransformer",
        model_name="pythia-14m",
        device="cpu",
        model_from_pretrained_kwargs={"checkpoint_index": 0},
    )
    assert model is not None
    assert isinstance(model, HookedTransformer)
    assert model.cfg.checkpoint_index == 0
