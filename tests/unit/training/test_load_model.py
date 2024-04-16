from mamba_lens import HookedMamba

from sae_lens.training.load_model import load_model


def test_load_model_works_with_mamba():
    model = load_model(
        model_class_name="HookedMamba",
        model_name="state-spaces/mamba-370m",
        device="cpu",
    )
    assert model is not None
    assert isinstance(model, HookedMamba)
