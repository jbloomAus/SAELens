# import pandas as pd
# import plotly.express as px
# import numpy as np
import pytest
import torch

from sae_lens import SAE, ActivationsStore
from sae_lens.analysis.neuronpedia_integration import open_neuronpedia_feature_dashboard
from sae_lens.evals import all_loadable_saes, get_eval_everything_config, run_evals
from sae_lens.toolkit.pretrained_sae_loaders import (
    SAEConfigLoadOptions,
    get_sae_config_from_hf,
)
from tests.unit.helpers import load_model_cached

# from sae_lens.evals import run_evals


# from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)

example_text = """
Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say
that they were perfectly normal, thank you very much. They were the last
people you'd expect to be involved in anything strange or mysterious,
because they just didn't hold with such nonsense.
"""


def test_get_sae_config():
    repo_id = "jbloom/GPT2-Small-SAEs-Reformatted"
    cfg = get_sae_config_from_hf(
        repo_id=repo_id,
        folder_name="blocks.0.hook_resid_pre",
        options=SAEConfigLoadOptions(),
    )
    assert cfg is not None


@pytest.mark.parametrize(
    "release, sae_name, expected_var_explained, expected_l0", all_loadable_saes()
)
def test_loading_pretrained_saes(
    release: str, sae_name: str, expected_var_explained: float, expected_l0: float
):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    sae, _, _ = SAE.from_pretrained(release, sae_name, device=device)
    assert isinstance(sae, SAE)


@pytest.mark.parametrize(
    "release, sae_name, expected_var_explained, expected_l0", all_loadable_saes()
)
def test_loading_pretrained_saes_open_neuronpedia(
    release: str, sae_name: str, expected_var_explained: float, expected_l0: float
):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    sae, _, _ = SAE.from_pretrained(release, sae_name, device=device)
    assert isinstance(sae, SAE)

    open_neuronpedia_feature_dashboard(sae, 0)


@pytest.mark.parametrize(
    "release, sae_name, expected_var_explained, expected_l0", all_loadable_saes()
)
def test_loading_pretrained_saes_do_forward_pass(
    release: str, sae_name: str, expected_var_explained: float, expected_l0: float
):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    sae, _, _ = SAE.from_pretrained(release, sae_name, device=device)
    assert isinstance(sae, SAE)

    # from transformer_lens import HookedTransformer
    # model = HookedTransformer.from_pretrained("gemma-2-9b")
    # sae_in = model.run_with_cache("test test test")[1][sae.cfg.hook_name]

    if "hook_z" in sae.cfg.hook_name:
        # check that reshaping works as intended
        from transformer_lens.loading_from_pretrained import get_pretrained_model_config

        model_cfg = get_pretrained_model_config(sae.cfg.model_name)
        sae_in = torch.randn(1, 4, model_cfg.n_heads, model_cfg.d_head).to(device)
        sae_out = sae(sae_in)
        assert sae_out.shape == sae_in.shape

    sae.turn_off_forward_pass_hook_z_reshaping()  # just in case
    sae_in = torch.randn(1, sae.cfg.d_in).to(device)
    sae_out = sae(sae_in)
    assert sae_out.shape == sae_in.shape

    assert True  # If we get here, we're good


@pytest.mark.parametrize(
    "release, sae_name, expected_var_explained, expected_l0", all_loadable_saes()
)
def test_eval_all_loadable_saes(
    release: str, sae_name: str, expected_var_explained: float, expected_l0: float
):
    """This test is currently only passing for a subset of SAEs because we need to
    have the normalization factors on hand to normalize the activations. We should
    really fold these into SAEs so this test is easy to do."""

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    sae, _, _ = SAE.from_pretrained(release, sae_name, device=device)
    sae.fold_W_dec_norm()

    model = load_model_cached(sae.cfg.model_name)
    model.to(device)

    activation_store = ActivationsStore.from_sae(
        model=model,
        sae=sae,
        streaming=True,
        # fairly conservative parameters here so can use same for larger
        # models without running out of memory.
        store_batch_size_prompts=8,
        train_batch_size_tokens=4096,
        n_batches_in_buffer=4,
        device=device,
    )

    eval_config = get_eval_everything_config(
        batch_size_prompts=8,
        n_eval_reconstruction_batches=10,
        n_eval_sparsity_variance_batches=10,
    )

    metrics = run_evals(
        sae=sae,
        activation_store=activation_store,
        model=model,
        eval_config=eval_config,
        ignore_tokens={
            model.tokenizer.pad_token_id,  # type: ignore
            model.tokenizer.eos_token_id,  # type: ignore
            model.tokenizer.bos_token_id,  # type: ignore
        },
    )

    assert pytest.approx(metrics["l0"], abs=5) == expected_l0
    assert (
        pytest.approx(metrics["explained_variance"], abs=0.1) == expected_var_explained
    )
