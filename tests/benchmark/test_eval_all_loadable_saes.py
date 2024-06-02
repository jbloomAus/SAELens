# import pandas as pd
# import plotly.express as px
# import numpy as np
import pytest
import torch
from tqdm import tqdm

from sae_lens.evals import run_evals

# from sae_lens.training.evals import run_evals
from sae_lens.sae import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.training.activations_store import ActivationsStore
from tests.unit.helpers import load_model_cached

# from transformer_lens import HookedTransformer


example_text = """
Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say
that they were perfectly normal, thank you very much. They were the last
people you'd expect to be involved in anything strange or mysterious,
because they just didn't hold with such nonsense.
"""


# @pytest.fixture
def all_loadable_saes() -> list[tuple[str, str]]:
    all_loadable_saes = []
    saes_directory = get_pretrained_saes_directory()
    for release, lookup in tqdm(saes_directory.items()):
        for sae_name in lookup.saes_map.keys():
            all_loadable_saes.append((release, sae_name))

    return all_loadable_saes


@pytest.mark.parametrize("release, sae_name", all_loadable_saes())
def test_loading_pretrained_saes(release: str, sae_name: str):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    sae, _, _ = SAE.from_pretrained(release, sae_name, device=device)
    assert isinstance(sae, SAE)


@pytest.mark.parametrize("release, sae_name", all_loadable_saes())
def test_eval_all_loadable_saes(release: str, sae_name: str):
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

    if sae.cfg.normalize_activations:
        norm_scaling_factor = activation_store.estimate_norm_scaling_factor(
            n_batches_for_norm_estimate=100
        )
        sae.fold_activation_norm_scaling_factor(norm_scaling_factor)
        activation_store.normalize_activations = False

    eval_metrics = run_evals(
        sae=sae,
        activation_store=activation_store,
        model=model,
        n_eval_batches=10,
        eval_batch_size_prompts=8,
    )  #
    eval_metrics = dict(eval_metrics)
    eval_metrics["ce_loss_diff"] = (
        eval_metrics["metrics/ce_loss_with_sae"].item()
        - eval_metrics["metrics/ce_loss_without_sae"].item()
    )
    assert eval_metrics["ce_loss_diff"] < 0.1, "CE Loss Difference is too high"

    # CE Loss Difference
    _, cache = model.run_with_cache(
        example_text, names_filter=[sae.cfg.hook_name], prepend_bos=sae.cfg.prepend_bos
    )

    # Use the SAE
    sae_in = cache[sae.cfg.hook_name].squeeze()[1:]

    feature_acts = sae.encode(sae_in)
    # sae_out = sae.decode(feature_acts)

    mean_l0 = (feature_acts[1:] > 0).float().sum(-1).detach().cpu().numpy().mean()
    assert mean_l0 < 1000, f"mean L0 norm is too high: {mean_l0}"

    # # get the FVE of teh SAE
    # per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
    # total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
    # explained_variance = 1 - per_token_l2_loss / total_variance

    # assert (
    #     explained_variance.mean().item() > 0.7
    # ), f"Explained variance is too low: {explained_variance.mean().item()}"
