import pytest
from datasets import Dataset
from transformer_lens import HookedTransformer

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.evals import run_evals
from sae_lens.sae import SAE
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.training_sae import TrainingSAE
from tests.unit.helpers import TINYSTORIES_MODEL, build_sae_cfg, load_model_cached


@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "tokenized": False,
            "hook_name": "blocks.1.hook_resid_pre",
            "hook_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_name": "blocks.1.attn.hook_z",
            "hook_layer": 1,
            "d_in": 16 * 4,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_name": "blocks.1.attn.hook_q",
            "hook_layer": 1,
            "d_in": 16 * 4,
        },
    ],
    ids=[
        "tiny-stories-1M-resid-pre",
        "tiny-stories-1M-resid-pre-L1-W-dec-Norm",
        "tiny-stories-1M-resid-pre-pretokenized",
        "tiny-stories-1M-hook-z",
        "tiny-stories-1M-hook-q",
    ],
)
def cfg(request: pytest.FixtureRequest):
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    params = request.param
    return build_sae_cfg(**params)


@pytest.fixture
def model():
    return load_model_cached(TINYSTORIES_MODEL)


@pytest.fixture
def activation_store(model: HookedTransformer, cfg: LanguageModelSAERunnerConfig):
    return ActivationsStore.from_config(
        model, cfg, dataset=Dataset.from_list([{"text": "hello world"}] * 2000)
    )


@pytest.fixture
def base_sae(cfg: LanguageModelSAERunnerConfig):
    return SAE.from_dict(cfg.get_base_sae_cfg_dict())


@pytest.fixture
def training_sae(cfg: LanguageModelSAERunnerConfig):
    return TrainingSAE.from_dict(cfg.get_training_sae_cfg_dict())


def test_run_evals_base_sae(
    base_sae: SAE,
    activation_store: ActivationsStore,
    model: HookedTransformer,
):

    eval_metrics = run_evals(
        sae=base_sae,
        activation_store=activation_store,
        model=model,
        n_eval_batches=2,
        eval_batch_size_prompts=None,
    )

    expected_keys = [
        "metrics/l2_norm",
        "metrics/l2_ratio",
        "metrics/l2_norm_in",
        "metrics/CE_loss_score",
        "metrics/ce_loss_without_sae",
        "metrics/ce_loss_with_sae",
        "metrics/ce_loss_with_ablation",
    ]

    # results will be garbage without a real model.
    for key in expected_keys:
        assert key in eval_metrics


def test_run_evals_training_sae(
    training_sae: TrainingSAE,
    activation_store: ActivationsStore,
    model: HookedTransformer,
):

    eval_metrics = run_evals(
        sae=training_sae,
        activation_store=activation_store,
        model=model,
        n_eval_batches=10,
        eval_batch_size_prompts=None,
    )

    expected_keys = [
        "metrics/l2_norm",
        "metrics/l2_ratio",
        "metrics/l2_norm_in",
        "metrics/CE_loss_score",
        "metrics/ce_loss_without_sae",
        "metrics/ce_loss_with_sae",
        "metrics/ce_loss_with_ablation",
    ]

    for key in expected_keys:
        assert key in eval_metrics
