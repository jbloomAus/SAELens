import os
from pathlib import Path

import pytest
import torch

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae import SAE
from tests.unit.helpers import build_sae_cfg


# Define a new fixture for different configurations
@pytest.fixture(
    params=[
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 64,
            "normalize_sae_decoder": False,
            "scale_sparsity_penalty_by_decoder_norm": True,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
            "tokenized": False,
            "hook_point": "blocks.1.hook_resid_pre",
            "hook_point_layer": 1,
            "d_in": 64,
        },
        {
            "model_name": "tiny-stories-1M",
            "dataset_path": "roneneldan/TinyStories",
            "tokenized": False,
            "hook_point": "blocks.1.attn.hook_z",
            "hook_point_layer": 1,
            "d_in": 64,
        },
    ],
    ids=[
        "tiny-stories-1M-resid-pre",
        "tiny-stories-1M-resid-pre-L1-W-dec-Norm",
        "tiny-stories-1M-resid-pre-pretokenized",
        "tiny-stories-1M-attn-out",
    ],
)
def cfg(request: pytest.FixtureRequest):
    """
    Pytest fixture to create a mock instance of LanguageModelSAERunnerConfig.
    """
    params = request.param
    return build_sae_cfg(**params)


def test_sae_init(cfg: LanguageModelSAERunnerConfig):
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())

    assert isinstance(sae, SAE)

    assert sae.W_enc.shape == (cfg.d_in, cfg.d_sae)
    assert sae.W_dec.shape == (cfg.d_sae, cfg.d_in)
    assert sae.b_enc.shape == (cfg.d_sae,)
    assert sae.b_dec.shape == (cfg.d_in,)


def test_sae_save_and_load_from_pretrained(tmp_path: Path) -> None:
    cfg = build_sae_cfg(device="cpu")
    model_path = str(tmp_path)
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    sae_state_dict = sae.state_dict()
    sae.save_model(model_path)

    assert os.path.exists(model_path)

    sae_loaded = SAE.load_from_pretrained(model_path, device="cpu")

    sae_loaded_state_dict = sae_loaded.state_dict()

    # check state_dict matches the original
    for key in sae.state_dict().keys():
        assert torch.allclose(
            sae_state_dict[key],
            sae_loaded_state_dict[key],
        )

    sae_in = torch.randn(10, cfg.d_in, device=cfg.device)
    sae_out_1 = sae(sae_in)
    sae_out_2 = sae_loaded(sae_in)
    assert torch.allclose(sae_out_1, sae_out_2)


# TODO: Handle scaling factor in saeBase
# def test_sae_save_and_load_from_pretrained_lacks_scaling_factor(
#     tmp_path: Path,
# ) -> None:
#     cfg = build_sae_cfg(device="cpu")
#     model_path = str(tmp_path)
#     sparse_autoencoder = saeBase(**cfg.get_sae_base_parameters())
#     sparse_autoencoder_state_dict = sparse_autoencoder.state_dict()

#     sparse_autoencoder.save_model(model_path)

#     assert os.path.exists(model_path)

#     sparse_autoencoder_loaded = saeBase.load_from_pretrained(model_path)
#     sparse_autoencoder_loaded.cfg.verbose = True
#     sparse_autoencoder_loaded.cfg.checkpoint_path = cfg.checkpoint_path
#     sparse_autoencoder_loaded = sparse_autoencoder_loaded.to("cpu")
#     sparse_autoencoder_loaded_state_dict = sparse_autoencoder_loaded.state_dict()
#     # check cfg matches the original
#     assert sparse_autoencoder_loaded.cfg == cfg

#     # check state_dict matches the original
#     for key in sparse_autoencoder.state_dict().keys():
#         if key == "scaling_factor":
#             assert isinstance(cfg.d_sae, int)
#             assert torch.allclose(
#                 torch.ones(cfg.d_sae, dtype=cfg.dtype, device=cfg.device),
#                 sparse_autoencoder_loaded_state_dict[key],
#             )
#         else:
#             assert torch.allclose(
#                 sparse_autoencoder_state_dict[key],
#                 sparse_autoencoder_loaded_state_dict[key],
#             )


def test_sae_get_name_returns_correct_name_from_cfg_vals() -> None:
    cfg = build_sae_cfg(
        model_name="test_model", hook_point="test_hook_point", d_sae=128
    )
    sae = SAE.from_dict(cfg.get_base_sae_cfg_dict())
    assert sae.get_name() == "sae_test_model_test_hook_point_128"
