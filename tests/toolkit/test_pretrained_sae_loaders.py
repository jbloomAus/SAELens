from pathlib import Path

import pytest
import torch
from sparsify import SparseCoder, SparseCoderConfig

from sae_lens.sae import SAE
from sae_lens.toolkit.pretrained_sae_loaders import (
    get_deepseek_r1_config_from_hf,
    get_llama_scope_r1_distill_config_from_hf,
    load_sae_config_from_huggingface,
    sparsify_disk_loader,
)


def test_load_sae_config_from_huggingface():
    cfg_dict = load_sae_config_from_huggingface(
        "gpt2-small-res-jb",
        sae_id="blocks.0.hook_resid_pre",
    )

    expected_cfg_dict = {
        "activation_fn_str": "relu",
        "apply_b_dec_to_input": True,
        "architecture": "standard",
        "model_name": "gpt2-small",
        "hook_point": "blocks.0.hook_resid_pre",
        "hook_point_layer": 0,
        "hook_point_head_index": None,
        "dataset_path": "Skylion007/openwebtext",
        "dataset_trust_remote_code": True,
        "is_dataset_tokenized": False,
        "context_size": 128,
        "use_cached_activations": False,
        "cached_activations_path": "activations/Skylion007_openwebtext/gpt2-small/blocks.0.hook_resid_pre",
        "d_in": 768,
        "n_batches_in_buffer": 128,
        "total_training_tokens": 300000000,
        "store_batch_size": 32,
        "device": "cpu",
        "seed": 42,
        "dtype": "torch.float32",
        "b_dec_init_method": "geometric_median",
        "expansion_factor": 32,
        "from_pretrained_path": None,
        "l1_coefficient": 8e-05,
        "lr": 0.0004,
        "lr_scheduler_name": None,
        "lr_warm_up_steps": 5000,
        "model_from_pretrained_kwargs": {
            "center_writing_weights": True,
        },
        "train_batch_size": 4096,
        "use_ghost_grads": False,
        "feature_sampling_window": 1000,
        "finetuning_scaling_factor": False,
        "feature_sampling_method": None,
        "resample_batches": 1028,
        "feature_reinit_scale": 0.2,
        "dead_feature_window": 5000,
        "dead_feature_estimation_method": "no_fire",
        "dead_feature_threshold": 1e-08,
        "log_to_wandb": True,
        "wandb_project": "mats_sae_training_gpt2_small_resid_pre_5",
        "wandb_entity": None,
        "wandb_log_frequency": 100,
        "n_checkpoints": 10,
        "checkpoint_path": "checkpoints/y1t51byy",
        "d_sae": 24576,
        "tokens_per_buffer": 67108864,
        "run_name": "24576-L1-8e-05-LR-0.0004-Tokens-3.000e+08",
        "neuronpedia_id": "gpt2-small/0-res-jb",
        "normalize_activations": "none",
        "prepend_bos": True,
        "sae_lens_training_version": None,
    }

    assert cfg_dict == expected_cfg_dict


def test_load_sae_config_from_huggingface_connor_rob_hook_z():
    cfg_dict = load_sae_config_from_huggingface(
        "gpt2-small-hook-z-kk",
        sae_id="blocks.0.hook_z",
    )

    expected_cfg_dict = {
        "architecture": "standard",
        "d_in": 768,
        "d_sae": 24576,
        "dtype": "float32",
        "device": "cpu",
        "model_name": "gpt2-small",
        "hook_name": "blocks.0.attn.hook_z",
        "hook_layer": 0,
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "Skylion007/openwebtext",
        "context_size": 128,
        "normalize_activations": "none",
        "dataset_trust_remote_code": True,
        "neuronpedia_id": "gpt2-small/0-att-kk",
    }

    assert cfg_dict == expected_cfg_dict


def test_load_sae_config_from_huggingface_gemma_2():
    cfg_dict = load_sae_config_from_huggingface(
        "gemma-scope-2b-pt-res",
        sae_id="embedding/width_4k/average_l0_6",
    )

    expected_cfg_dict = {
        "architecture": "jumprelu",
        "d_in": 2304,
        "d_sae": 4096,
        "dtype": "float32",
        "model_name": "gemma-2-2b",
        "hook_name": "hook_embed",
        "hook_layer": 0,
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": None,
        "device": "cpu",
        "neuronpedia_id": None,
    }

    assert cfg_dict == expected_cfg_dict


def test_load_sae_config_from_huggingface_dictionary_learning_1():
    cfg_dict = load_sae_config_from_huggingface(
        "sae_bench_gemma-2-2b_topk_width-2pow16_date-1109",
        sae_id="blocks.12.hook_resid_post__trainer_0",
    )

    expected_cfg_dict = {
        "architecture": "standard",
        "d_in": 2304,
        "d_sae": 65536,
        "dtype": "float32",
        "device": "cpu",
        "model_name": "gemma-2-2b",
        "hook_name": "blocks.12.hook_resid_post",
        "hook_layer": 12,
        "hook_head_index": None,
        "activation_fn_str": "topk",
        "activation_fn_kwargs": {"k": 20},
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "dataset_trust_remote_code": True,
        "context_size": 128,
        "normalize_activations": "none",
        "neuronpedia_id": "gemma-2-2b/12-sae_bench-topk-res-65k__trainer_0_step_final",
    }

    assert cfg_dict == expected_cfg_dict


def test_load_sae_config_from_huggingface_matches_from_pretrained():
    from_pretrained_cfg_dict = SAE.from_pretrained(
        "gpt2-small-res-jb",
        sae_id="blocks.0.hook_resid_pre",
        device="cpu",
    )[1]
    direct_sae_cfg = load_sae_config_from_huggingface(
        "gpt2-small-res-jb",
        sae_id="blocks.0.hook_resid_pre",
        device="cpu",
    )

    assert direct_sae_cfg == from_pretrained_cfg_dict


def test_get_deepseek_r1_config_from_hf():
    """Test that the DeepSeek R1 config is generated correctly."""
    cfg = get_deepseek_r1_config_from_hf(
        repo_id="some/repo",
        folder_name="DeepSeek-R1-Distill-Llama-8B-SAE-l19.pt",
        device="cpu",
    )

    expected_cfg = {
        "architecture": "standard",
        "d_in": 4096,  # LLaMA 8B hidden size
        "d_sae": 4096 * 16,  # Expansion factor 16
        "dtype": "bfloat16",
        "context_size": 1024,
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "hook_name": "blocks.19.hook_resid_post",
        "hook_layer": 19,
        "hook_head_index": None,
        "prepend_bos": True,
        "dataset_path": "lmsys/lmsys-chat-1m",
        "dataset_trust_remote_code": True,
        "sae_lens_training_version": None,
        "activation_fn_str": "relu",
        "normalize_activations": "none",
        "device": "cpu",
        "apply_b_dec_to_input": False,
        "finetuning_scaling_factor": False,
    }

    assert cfg == expected_cfg


def test_get_deepseek_r1_config_with_invalid_layer():
    """Test that get_deepseek_r1_config raises ValueError with invalid layer in filename."""
    with pytest.raises(
        ValueError, match="Could not find layer number in filename: invalid_filename.pt"
    ):
        get_deepseek_r1_config_from_hf(
            repo_id="some/repo", folder_name="invalid_filename.pt", device="cpu"
        )


def test_get_llama_scope_r1_distill_config_from_hf():
    """Test that the Llama Scope R1 Distill config is generated correctly."""
    cfg = get_llama_scope_r1_distill_config_from_hf(
        repo_id="fnlp/Llama-Scope-R1-Distill",
        folder_name="800M-Slimpajama-0-OpenR1-Math-220k/L5R",
        device="cpu",
        force_download=False,
        cfg_overrides=None,
    )

    expected_cfg = {
        "architecture": "jumprelu",
        "d_in": 4096,  # LLaMA 8B hidden size
        "d_sae": 4096 * 8,  # Expansion factor
        "dtype": "float32",
        "device": "cpu",
        "model_name": "meta-llama/Llama-3.1-8B",
        "hook_name": "blocks.5.hook_resid_post",
        "hook_layer": 5,
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "cerebras/SlimPajama-627B",
        "context_size": 1024,
        "dataset_trust_remote_code": True,
        "apply_b_dec_to_input": False,
        "normalize_activations": "expected_average_only_in",
    }

    assert cfg == expected_cfg


def test_get_llama_scope_r1_distill_config_with_overrides():
    """Test that config overrides work correctly for Llama Scope R1 Distill."""
    cfg_overrides = {
        "device": "cuda",
        "dtype": "float16",
        "d_sae": 8192,
    }

    cfg = get_llama_scope_r1_distill_config_from_hf(
        repo_id="fnlp/Llama-Scope-R1-Distill",
        folder_name="400M-Slimpajama-400M-OpenR1-Math-220k/L10R",
        device="cuda",
        cfg_overrides=cfg_overrides,
    )

    assert cfg["device"] == "cuda"
    assert cfg["dtype"] == "float16"
    assert cfg["d_sae"] == 8192
    assert cfg["hook_layer"] == 10


def test_sparsify_disk_loader(tmp_path: Path):
    d_in = 5
    cfg = SparseCoderConfig(
        expansion_factor=3,
        num_latents=d_in * 3,
        k=2,
        normalize_decoder=False,
    )
    sparsify_sae = SparseCoder(d_in, cfg=cfg, dtype=torch.bfloat16)
    path = tmp_path / "layers.0"
    sparsify_sae.save_to_disk(path)

    cfg_dict, state_dict = sparsify_disk_loader(path)

    assert cfg_dict["d_in"] == sparsify_sae.d_in
    assert cfg_dict["d_sae"] == sparsify_sae.num_latents
    assert cfg_dict["activation_fn_str"] == sparsify_sae.cfg.activation
    assert cfg_dict["activation_fn_kwargs"]["k"] == sparsify_sae.cfg.k

    torch.testing.assert_close(state_dict["W_enc"], sparsify_sae.encoder.weight.data.T)
    torch.testing.assert_close(state_dict["b_enc"], sparsify_sae.encoder.bias.data)
    # sparsify_sae.W_dec is Optional in the type stubs, so first assert it's present
    assert sparsify_sae.W_dec is not None
    torch.testing.assert_close(state_dict["W_dec"], sparsify_sae.W_dec.detach().T)
    torch.testing.assert_close(state_dict["b_dec"], sparsify_sae.b_dec.data)
