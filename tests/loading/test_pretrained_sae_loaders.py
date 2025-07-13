import pytest

from sae_lens.loading.pretrained_sae_loaders import (
    get_deepseek_r1_config_from_hf,
    get_llama_scope_r1_distill_config_from_hf,
    load_sae_config_from_huggingface,
)
from sae_lens.saes.sae import SAE


def test_load_sae_config_from_huggingface():
    cfg_dict = load_sae_config_from_huggingface(
        "gpt2-small-res-jb",
        sae_id="blocks.0.hook_resid_pre",
    )

    expected_cfg_dict = {
        "d_in": 768,
        "device": "cpu",
        "dtype": "torch.float32",
        "d_sae": 24576,
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "metadata": {
            "model_name": "gpt2-small",
            "hook_name": "blocks.0.hook_resid_pre",
            "hook_head_index": None,
            "context_size": 128,
            "model_from_pretrained_kwargs": {"center_writing_weights": True},
            "neuronpedia_id": "gpt2-small/0-res-jb",
            "prepend_bos": True,
            "dataset_path": "Skylion007/openwebtext",
            "sae_lens_training_version": None,
        },
        "architecture": "standard",
    }

    assert cfg_dict == expected_cfg_dict


def test_load_sae_config_from_huggingface_connor_rob_hook_z():
    cfg_dict = load_sae_config_from_huggingface(
        "gpt2-small-hook-z-kk",
        sae_id="blocks.0.hook_z",
    )

    expected_cfg_dict = {
        "d_in": 768,
        "d_sae": 24576,
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "metadata": {
            "model_name": "gpt2-small",
            "hook_name": "blocks.0.attn.hook_z",
            "hook_head_index": None,
            "prepend_bos": True,
            "dataset_path": "Skylion007/openwebtext",
            "context_size": 128,
            "neuronpedia_id": "gpt2-small/0-att-kk",
            "sae_lens_training_version": None,
        },
        "architecture": "standard",
    }

    assert cfg_dict == expected_cfg_dict


def test_load_sae_config_from_huggingface_gemma_2():
    cfg_dict = load_sae_config_from_huggingface(
        "gemma-scope-2b-pt-res",
        sae_id="embedding/width_4k/average_l0_6",
    )

    expected_cfg_dict = {
        "d_in": 2304,
        "d_sae": 4096,
        "dtype": "float32",
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
        "device": "cpu",
        "metadata": {
            "model_name": "gemma-2-2b",
            "hook_name": "hook_embed",
            "hook_head_index": None,
            "prepend_bos": True,
            "dataset_path": "monology/pile-uncopyrighted",
            "context_size": 1024,
            "neuronpedia_id": None,
            "sae_lens_training_version": None,
        },
        "architecture": "jumprelu",
    }

    assert cfg_dict == expected_cfg_dict


def test_load_sae_config_from_huggingface_dictionary_learning_1():
    cfg_dict = load_sae_config_from_huggingface(
        "sae_bench_gemma-2-2b_topk_width-2pow16_date-1109",
        sae_id="blocks.12.hook_resid_post__trainer_0",
    )

    expected_cfg_dict = {
        "d_in": 2304,
        "d_sae": 65536,
        "dtype": "float32",
        "device": "cpu",
        "apply_b_dec_to_input": True,
        "normalize_activations": "none",
        "metadata": {
            "model_name": "gemma-2-2b",
            "hook_name": "blocks.12.hook_resid_post",
            "hook_head_index": None,
            "prepend_bos": True,
            "dataset_path": "monology/pile-uncopyrighted",
            "context_size": 128,
            "neuronpedia_id": "gemma-2-2b/12-sae_bench-topk-res-65k__trainer_0_step_final",
            "sae_lens_training_version": None,
        },
        "architecture": "standard",
    }

    assert cfg_dict == expected_cfg_dict


def test_load_sae_config_from_huggingface_matches_from_pretrained():
    from_pretrained_cfg_dict = SAE.from_pretrained_with_cfg_and_sparsity(
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
        "hook_head_index": None,
        "prepend_bos": True,
        "dataset_path": "lmsys/lmsys-chat-1m",
        "dataset_trust_remote_code": True,
        "sae_lens_training_version": None,
        "activation_fn": "relu",
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
        "hook_head_index": None,
        "activation_fn": "relu",
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
