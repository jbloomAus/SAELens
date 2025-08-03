from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from sparsify import SparseCoder, SparseCoderConfig

from sae_lens.loading.pretrained_sae_loaders import (
    dictionary_learning_sae_huggingface_loader_1,
    get_deepseek_r1_config_from_hf,
    get_gemma_2_transcoder_config_from_hf,
    get_llama_scope_config_from_hf,
    get_llama_scope_r1_distill_config_from_hf,
    load_sae_config_from_huggingface,
    read_sae_components_from_disk,
    sparsify_disk_loader,
    sparsify_huggingface_loader,
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
        "reshape_activations": "none",
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
        "reshape_activations": "hook_z",
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
        "reshape_activations": "none",
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


def test_load_sae_config_from_huggingface_gemma_2_hook_z_saes():
    cfg_dict = load_sae_config_from_huggingface(
        "gemma-scope-2b-pt-att",
        sae_id="layer_0/width_16k/average_l0_104",
    )

    expected_cfg_dict = {
        "d_in": 2048,
        "d_sae": 16384,
        "dtype": "float32",
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
        "reshape_activations": "hook_z",
        "device": "cpu",
        "metadata": {
            "model_name": "gemma-2-2b",
            "hook_name": "blocks.0.attn.hook_z",
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
        "reshape_activations": "none",
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


def test_get_gemma_2_transcoder_config_from_hf():
    cfg = get_gemma_2_transcoder_config_from_hf(
        repo_id="google/gemma-scope-2b-pt-transcoders",
        folder_name="layer_3/width_16k/average_l0_54",
        device="cpu",
    )

    expected_cfg = expected_cfg = {
        "architecture": "jumprelu_transcoder",
        "d_in": 2304,
        "d_out": 2304,
        "d_sae": 16384,
        "dtype": "float32",
        "device": "cpu",
        "activation_fn": "relu",
        "normalize_activations": "none",
        "model_name": "gemma-2-2b",
        "hook_name": "blocks.3.ln2.hook_normalized",
        "hook_name_out": "blocks.3.hook_mlp_out",
        "hook_head_index": None,
        "hook_head_index_out": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 1024,
    }

    assert cfg == expected_cfg


def test_load_sae_config_from_huggingface_gemma_2_transcoder():
    cfg = load_sae_config_from_huggingface(
        release="gemma-scope-2b-pt-transcoders",
        sae_id="layer_3/width_16k/average_l0_54",
        device="cpu",
    )

    expected_cfg = {
        "d_in": 2304,
        "d_out": 2304,
        "d_sae": 16384,
        "dtype": "float32",
        "device": "cpu",
        "normalize_activations": "none",
        "apply_b_dec_to_input": True,
        "reshape_activations": "none",
        "metadata": {
            "model_name": "gemma-2-2b",
            "hook_name": "blocks.3.ln2.hook_normalized",
            "hook_name_out": "blocks.3.hook_mlp_out",
            "hook_head_index": None,
            "hook_head_index_out": None,
            "prepend_bos": True,
            "dataset_path": "monology/pile-uncopyrighted",
            "context_size": 1024,
            "neuronpedia_id": "gemma-2-2b/3-gemmascope-transcoder-16k",
            "sae_lens_training_version": None,
        },
        "architecture": "jumprelu_transcoder",
    }

    assert cfg == expected_cfg


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


def test_get_llama_scope_config_from_hf():
    cfg = get_llama_scope_config_from_hf(
        repo_id="fnlp/Llama3_1-8B-Base-LXA-32x",
        folder_name="Llama3_1-8B-Base-L0A-32x",
        device="cpu",
        force_download=False,
        cfg_overrides=None,
    )

    expected_cfg = {
        "architecture": "jumprelu",
        "d_in": 4096,
        "d_sae": 4096 * 32,
        "dtype": "bfloat16",
        "device": "cpu",
        "model_name": "meta-llama/Llama-3.1-8B",
        "hook_name": "blocks.0.hook_attn_out",
        "jump_relu_threshold": 1.0616438356164384,
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


def test_sparsify_huggingface_loader():
    sparsify_sae = SparseCoder.load_from_hub(
        "EleutherAI/sae-llama-3-8b-32x", hookpoint="layers.10"
    )

    cfg_dict, state_dict, _ = sparsify_huggingface_loader(
        "EleutherAI/sae-llama-3-8b-32x", folder_name="layers.10"
    )

    assert cfg_dict["d_in"] == sparsify_sae.d_in
    assert cfg_dict["d_sae"] == sparsify_sae.num_latents
    assert cfg_dict["activation_fn_str"] == sparsify_sae.cfg.activation
    assert cfg_dict["activation_fn_kwargs"]["k"] == sparsify_sae.cfg.k

    torch.testing.assert_close(
        state_dict["W_enc"], sparsify_sae.encoder.weight.data.T, check_dtype=False
    )
    torch.testing.assert_close(
        state_dict["b_enc"], sparsify_sae.encoder.bias.data, check_dtype=False
    )
    # sparsify_sae.W_dec is Optional in the type stubs, so first assert it's present
    assert sparsify_sae.W_dec is not None
    torch.testing.assert_close(
        state_dict["W_dec"], sparsify_sae.W_dec.detach().T, check_dtype=False
    )
    torch.testing.assert_close(
        state_dict["b_dec"], sparsify_sae.b_dec.data, check_dtype=False
    )


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


def test_dictionary_learning_sae_huggingface_loader_1():
    cfg_dict, state_dict, sparsity = dictionary_learning_sae_huggingface_loader_1(
        "canrager/lm_sae",
        "pythia70m_sweep_gated_ctx128_0730/resid_post_layer_3/trainer_0",
        device="cpu",
        force_download=False,
        cfg_overrides=None,
    )
    assert sparsity is None
    assert state_dict.keys() == {"W_enc", "W_dec", "b_dec", "b_mag", "b_gate", "r_mag"}
    assert cfg_dict == {
        "architecture": "gated",
        "d_in": 512,
        "d_sae": 4096,
        "dtype": "float32",
        "device": "cpu",
        "model_name": "pythia-70m-deduped",
        "hook_name": "blocks.3.hook_resid_post",
        "hook_head_index": None,
        "activation_fn": "relu",
        "activation_fn_kwargs": {},
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "context_size": 128,
        "normalize_activations": "none",
        "neuronpedia_id": None,
        "dataset_trust_remote_code": True,
    }
    assert state_dict["W_enc"].shape == (512, 4096)
    assert state_dict["W_dec"].shape == (4096, 512)
    assert state_dict["b_dec"].shape == (512,)
    assert state_dict["b_mag"].shape == (4096,)
    assert state_dict["b_gate"].shape == (4096,)
    assert state_dict["r_mag"].shape == (4096,)


def test_read_sae_components_from_disk(tmp_path: Path):
    d_in = 256
    d_sae = 512
    device = "cpu"
    dtype = torch.float32

    # Create dummy SAE components
    W_enc = torch.randn(d_in, d_sae, dtype=dtype)
    W_dec = torch.randn(d_sae, d_in, dtype=dtype)
    b_enc = torch.randn(d_sae, dtype=dtype)
    b_dec = torch.randn(d_in, dtype=dtype)

    # Create state dict
    state_dict = {
        "W_enc": W_enc,
        "W_dec": W_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
    }

    # Save to disk
    weights_path = tmp_path / "sae_weights.safetensors"
    save_file(state_dict, weights_path)

    # Create config dict
    cfg_dict = {
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "device": device,
        "finetuning_scaling_factor": False,
    }

    # Read back from disk
    loaded_cfg_dict, loaded_state_dict = read_sae_components_from_disk(
        cfg_dict=cfg_dict,
        weight_path=weights_path,
        device=device,
    )

    # Check that config dict is returned unchanged (except for finetuning_scaling_factor)
    assert loaded_cfg_dict["d_in"] == d_in
    assert loaded_cfg_dict["d_sae"] == d_sae
    assert loaded_cfg_dict["dtype"] == "float32"
    assert loaded_cfg_dict["device"] == device
    assert loaded_cfg_dict["finetuning_scaling_factor"] is False

    # Check that all tensors are loaded correctly
    assert loaded_state_dict.keys() == {"W_enc", "W_dec", "b_enc", "b_dec"}
    torch.testing.assert_close(loaded_state_dict["W_enc"], W_enc)
    torch.testing.assert_close(loaded_state_dict["W_dec"], W_dec)
    torch.testing.assert_close(loaded_state_dict["b_enc"], b_enc)
    torch.testing.assert_close(loaded_state_dict["b_dec"], b_dec)

    # Check tensor shapes
    assert loaded_state_dict["W_enc"].shape == (d_in, d_sae)
    assert loaded_state_dict["W_dec"].shape == (d_sae, d_in)
    assert loaded_state_dict["b_enc"].shape == (d_sae,)
    assert loaded_state_dict["b_dec"].shape == (d_in,)

    # Check tensor dtypes
    assert loaded_state_dict["W_enc"].dtype == dtype
    assert loaded_state_dict["W_dec"].dtype == dtype
    assert loaded_state_dict["b_enc"].dtype == dtype
    assert loaded_state_dict["b_dec"].dtype == dtype


def test_read_sae_components_from_disk_with_scaling_factor(tmp_path: Path):
    d_in = 128
    d_sae = 256
    device = "cpu"
    dtype = torch.float32

    # Create dummy SAE components with scaling factor
    W_enc = torch.randn(d_in, d_sae, dtype=dtype)
    W_dec = torch.randn(d_sae, d_in, dtype=dtype)
    b_enc = torch.randn(d_sae, dtype=dtype)
    b_dec = torch.randn(d_in, dtype=dtype)
    scaling_factor = torch.tensor([1.5, 2.0, 0.8], dtype=dtype)

    # Create state dict with scaling factor
    state_dict = {
        "W_enc": W_enc,
        "W_dec": W_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
        "scaling_factor": scaling_factor,
    }

    # Save to disk
    weights_path = tmp_path / "sae_weights.safetensors"
    save_file(state_dict, weights_path)

    # Create config dict with finetuning_scaling_factor enabled
    cfg_dict = {
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "device": device,
        "finetuning_scaling_factor": True,
    }

    # Read back from disk
    loaded_cfg_dict, loaded_state_dict = read_sae_components_from_disk(
        cfg_dict=cfg_dict,
        weight_path=weights_path,
        device=device,
    )

    # Check that scaling factor is renamed to finetuning_scaling_factor
    assert "scaling_factor" not in loaded_state_dict
    assert "finetuning_scaling_factor" in loaded_state_dict
    torch.testing.assert_close(
        loaded_state_dict["finetuning_scaling_factor"], scaling_factor
    )

    # Check that config dict is returned correctly
    assert loaded_cfg_dict["d_in"] == d_in
    assert loaded_cfg_dict["d_sae"] == d_sae
    assert loaded_cfg_dict["dtype"] == "float32"
    assert loaded_cfg_dict["device"] == device
    assert loaded_cfg_dict["finetuning_scaling_factor"] is True

    # Check that other tensors are still there
    assert loaded_state_dict.keys() == {
        "W_enc",
        "W_dec",
        "b_enc",
        "b_dec",
        "finetuning_scaling_factor",
    }
    torch.testing.assert_close(loaded_state_dict["W_enc"], W_enc)
    torch.testing.assert_close(loaded_state_dict["W_dec"], W_dec)
    torch.testing.assert_close(loaded_state_dict["b_enc"], b_enc)
    torch.testing.assert_close(loaded_state_dict["b_dec"], b_dec)


def test_read_sae_components_from_disk_with_ones_scaling_factor(tmp_path: Path):
    d_in = 64
    d_sae = 128
    device = "cpu"
    dtype = torch.float32

    # Create dummy SAE components with scaling factor of all ones
    W_enc = torch.randn(d_in, d_sae, dtype=dtype)
    W_dec = torch.randn(d_sae, d_in, dtype=dtype)
    b_enc = torch.randn(d_sae, dtype=dtype)
    b_dec = torch.randn(d_in, dtype=dtype)
    scaling_factor = torch.ones(3, dtype=dtype)

    # Create state dict with scaling factor
    state_dict = {
        "W_enc": W_enc,
        "W_dec": W_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
        "scaling_factor": scaling_factor,
    }

    # Save to disk
    weights_path = tmp_path / "sae_weights.safetensors"
    save_file(state_dict, weights_path)

    # Create config dict
    cfg_dict = {
        "d_in": d_in,
        "d_sae": d_sae,
        "dtype": "float32",
        "device": device,
        "finetuning_scaling_factor": False,
    }

    # Read back from disk
    loaded_cfg_dict, loaded_state_dict = read_sae_components_from_disk(
        cfg_dict=cfg_dict,
        weight_path=weights_path,
        device=device,
    )

    # Check that scaling factor of all ones is removed
    assert "scaling_factor" not in loaded_state_dict
    assert "finetuning_scaling_factor" not in loaded_state_dict
    assert loaded_cfg_dict["finetuning_scaling_factor"] is False

    # Check that config dict is returned correctly
    assert loaded_cfg_dict["d_in"] == d_in
    assert loaded_cfg_dict["d_sae"] == d_sae
    assert loaded_cfg_dict["dtype"] == "float32"
    assert loaded_cfg_dict["device"] == device

    # Check that other tensors are still there
    assert loaded_state_dict.keys() == {"W_enc", "W_dec", "b_enc", "b_dec"}
    torch.testing.assert_close(loaded_state_dict["W_enc"], W_enc)
    torch.testing.assert_close(loaded_state_dict["W_dec"], W_dec)
    torch.testing.assert_close(loaded_state_dict["b_enc"], b_enc)
    torch.testing.assert_close(loaded_state_dict["b_dec"], b_dec)
