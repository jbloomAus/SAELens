from sae_lens.toolkit.pretrained_sae_loaders import SAEConfigLoadOptions, get_sae_config


def test_get_sae_config_sae_lens():
    cfg_dict = get_sae_config(
        "gpt2-small-res-jb",
        sae_id="blocks.0.hook_resid_pre",
        options=SAEConfigLoadOptions(),
    )

    expected_cfg_dict = {
        "model_name": "gpt2-small",
        "hook_point": "blocks.0.hook_resid_pre",
        "hook_point_layer": 0,
        "hook_point_head_index": None,
        "dataset_path": "Skylion007/openwebtext",
        "is_dataset_tokenized": False,
        "context_size": 128,
        "use_cached_activations": False,
        "cached_activations_path": "activations/Skylion007_openwebtext/gpt2-small/blocks.0.hook_resid_pre",
        "d_in": 768,
        "n_batches_in_buffer": 128,
        "total_training_tokens": 300000000,
        "store_batch_size": 32,
        "device": "mps",
        "seed": 42,
        "dtype": "torch.float32",
        "b_dec_init_method": "geometric_median",
        "expansion_factor": 32,
        "from_pretrained_path": None,
        "l1_coefficient": 8e-05,
        "lr": 0.0004,
        "lr_scheduler_name": None,
        "lr_warm_up_steps": 5000,
        "train_batch_size": 4096,
        "use_ghost_grads": False,
        "feature_sampling_window": 1000,
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
    }

    assert cfg_dict == expected_cfg_dict


def test_get_sae_config_connor_rob_hook_z():
    cfg_dict = get_sae_config(
        "gpt2-small-hook-z-kk",
        sae_id="blocks.0.hook_z",
        options=SAEConfigLoadOptions(),
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
    }

    assert cfg_dict == expected_cfg_dict


def test_get_sae_config_gemma_2():
    cfg_dict = get_sae_config(
        "gemma-scope-2b-pt-res",
        sae_id="embedding/width_4k/average_l0_6",
        options=SAEConfigLoadOptions(),
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
    }

    assert cfg_dict == expected_cfg_dict


def test_get_sae_config_dictionary_learning_1():
    cfg_dict = get_sae_config(
        "sae_bench_gemma-2-2b_sweep_standard_ctx128_ef2_0824",
        sae_id="blocks.3.hook_resid_post__trainer_1_step_29292",
        options=SAEConfigLoadOptions(),
    )

    expected_cfg_dict = {
        "architecture": "standard",
        "d_in": 2304,
        "d_sae": 4608,
        "dtype": "float32",
        "device": "cpu",
        "model_name": "gemma-2-2b",
        "hook_name": "blocks.3.hook_resid_post",
        "hook_layer": 3,
        "hook_head_index": None,
        "activation_fn_str": "relu",
        "activation_fn_kwargs": {},
        "apply_b_dec_to_input": True,
        "finetuning_scaling_factor": False,
        "sae_lens_training_version": None,
        "prepend_bos": True,
        "dataset_path": "monology/pile-uncopyrighted",
        "dataset_trust_remote_code": False,
        "context_size": 128,
        "normalize_activations": "none",
        "neuronpedia_id": None,
    }

    assert cfg_dict == expected_cfg_dict
