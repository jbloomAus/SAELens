import torch

from sae_lens.toy_model_runner import ToyModelSAERunnerConfig, toy_model_sae_runner


# @pytest.mark.skip(reason="I (joseph) broke this at some point, on my to do list to fix.")
def test_toy_model_sae_runner():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    cfg = ToyModelSAERunnerConfig(
        # Model Details
        n_features=100,
        n_hidden=10,
        n_correlated_pairs=0,
        n_anticorrelated_pairs=0,
        feature_probability=0.025,
        model_training_steps=10_000,
        # SAE Parameters
        d_sae=10,
        lr=3e-4,
        l1_coefficient=0.001,
        use_ghost_grads=False,
        b_dec_init_method="mean",
        # SAE Train Config
        train_batch_size=1028,
        feature_sampling_window=3_000,
        dead_feature_window=1_000,
        total_training_tokens=4096 * 1000,
        # Other parameters
        log_to_wandb=True,
        wandb_project="mats_sae_training_benchmarks_toy",
        wandb_log_frequency=5,
        device=device,
    )

    trained_sae = toy_model_sae_runner(cfg)

    assert trained_sae is not None
