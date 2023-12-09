import pytest

from sae_training.toy_model_runner import SAEToyModelRunnerConfig, toy_model_sae_runner


# @pytest.mark.skip(reason="I (joseph) broke this at some point, on my to do list to fix.")
def test_toy_model_sae_runner():
    cfg = SAEToyModelRunnerConfig(
        
        # Model Details
        n_features=10,
        n_hidden=2,
        n_correlated_pairs=0,
        n_anticorrelated_pairs=0,
        feature_probability=0.025,
        model_training_steps=10_000,
        
        # SAE Parameters
        d_sae=10,
        l1_coefficient=0.001,
        
        # SAE Train Config
        train_batch_size=1028,
        feature_sampling_window=3_000,
        dead_feature_window=1_000,
        feature_reinit_scale=0.5,
        total_training_tokens=4096*300,
        
        # Other parameters
        log_to_wandb=True,
        wandb_project="sae-training-test",
        wandb_log_frequency=5,
        device="mps",
    )

    trained_sae = toy_model_sae_runner(cfg)

    assert trained_sae is not None
