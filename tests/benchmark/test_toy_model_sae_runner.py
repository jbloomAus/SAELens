from sae_training.toy_model_runner import SAEToyModelRunnerConfig, toy_model_sae_runner


def test_toy_model_sae_runner():
    cfg = SAEToyModelRunnerConfig(
        n_features=5,
        n_hidden=2,
        n_correlated_pairs=0,
        n_anticorrelated_pairs=0,
        feature_probability=0.025,
        # SAE Parameters
        d_sae=5,
        l1_coefficient=0.005,
        # SAE Train Config
        train_batch_size=1024,
        feature_sampling_window=3_000,
        feature_reinit_scale=0.5,
        model_training_steps=10_000,
        n_sae_training_tokens=1024*10_000,
        train_epochs=1,
        log_to_wandb=True,
        wandb_log_frequency=5,
    )

    trained_sae = toy_model_sae_runner(cfg)

    assert trained_sae is not None
