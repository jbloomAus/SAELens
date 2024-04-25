import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from scripts.config import get_lm_sae_runner_config

from sae_lens import language_model_sae_runner

if __name__ == "__main__":
    cfg = get_lm_sae_runner_config()
    cfg.sae_class_name = "GatedSparseAutoencoder"

    sparse_autoencoder = language_model_sae_runner(cfg)
