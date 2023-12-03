import os

import torch

import wandb

# from sae_training.activation_store import ActivationStore
from sae_training.train_sae_on_language_model import train_sae_on_language_model
from sae_training.utils import LMSparseAutoencoderSessionloader


def language_model_sae_runner(cfg):
    """
    
    """
    
    
    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sparse_autoencoder, activations_loader = loader.load_session()

    
    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, config=cfg)
    
    # train SAE
    sparse_autoencoder = train_sae_on_language_model(
        model, sparse_autoencoder, activations_loader,
        batch_size = cfg.train_batch_size,
        feature_sampling_method = cfg.feature_sampling_method,
        feature_sampling_window = cfg.feature_sampling_window,
        feature_reinit_scale = cfg.feature_reinit_scale,
        dead_feature_threshold = cfg.dead_feature_threshold,
        use_wandb = cfg.log_to_wandb,
        wandb_log_frequency = cfg.wandb_log_frequency
    )

    # save sae to checkpoints folder
    unique_id = wandb.util.generate_id()
    path = f"{cfg.checkpoint_path}/{unique_id}/final_{sparse_autoencoder.get_name()}.pt"
    sparse_autoencoder.save_model(path)
    
    # upload to wandb
    if cfg.log_to_wandb:
        model_artifact = wandb.Artifact(
            f"{sparse_autoencoder.get_name()}", type="model", metadata=dict(cfg.__dict__)
        )
        model_artifact.add_file(path)
        wandb.log_artifact(model_artifact, aliases=["final_model"])
        

    if cfg.log_to_wandb:
        wandb.finish()
        
    return sparse_autoencoder