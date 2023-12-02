import os
from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

import wandb
from sae_training.activations_buffer import DataLoaderBuffer
from sae_training.lm_datasets import preprocess_tokenized_dataset

# from sae_training.activation_store import ActivationStore
from sae_training.SAE import SAE
from sae_training.train_sae_on_language_model import train_sae_on_language_model


@dataclass
class LanguageModelSAERunnerConfig:

    # Data Generating Function (Model + Training Distibuion)
    model_name: str = "gelu-2l"
    hook_point: str = "blocks.0.hook_mlp_out"
    hook_point_layer: int = 0
    dataset_path: str = "NeelNanda/c4-tokenized-2b"
    is_dataset_tokenized: bool = True
    
    # SAE Parameters
    d_in: int = 512
    expansion_factor: int = 4
    
    # Training Parameters
    l1_coefficient: float = 1e-3
    lr: float = 3e-4
    train_batch_size: int = 4096
    context_size: int = 128
    
    # Resampling protocol args
    feature_sampling_window: int = 100
    feature_reinit_scale: float = 0.2
    dead_feature_threshold: float = 1e-8
    
    
    # Activation Store Parameters
    n_batches_in_buffer: int = 20
    total_training_tokens: int = 2_000_000
    store_batch_size: int = 4096
    
    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "mats_sae_training_language_model"
    wandb_entity: str = None
    wandb_log_frequency: int = 5
    
    # Misc
    device: str = "cpu"
    seed: int = 42
    checkpoint_path: str = "checkpoints"
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        self.d_sae = self.d_in * self.expansion_factor
        self.tokens_per_buffer = self.train_batch_size * self.context_size * self.n_batches_in_buffer

def language_model_sae_runner(cfg):


    # get the model
    model = HookedTransformer.from_pretrained(cfg.model_name) # any other cfg we should pass in here?
    
    # initialize dataset
    activations_buffer = DataLoaderBuffer(
        cfg, model, data_path=cfg.dataset_path, is_dataset_tokenized=cfg.is_dataset_tokenized,
    )

    # initialize the SAE
    sparse_autoencoder = SAE(cfg)
    
    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, config=cfg)
    
    # train SAE
    sparse_autoencoder = train_sae_on_language_model(
        model, sparse_autoencoder, activations_buffer,
        batch_size = cfg.train_batch_size,
        feature_sampling_window = cfg.feature_sampling_window,
        feature_reinit_scale = cfg.feature_reinit_scale,
        dead_feature_threshold = cfg.dead_feature_threshold,
        use_wandb = cfg.log_to_wandb,
        wandb_log_frequency = cfg.wandb_log_frequency
    )


        
    # save sae to checkpoints folder
    unique_id = wandb.util.generate_id()
    #make sure directory exists

    os.makedirs(f"{cfg.checkpoint_path}/{unique_id}", exist_ok=True)
    torch.save(sparse_autoencoder.state_dict(), f"{cfg.checkpoint_path}/{unique_id}/sae.pt")
    # upload to wandb
    if cfg.log_to_wandb:
        model_artifact = wandb.Artifact(
            "sae", type="model", metadata=dict(cfg.__dict__)
        )
        model_artifact.add_file(f"{cfg.checkpoint_path}/{unique_id}/sae.pt")
        wandb.log_artifact(model_artifact)
        

    if cfg.log_to_wandb:
        wandb.finish()
        
    return sparse_autoencoder