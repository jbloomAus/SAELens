from dataclasses import dataclass

import torch
from transformer_lens import HookedTransformer

from sae_training.activation_store import ActivationStore
from sae_training.lm_datasets import get_mapped_dataset
from sae_training.SAE import SAE
from sae_training.train_sae import train_sae


@dataclass
class SAERunnerConfig:

    # Data Generating Function (Model + Training Distibuion)
    model_name: str = "gelu-2l"
    hook_point: str = "blocks.0.hook_mlp_out"
    hook_point_layer: int = 0
    dataset_path: str = "NeelNanda/c4-tokenized-2b"
    
    # SAE Parameters
    expansion_factor: int = 4
    
    # Training Parameters
    l1_coefficient: float = 1e-3
    lr: float = 3e-4
    train_batch_size: int = 4096
    context_size: int = 128
    
    # Activation Store Parameters
    # max_store_size: int = 384 * 4096 * 2
    # max_activations: int = 2_000_000_000
    # resample_frequency: int = 122_880_000
    # checkpoint_frequency: int = 100_000_000
    # validation_frequency: int = 384 * 4096 * 2 * 100
    
    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "mats_sae_training"
    wandb_entity: str = None
    
    # Misc
    device: str = "cpu"
    seed: int = 42
    checkpoint_path: str = "checkpoints"
    dtype: torch.dtype = torch.float32

def sae_runner(cfg):


    model = HookedTransformer.from_pretrained("gelu-2l") # any other cfg we should pass in here?
    
    # initialize dataset
    dataset = get_mapped_dataset(cfg)
    activation_store = ActivationStore(cfg, dataset)
    
    # initialize the SAE
    sparse_autoencoder = SAE(cfg)
    
    # train SAE
    sparse_autoencoder = train_sae(
        model, 
        activation_store, 
        sparse_autoencoder, 
        cfg)
    
    return trained_sae