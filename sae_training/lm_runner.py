from dataclasses import dataclass

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

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
    
    # SAE Parameters
    d_in: int = 768
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
    shuffle_buffer_size: int = 10_000
    # max_store_size: int = 384 * 4096 * 2
    # max_activations: int = 2_000_000_000
    # resample_frequency: int = 122_880_000
    # checkpoint_frequency: int = 100_000_000
    # validation_frequency: int = 384 * 4096 * 2 * 100
    
    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "mats_sae_training_language_model"
    wandb_entity: str = None
    wandb_log_frequency: int = 50
    
    # Misc
    device: str = "cpu"
    seed: int = 42
    checkpoint_path: str = "checkpoints"
    dtype: torch.dtype = torch.float32
    
    def __post_init__(self):
        self.d_sae = self.d_in * self.expansion_factor

def language_model_sae_runner(cfg):


    # get the model
    model = HookedTransformer.from_pretrained(cfg.model_name) # any other cfg we should pass in here?
    
    # initialize dataset
    dataset = load_dataset(cfg.dataset_path, streaming=True, split="train")
    existing_columns = list(next(iter(dataset)).keys())
    mapped_dataset = dataset.map(
        preprocess_tokenized_dataset, # preprocess is what differentiates different datasets
        batched=True,
        batch_size=cfg.train_batch_size,
        fn_kwargs={"context_size": cfg.context_size},
        remove_columns=existing_columns,
    )
    dataset = mapped_dataset.shuffle(buffer_size=cfg.shuffle_buffer_size)
    dataloader = DataLoader(dataset, batch_size=cfg.train_batch_size)

    # initialize the SAE
    sparse_autoencoder = SAE(cfg)
    
    # train SAE
    sparse_autoencoder = train_sae_on_language_model(
        model, sparse_autoencoder, dataloader,
        batch_size = cfg.train_batch_size,
        feature_sampling_window = cfg.feature_sampling_window,
        feature_reinit_scale = cfg.feature_reinit_scale,
        dead_feature_threshold = cfg.feature_reinit_scale,
        use_wandb = cfg.log_to_wandb,
        wandb_log_frequency = cfg.wandb_log_frequency
    )
    
    return sparse_autoencoder