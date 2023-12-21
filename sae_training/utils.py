from typing import Tuple

import torch
from transformer_lens import HookedTransformer

from sae_training.activations_store import ActivationsStore
from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.sparse_autoencoder import SparseAutoencoder


class LMSparseAutoencoderSessionloader():
    """
    Responsible for loading all required
    artifacts and files for training
    a sparse autoencoder on a language model
    or analysing a pretraining autoencoder
    """

    def __init__(self, cfg: LanguageModelSAERunnerConfig):
        self.cfg = cfg
        
        
    def load_session(self) -> Tuple[HookedTransformer, SparseAutoencoder, ActivationsStore]:
        '''
        Loads a session for training a sparse autoencoder on a language model.
        '''
        
        model = self.get_model(self.cfg.model_name)
        model.to(self.cfg.device)
        activations_loader = self.get_activations_loader(self.cfg, model)
        sparse_autoencoder = self.initialize_sparse_autoencoder(self.cfg)
            
        return model, sparse_autoencoder, activations_loader
    
    @classmethod
    def load_session_from_pretrained(cls, path: str) -> Tuple[HookedTransformer, SparseAutoencoder, ActivationsStore]:
        '''
        Loads a session for analysing a pretrained sparse autoencoder.
        '''
        if torch.backends.mps.is_available():
            cfg = torch.load(path, map_location="mps")["cfg"]
            cfg.device = "mps"
        elif torch.cuda.is_available():
            cfg = torch.load(path, map_location="cuda")["cfg"]
        else:
            cfg = torch.load(path, map_location="cpu")["cfg"]

        model, _, activations_loader = cls(cfg).load_session()
        sparse_autoencoder = SparseAutoencoder.load_from_pretrained(path)
        
        return model, sparse_autoencoder, activations_loader
    
    def get_model(self, model_name: str):
        '''
        Loads a model from transformer lens
        '''
        
        # Todo: add check that model_name is valid
        
        model = HookedTransformer.from_pretrained(model_name)
        
        return model 
    
    def initialize_sparse_autoencoder(self, cfg: LanguageModelSAERunnerConfig):
        '''
        Initializes a sparse autoencoder
        '''
        
        sparse_autoencoder = SparseAutoencoder(cfg)
        
        return sparse_autoencoder
    
    def get_activations_loader(self, cfg: LanguageModelSAERunnerConfig, model: HookedTransformer):
        '''
        Loads a DataLoaderBuffer for the activations of a language model.
        '''
        
        activations_loader = ActivationsStore(
            cfg, model,
        )
        
        return activations_loader

def shuffle_activations_pairwise(datapath: str, buffer_idx_range: Tuple[int, int]):
    """
    Shuffles two buffers on disk.
    """
    assert buffer_idx_range[0] < buffer_idx_range[1], \
        "buffer_idx_range[0] must be smaller than buffer_idx_range[1]"
    
    buffer_idx1 = torch.randint(buffer_idx_range[0], buffer_idx_range[1], (1,)).item()
    buffer_idx2 = torch.randint(buffer_idx_range[0], buffer_idx_range[1], (1,)).item()
    while buffer_idx1 == buffer_idx2: # Make sure they're not the same
        buffer_idx2 = torch.randint(buffer_idx_range[0], buffer_idx_range[1], (1,)).item()
    
    buffer1 = torch.load(f"{datapath}/{buffer_idx1}.pt")
    buffer2 = torch.load(f"{datapath}/{buffer_idx2}.pt")
    joint_buffer = torch.cat([buffer1, buffer2])
    
    # Shuffle them
    joint_buffer = joint_buffer[torch.randperm(joint_buffer.shape[0])]
    shuffled_buffer1 = joint_buffer[:buffer1.shape[0]]
    shuffled_buffer2 = joint_buffer[buffer1.shape[0]:]
    
    # Save them back
    torch.save(shuffled_buffer1, f"{datapath}/{buffer_idx1}.pt")
    torch.save(shuffled_buffer2, f"{datapath}/{buffer_idx2}.pt")
