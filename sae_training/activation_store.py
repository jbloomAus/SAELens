import torch
from torch.utils.data import Dataset

# Partially copied from Alan's code: https://github.com/ai-safety-foundation/sparse_autoencoder/blob/main/sparse_autoencoder/activation_store/tensor_store.py
class ActivationStore(Dataset):
    """
        Stores activations in a tensor (in RAM)
    """
    
    def __init__(self, d_activation: int, max_items: int = 1024, device="cpu", dtype=torch.float32):
        self.data = torch.zeros((max_items, d_activation), device=device, dtype=dtype)
        self.max_items = max_items
        self.items = 0
    
    def append(self, x: torch.Tensor):
        if self.items < self.max_items:
            self.data[self.items] = x
            self.items += 1
        else:
            raise RuntimeError("ActivationStore is full")
    
    def extend(self, xs: torch.Tensor):
        if self.items + xs.shape[0] <= self.max_items:
            self.data[self.items:self.items+xs.shape[0]] = xs
            self.items += xs.shape[0]
        else:
            raise RuntimeError("ActivationStore is full")
    
    def empty(self):
        self.items = 0
    
    def shuffle(self):
        self.data[:self.items] = self.data[torch.randperm(self.items)]
    
    def __len__(self):
        return self.items

    def __getitem__(self, i):
        if i >= self.items:
            raise IndexError("Index out of bounds")
        
        return self.data[i]
