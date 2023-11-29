#%%
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from activation_store import ActivationStore

#%%
def gather_activations(transformer: HookedTransformer,
                       dataloader: DataLoader,
                       hook_point: str,
                       layer: int,
                       activation_store: ActivationStore):
    """
        Takes a hooked transformer and a tokenized dataloader, gathers activations at a
        specific hook point, and stores them in an activation store

        Example:
            transformer = HookedTransformer.from_pretrained('gpt2-small')
            dataloader = DataLoader([torch.randint(0, 100, (64,)) for _ in range(16)])
            store = ActivationStore(768)
            gather_activations(transformer, dataloader, 'resid_post', 3, store) 
    """
    with torch.no_grad():
        for batch in dataloader:
            _, cache = transformer.run_with_cache(batch)
            d_activations = cache[hook_point, layer].shape[-1]
            activations = cache[hook_point, layer].reshape(-1, d_activations)
            activation_store.extend(activations)
