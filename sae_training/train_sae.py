#%%
import einops
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from sae_training.activation_store import ActivationStore
from sae_training.SAE import SAE


#%%
def train_sae(sae: SAE,
              activation_store: ActivationStore,
              n_epochs: int = 10,
              batch_size: int = 32,
              l1_coeff: float = 0.001,
              use_wandb: bool = False,
              wandb_log_freq: int = 10,):
    """
        Takes an SAE and a bunch of activations and does a bunch of training steps
    """
    
    dataloader = DataLoader(activation_store, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(sae.parameters())
    
    sae.train()
    n_training_steps = 0
    for epoch in range(n_epochs):
        pbar = tqdm(dataloader)
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            sae_out, hidden_post = sae(batch)
            # loss = reconstruction MSE + L1 regularization
            mse_loss = ((sae_out - batch)**2).mean()
            l1_loss = torch.abs(hidden_post).sum()
            loss = mse_loss + l1_coeff * l1_loss

            with torch.no_grad():
                
                batch_size = batch.shape[0]
                feature_mean_activation = hidden_post.mean(dim=0)
                n_dead_features = (feature_mean_activation == 0).sum()
                l0 = ((hidden_post != 0) / batch_size).sum()
                l2_norm = torch.norm(hidden_post, dim=1).mean()
                
                
            if use_wandb and (step % wandb_log_freq == 0):
                wandb.log({
                    "mse_loss": mse_loss.item(),
                    "l1_loss": l1_loss.item(),
                    "loss": loss.item(),
                    "l0":  l0.item(),
                    "l2": l2_norm.item(),
                    "hist": wandb.Histogram(feature_mean_activation.tolist()),
                    "n_dead_features": n_dead_features,
                }, step=n_training_steps)
                
            pbar.set_description(f"Epoch {epoch} | Step {step} | MSE Loss {mse_loss.item():.3f} | L1 Loss {l1_loss.item():.3f} | L0 {l0.item():.3f} | n_dead_features {n_dead_features}")
            
            loss.backward()
            
            # Taken from Artur's code https://github.com/ArthurConmy/sae/blob/3f8c314d9c008ec40de57828762ec5c9159e4092/sae/utils.py#L91
            # TODO do we actually need this?
            # Update grads so that they remove the parallel component
            # (d_sae, d_in) shape
            with torch.no_grad():
                parallel_component = einops.einsum(
                    sae.W_dec.grad,
                    sae.W_dec.data,
                    "d_sae d_in, d_sae d_in -> d_sae",
                )
                sae.W_dec.grad -= einops.einsum(
                    parallel_component,
                    sae.W_dec.data,
                    "d_sae, d_sae d_in -> d_sae d_in",
                )
            
            optimizer.step()
            
            # Make sure the W_dec is still zero-norm
            with torch.no_grad():
                sae.W_dec.data /= (torch.norm(sae.W_dec.data, dim=1, keepdim=True) + 1e-8)
                
            n_training_steps += 1


    return sae