import einops
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.toy_models import Model as ToyModel


def train_toy_sae(model: ToyModel, 
              sparse_autoencoder: SparseAutoencoder,
              activation_store,
              n_epochs: int = 10,
              batch_size: int = 1024,
              feature_sampling_window: int = 100, # how many training steps between resampling the features / considiring neurons dead
              feature_reinit_scale: float = 0.2, # how much to scale the resampled features by
              dead_feature_threshold: float = 1e-8, # how infrequently a feature has to be active to be considered dead
              use_wandb: bool = False,
              wandb_log_frequency: int = 50,):
    """
    Takes an SAE and a bunch of activations and does a bunch of training steps
    """
    
    dataloader = DataLoader(activation_store, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(sparse_autoencoder.parameters())
    frac_active_list = [] # track active features
    
    sparse_autoencoder.train()
    n_training_steps = 0
    for epoch in range(n_epochs):
        pbar = tqdm(dataloader)
        for step, batch in enumerate(pbar):
            
            # Make sure the W_dec is still zero-norm
            sparse_autoencoder.set_decoder_norm_to_unit_norm()
            
            # Resample dead neurons 
            if (feature_sampling_window is not None) and ((step + 1) % feature_sampling_window == 0):
                
                # Get the fraction of neurons active in the previous window
                frac_active_in_window = torch.stack(frac_active_list[-feature_sampling_window:], dim=0)
                
                # Compute batch of hidden activations which we'll use in resampling
                resampling_batch = model.generate_batch(batch_size)
                
                # Our version of running the model
                hidden = einops.einsum(
                    resampling_batch,
                    model.W,
                    "batch_size instances features, instances hidden features -> batch_size instances hidden",
                )
            
                # Resample
                sparse_autoencoder.resample_neurons(hidden, frac_active_in_window, feature_reinit_scale)
                
            
            # Update learning rate here if using scheduler.

            # Forward and Backward Passes
            optimizer.zero_grad()
            _, feature_acts, loss, mse_loss, l1_loss = sparse_autoencoder(batch)
            # loss = reconstruction MSE + L1 regularization
           
            with torch.no_grad():
                
                # Calculate the sparsities, and add it to a list
                frac_active = einops.reduce((feature_acts.abs() > dead_feature_threshold).float(), "batch_size hidden_ae -> hidden_ae", "mean")
                frac_active_list.append(frac_active)
                
                batch_size = batch.shape[0]
                log_frac_feature_activation = torch.log(frac_active + 1e-8)
                n_dead_features = (frac_active < dead_feature_threshold).sum()
                
                l0 = (feature_acts > 0).float().mean()
                l2_norm = torch.norm(feature_acts, dim=1).mean() 
                
                    
                if use_wandb and ((step + 1) % wandb_log_frequency == 0):
                    wandb.log({
                        "losses/mse_loss": mse_loss.item(),
                        "losses/l1_loss": batch_size*l1_loss.item(),
                        "losses/overall_loss": loss.item(),
                        "metrics/l0":  l0.item(),
                        "metrics/l2": l2_norm.item(),
                        # "metrics/feature_density_histogram": wandb.Histogram(log_frac_feature_activation.tolist()),
                        "metrics/n_dead_features": n_dead_features,
                        "metrics/n_alive_features": sparse_autoencoder.d_sae - n_dead_features,
                    }, step=n_training_steps)
                    
                pbar.set_description(f"{epoch}/{step}| MSE Loss {mse_loss.item():.3f} | L0 {l0.item():.3f} | n_dead_features {n_dead_features}")
            
            loss.backward()
            sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
            optimizer.step()
            
            n_training_steps += 1


    return sparse_autoencoder