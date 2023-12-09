import einops
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.toy_models import Model as ToyModel


def train_toy_sae(
    model: ToyModel,
    sparse_autoencoder: SparseAutoencoder,
    activation_store,
    batch_size: int = 1024,
    total_training_tokens: int = 1024 * 10_000,
    feature_sampling_method: str = "l2",  # None, l2, or anthropic
    feature_sampling_window: int = 100,  # how many training steps between resampling the features / considiring neurons dead
    dead_feature_window: int = 2000,  # how many training steps before a feature is considered dead
    feature_reinit_scale: float = 0.2,  # how much to scale the resampled features by
    dead_feature_threshold: float = 1e-8,  # how infrequently a feature has to be active to be considered dead
    use_wandb: bool = False,
    wandb_log_frequency: int = 50,
):
    """
    Takes an SAE and a bunch of activations and does a bunch of training steps
    """

    dataloader = iter(DataLoader(activation_store, batch_size=batch_size, shuffle=True))
    optimizer = torch.optim.Adam(sparse_autoencoder.parameters())
    sparse_autoencoder.train()
    frac_active_list = []  # track active features

    n_training_steps = 0
    n_training_tokens = 0
    n_resampled_neurons = 0

    pbar = tqdm(dataloader, desc="Training SAE")
    for _, batch in enumerate(pbar):
        
        batch = next(dataloader)
        # Make sure the W_dec is still zero-norm
        sparse_autoencoder.set_decoder_norm_to_unit_norm()

        # Resample dead neurons
        if (feature_sampling_method is not None) and ((n_training_steps + 1) % dead_feature_window == 0):

            # Get the fraction of neurons active in the previous window
            frac_active_in_window = torch.stack(frac_active_list[-dead_feature_window:], dim=0)
            feature_sparsity = frac_active_in_window.sum(0) / (
                                dead_feature_window * batch_size
                            )

            # Compute batch of hidden activations which we'll use in resampling
            resampling_batch = model.generate_batch(batch_size)

            # Our version of running the model
            hidden = einops.einsum(
                resampling_batch,
                model.W,
                "batch_size instances features, instances hidden features -> batch_size instances hidden",
            )

            # Resample
            n_resampled_neurons = sparse_autoencoder.resample_neurons(
                hidden, feature_sparsity, feature_reinit_scale
            )

        # Update learning rate here if using scheduler.

        # Forward and Backward Passes
        optimizer.zero_grad()
        _, feature_acts, loss, mse_loss, l1_loss = sparse_autoencoder(batch)
        n_training_tokens += batch_size

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list
            act_freq_scores = (feature_acts.abs() > 0).float().sum(0)
            frac_active_list.append(act_freq_scores)

            if len(frac_active_list) > feature_sampling_window:
                frac_active_in_window = torch.stack(
                    frac_active_list[-feature_sampling_window:], dim=0
                )
                feature_sparsity = frac_active_in_window.sum(0) / (
                    feature_sampling_window * batch_size
                )
            else:
                # use the whole list
                frac_active_in_window = torch.stack(frac_active_list, dim=0)
                feature_sparsity = frac_active_in_window.sum(0) / (
                    len(frac_active_list) * batch_size
                )


            l0 = (feature_acts > 0).float().sum(1).mean()
            l2_norm = torch.norm(feature_acts, dim=1).mean()

            if use_wandb and ((n_training_steps + 1) % wandb_log_frequency == 0):
                wandb.log(
                    {
                        "losses/mse_loss": mse_loss.item(),
                        "losses/l1_loss": l1_loss.item(),
                        "losses/overall_loss": loss.item(),
                        "metrics/l0": l0.item(),
                        "metrics/l2": l2_norm.item(),
                        "metrics/below_1e-5": (feature_sparsity < 1e-5)
                        .float()
                        .mean()
                        .item(),
                        "metrics/below_1e-6": (feature_sparsity < 1e-6)
                        .float()
                        .mean()
                        .item(),
                        "metrics/n_dead_features": (
                            feature_sparsity < dead_feature_threshold
                        )
                        .float()
                        .mean()
                        .item(),
                        "metrics/n_resampled_neurons": n_resampled_neurons,
                    },
                    step=n_training_steps,
                    
                )

            if (n_training_steps + 1) % (wandb_log_frequency * 100) == 0:
                log_feature_sparsity = torch.log10(feature_sparsity + 1e-8)
                wandb.log(
                    {
                        "plots/feature_density_histogram": wandb.Histogram(
                            log_feature_sparsity.tolist()
                        ),
                    },
                    step=n_training_steps,
                )

            pbar.set_description(
                f"{n_training_steps}| MSE Loss {mse_loss.item():.3f} | L0 {l0.item():.3f}"
            )
            pbar.update(batch_size)

        loss.backward()
        sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        
        # If we did checkpointing we'd do it here.

        n_training_steps += 1

    return sparse_autoencoder
