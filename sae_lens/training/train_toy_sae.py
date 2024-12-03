from typing import Any, cast

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from sae_lens.sae import SAE


def train_toy_sae(
    sparse_autoencoder: SAE,
    activation_store: torch.Tensor,  # TODO: this type seems strange / wrong
    batch_size: int = 1024,
    feature_sampling_window: int = 100,  # how many training steps between resampling the features / considiring neurons dead
    dead_feature_threshold: float = 1e-8,  # how infrequently a feature has to be active to be considered dead
    use_wandb: bool = False,
    wandb_log_frequency: int = 50,
):
    """
    Takes an SAE and a bunch of activations and does a bunch of training steps
    """

    # TODO: this type seems strange
    dataloader = iter(
        DataLoader(cast(Any, activation_store), batch_size=batch_size, shuffle=True)
    )
    optimizer = torch.optim.Adam(sparse_autoencoder.parameters())
    sparse_autoencoder.train()
    frac_active_list = []  # track active features

    n_training_tokens = 0

    pbar = tqdm(dataloader, desc="Training SAE")
    for n_training_steps, batch in enumerate(
        pbar
    ):  # Use enumerate to track training steps
        batch = next(dataloader)
        # Make sure the W_dec is still zero-norm
        if sparse_autoencoder.normalize_sae_decoder:
            sparse_autoencoder.set_decoder_norm_to_unit_norm()

        # Forward and Backward Passes
        optimizer.zero_grad()
        sae_out, feature_acts, loss, mse_loss, l1_loss, _ = sparse_autoencoder(batch)
        loss.backward()
        if sparse_autoencoder.normalize_sae_decoder:
            sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()

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

            l0 = (feature_acts > 0).float().sum(-1).mean()
            l2_norm = torch.norm(feature_acts, dim=1).mean()

            l2_norm_in = torch.norm(batch, dim=-1)
            l2_norm_out = torch.norm(sae_out, dim=-1)
            l2_norm_ratio = l2_norm_out / (1e-6 + l2_norm_in)

            if use_wandb and ((n_training_steps + 1) % wandb_log_frequency == 0):
                wandb.log(
                    {
                        "details/n_training_tokens": n_training_tokens,
                        "losses/mse_loss": mse_loss.item(),
                        "losses/l1_loss": l1_loss.item(),
                        "losses/overall_loss": loss.item(),
                        "metrics/l0": l0.item(),
                        "metrics/l2": l2_norm.item(),
                        "metrics/l2_ratio": l2_norm_ratio.mean().item(),
                        "sparsity/below_1e-5": (feature_sparsity < 1e-5)
                        .float()
                        .mean()
                        .item(),
                        "sparsity/below_1e-6": (feature_sparsity < 1e-6)
                        .float()
                        .mean()
                        .item(),
                        "sparsity/n_dead_features": (
                            feature_sparsity < dead_feature_threshold
                        )
                        .float()
                        .mean()
                        .item(),
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

        # If we did checkpointing we'd do it here.

        # If we did checkpointing we'd do it here.

    return sparse_autoencoder
