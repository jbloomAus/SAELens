from functools import partial

import einops
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

import wandb
from sae_training.sparse_autoencoder import SparseAutoencoder


def train_sae_on_language_model(
    model: HookedTransformer,
    sparse_autoencder: SparseAutoencoder,
    data_loader_buffer,
    batch_size: int = 1024,
    feature_sampling_method: str = "l2",  # None, l2, or anthropic
    feature_sampling_window: int = 100,  # how many training steps between resampling the features / considiring neurons dead
    feature_reinit_scale: float = 0.2,  # how much to scale the resampled features by
    dead_feature_threshold: float = 1e-8,  # how infrequently a feature has to be active to be considered dead
    use_wandb: bool = False,
    wandb_log_frequency: int = 50,
):
    optimizer = torch.optim.Adam(sparse_autoencder.parameters())
    frac_active_list = []  # track active features

    sparse_autoencder.train()
    n_training_steps = 0
    n_training_tokens = 0

    # start the buffer
    buffer = data_loader_buffer.get_buffer()
    dataloader = iter(DataLoader(buffer, batch_size=batch_size, shuffle=True))
    n_remaining_batches_in_buffer = len(dataloader)

    total_training_tokens = sparse_autoencder.cfg.total_training_tokens
    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        # Do a training step.

        # Make sure the W_dec is still zero-norm
        sparse_autoencder.set_decoder_norm_to_unit_norm()

        # Resample dead neurons
        if (feature_sampling_method is not None) and ((n_training_steps + 1) % feature_sampling_window == 0):

            # Get the fraction of neurons active in the previous window
            frac_active_in_window = torch.stack(frac_active_list[-feature_sampling_window:], dim=0)
            feature_sparsity = frac_active_in_window.sum(0) / (
                                feature_sampling_window * batch_size
                            )
            # if standard resampling <- do this
            n_resampled_neurons = sparse_autoencder.resample_neurons(next(dataloader), feature_sparsity, feature_reinit_scale)
            n_remaining_batches_in_buffer -= 1

            # elif anthropic resampling <- do this
            # run the model and reinit where recons loss is high. 
            if n_remaining_batches_in_buffer == 0:
                dataloader, n_remaining_batches_in_buffer = get_new_dataloader(
                    data_loader_buffer, n_remaining_batches_in_buffer, batch_size)
        else:
            n_resampled_neurons = 0

        # # Update learning rate here if using scheduler.

        # Generate Activations

        # Forward and Backward Passes
        optimizer.zero_grad()
        _, feature_acts, loss, mse_loss, l1_loss = sparse_autoencder(next(dataloader))
        n_training_tokens += batch_size
        n_remaining_batches_in_buffer -= 1

        # Update the buffer if we've run out of batches
        if n_remaining_batches_in_buffer == 0:
            dataloader, n_remaining_batches_in_buffer = get_new_dataloader(
                data_loader_buffer, n_remaining_batches_in_buffer, batch_size)

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
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
                feature_sparsity = act_freq_scores.sum(0) / (
                    len(frac_active_list) * batch_size
                )

            # metrics for currents acts
            l0 = (feature_acts > 0).float().sum(0).mean()
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
                        "metrics/dead_features": (
                            feature_sparsity < dead_feature_threshold
                        )
                        .float()
                        .mean()
                        .item(),
                        "metrics/n_resampled_neurons": n_resampled_neurons,
                        "details/n_training_tokens": n_training_tokens,
                    },
                    step=n_training_steps,
                )

                if (n_training_steps + 1) % (wandb_log_frequency * 10) == 0:
                    log_feature_sparsity = torch.log(feature_sparsity + 1e-8)
                    wandb.log(
                        {
                            "plots/feature_density_histogram": wandb.Histogram(
                                log_feature_sparsity.tolist()
                            ),
                        },
                        step=n_training_steps,
                    )

                    # Now we want the reconstruction loss.
                    recons_score, _, _, _ = get_recons_loss(
                        sparse_autoencder, model, data_loader_buffer=data_loader_buffer, num_batches=5)
                    
                    wandb.log(
                        {
                            "metrics/reconstruction_score": recons_score,
                        },
                        step=n_training_steps,
                    )



            pbar.set_description(
                f"{n_training_steps}| MSE Loss {mse_loss.item():.3f} | L0 {l0.item():.3f}"
            )

        loss.backward()
        sparse_autoencder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()

        n_training_steps += 1

    return sparse_autoencder


def get_new_dataloader(data_loader_buffer, n_remaining_batches_in_buffer, batch_size):
    buffer = data_loader_buffer.get_buffer()
    dataloader = iter(DataLoader(buffer, batch_size=batch_size, shuffle=True))
    n_remaining_batches_in_buffer = len(dataloader) // 2 # only ever use half the buffer .
    return dataloader, n_remaining_batches_in_buffer



@torch.no_grad()
def get_recons_loss(sparse_autoencder, model, data_loader_buffer, num_batches=5):
    hook_point = data_loader_buffer.cfg.hook_point
    loss_list = []
    for _ in range(num_batches):
        batch_tokens = data_loader_buffer.get_batch_tokens()
        loss = model(batch_tokens, return_type="loss")

        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss",
        # fwd_hooks=[(utils.get_act_name("post", 0), mean_ablate_hook)])

        recons_loss = model.run_with_hooks(
            batch_tokens,
            return_type="loss",
            fwd_hooks=[(hook_point, partial(replacement_hook, encoder=sparse_autoencder))],
        )

        zero_abl_loss = model.run_with_hooks(
            batch_tokens, return_type="loss", fwd_hooks=[(hook_point, zero_ablate_hook)]
        )
        loss_list.append((loss, recons_loss, zero_abl_loss))

    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    # print(loss, recons_loss, zero_abl_loss)
    # print(f"{score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")

    return score, loss, recons_loss, zero_abl_loss


def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[0]
    return mlp_post_reconstr


def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post


def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.0
    return mlp_post
