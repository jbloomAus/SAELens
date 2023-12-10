from functools import partial

import einops
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

import wandb
from sae_training.activations_store import ActivationsStore
from sae_training.optim import get_scheduler
from sae_training.sparse_autoencoder import SparseAutoencoder


def train_sae_on_language_model(
    model: HookedTransformer,
    sparse_autoencoder: SparseAutoencoder,
    activation_store: ActivationsStore,
    batch_size: int = 1024,
    n_checkpoints: int = 0,
    feature_sampling_method: str = "l2",  # None, l2, or anthropic
    feature_sampling_window: int = 1000,  # how many training steps between resampling the features / considiring neurons dead
    feature_reinit_scale: float = 0.2,  # how much to scale the resampled features by
    dead_feature_threshold: float = 1e-8,  # how infrequently a feature has to be active to be considered dead
    dead_feature_window: int = 2000,  # how many training steps before a feature is considered dead
    use_wandb: bool = False,
    wandb_log_frequency: int = 50,
):


    total_training_tokens = sparse_autoencoder.cfg.total_training_tokens
    total_training_steps = total_training_tokens // batch_size
    n_training_steps = 0
    n_training_tokens = 0
    n_resampled_neurons = 0
    if n_checkpoints > 0:
        checkpoint_thresholds = list(range(0, total_training_tokens, total_training_tokens // n_checkpoints))[1:]
    
    # track active features
    act_freq_scores = torch.zeros(sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device)
    n_frac_active_tokens = 0
    
    optimizer = Adam(sparse_autoencoder.parameters(),
                     lr = sparse_autoencoder.cfg.lr)
    scheduler = get_scheduler(
        sparse_autoencoder.cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = sparse_autoencoder.cfg.lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end=sparse_autoencoder.cfg.lr / 10, # heuristic for now. 
    )
    sparse_autoencoder.train()
    

    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        # Do a training step.

        # Make sure the W_dec is still zero-norm
        sparse_autoencoder.set_decoder_norm_to_unit_norm()

        # Resample dead neurons
        if (feature_sampling_method is not None) and ((n_training_steps + 1) % dead_feature_window == 0):

            # Get the fraction of neurons active in the previous window
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            is_dead = (feature_sparsity < sparse_autoencoder.cfg.dead_feature_threshold)
            
            # if standard resampling <- do this
            n_resampled_neurons = sparse_autoencoder.resample_neurons(
                activation_store.next_batch(), 
                feature_sparsity, 
                feature_reinit_scale,
                optimizer
            )

        else:
            n_resampled_neurons = 0

        # Update learning rate here if using scheduler.

        # Forward and Backward Passes
        optimizer.zero_grad()
        x = activation_store.next_batch()
        sae_out, feature_acts, loss, mse_loss, l1_loss = sparse_autoencoder(activation_store.next_batch())
        n_training_tokens += batch_size

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            n_frac_active_tokens += batch_size
            feature_sparsity = act_freq_scores / n_frac_active_tokens

            # metrics for currents acts
            l0 = (feature_acts > 0).float().sum(1).mean()
            l2_norm_in = torch.norm(x, dim=-1).mean()
            l2_norm_out = torch.norm(sae_out, dim=-1).mean()
            l2_norm_ratio = l2_norm_out / l2_norm_in
            current_learning_rate = optimizer.param_groups[0]["lr"]

            if use_wandb and ((n_training_steps + 1) % wandb_log_frequency == 0):
                wandb.log(
                    {
                        "losses/mse_loss": mse_loss.item(),
                        "losses/l1_loss": l1_loss.item(),
                        "losses/overall_loss": loss.item(),
                        "metrics/l0": l0.item(),
                        "metrics/l2": l2_norm_out.item(),
                        "metrics/l2_ratio": l2_norm_ratio.item(),
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
                        "details/n_training_tokens": n_training_tokens,
                        "metrics/n_resampled_neurons": n_resampled_neurons,
                        "metrics/current_learning_rate": current_learning_rate,
                    },
                    step=n_training_steps,
                )

                # record loss frequently, but not all the time.
                if (n_training_steps + 1) % (wandb_log_frequency * 10) == 0:
                    # Now we want the reconstruction loss.
                    recons_score, ntp_loss, recons_loss, zero_abl_loss = get_recons_loss(sparse_autoencoder, model, activation_store, num_batches=3)
                    
                    wandb.log(
                        {
                            "metrics/reconstruction_score": recons_score,
                            "metrics/ce_loss_without_sae": ntp_loss,
                            "metrics/ce_loss_with_sae": recons_loss,
                            "metrics/ce_loss_with_ablation": zero_abl_loss,
                            
                        },
                        step=n_training_steps,
                    )
                    
                # use feature window to log feature sparsity
                if ((n_training_steps + 1) % feature_sampling_window == 0):
                    log_feature_sparsity = torch.log10(feature_sparsity + 1e-10)
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
        scheduler.step()

        # checkpoint if at checkpoint frequency
        if n_checkpoints > 0 and n_training_tokens > checkpoint_thresholds[0]:
            cfg = sparse_autoencoder.cfg
            path = f"{sparse_autoencoder.cfg.checkpoint_path}/{n_training_tokens}_{sparse_autoencoder.get_name()}.pt"
            sparse_autoencoder.save_model(path)
            checkpoint_thresholds.pop(0)
            if len(checkpoint_thresholds) == 0:
                n_checkpoints = 0
            if cfg.log_to_wandb:
                model_artifact = wandb.Artifact(
                    f"{sparse_autoencoder.get_name()}", type="model", metadata=dict(cfg.__dict__)
                )
                model_artifact.add_file(path)
                wandb.log_artifact(model_artifact)
            
        n_training_steps += 1

    return sparse_autoencoder

@torch.no_grad()
def get_recons_loss(sparse_autoencder, model, activation_store, num_batches=5):
    hook_point = activation_store.cfg.hook_point
    loss_list = []
    for _ in range(num_batches):
        batch_tokens = activation_store.get_batch_tokens()
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
