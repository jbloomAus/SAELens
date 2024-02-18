import torch
from torch.optim import Adam
from tqdm import tqdm
from transformer_lens import HookedTransformer

import wandb
from sae_training.activations_store import ActivationsStore
from sae_training.evals import run_evals
from sae_training.optim import get_scheduler
from sae_training.sparse_autoencoder import SparseAutoencoder


def train_sae_on_language_model(
    model: HookedTransformer,
    sparse_autoencoder: SparseAutoencoder,
    activation_store: ActivationsStore,
    batch_size: int = 1024,
    n_checkpoints: int = 0,
    feature_sampling_window: int = 1000,  # how many training steps between resampling the features / considiring neurons dead
    dead_feature_threshold: float = 1e-8,  # how infrequently a feature has to be active to be considered dead
    use_wandb: bool = False,
    wandb_log_frequency: int = 50,
):
    total_training_tokens = sparse_autoencoder.cfg.total_training_tokens
    total_training_steps = total_training_tokens // batch_size
    n_training_steps = 0
    n_training_tokens = 0

    if n_checkpoints > 0:
        checkpoint_thresholds = list(
            range(0, total_training_tokens, total_training_tokens // n_checkpoints)
        )[1:]

    # track active features
    act_freq_scores = torch.zeros(
        sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device
    )
    n_forward_passes_since_fired = torch.zeros(
        sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device
    )
    n_frac_active_tokens = 0

    optimizer = Adam(sparse_autoencoder.parameters(), lr=sparse_autoencoder.cfg.lr, betas=(0.9, 0.99))
    scheduler = get_scheduler(
        sparse_autoencoder.cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps=sparse_autoencoder.cfg.lr_warm_up_steps,
        training_steps=total_training_steps,
        lr_end=sparse_autoencoder.cfg.lr / 10,  # heuristic for now.
    )
    sparse_autoencoder.initialize_b_dec(activation_store)
    sparse_autoencoder.train()

    # Initialise this to None, in case we train on very few tokens
    log_feature_sparsity = None

    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        # Do a training step.
        sparse_autoencoder.train()
        # Make sure the W_dec is still zero-norm
        sparse_autoencoder.set_decoder_norm_to_unit_norm()

        # log and then reset the feature sparsity every feature_sampling_window steps
        if (n_training_steps + 1) % feature_sampling_window == 0:
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()

            if use_wandb:
                wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
                wandb.log(
                    {
                        "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
                        "plots/feature_density_line_chart": wandb_histogram,
                        "sparsity/below_1e-5": (feature_sparsity < 1e-5).sum().item(),
                        "sparsity/below_1e-6": (feature_sparsity < 1e-6).sum().item(),
                    },
                    step=n_training_steps,
                )

            act_freq_scores = torch.zeros(
                sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device
            )
            n_frac_active_tokens = 0

        scheduler.step()
        optimizer.zero_grad()

        ghost_grad_neuron_mask = (
            n_forward_passes_since_fired > sparse_autoencoder.cfg.dead_feature_window
        ).bool()
        sae_in = activation_store.next_batch()

        # Forward and Backward Passes
        (
            sae_out,
            feature_acts,
            loss,
            mse_loss,
            l1_loss,
            ghost_grad_loss,
        ) = sparse_autoencoder(
            sae_in,
            ghost_grad_neuron_mask,
        )
        did_fire = (feature_acts > 0).float().sum(-2) > 0
        n_forward_passes_since_fired += 1
        n_forward_passes_since_fired[did_fire] = 0

        n_training_tokens += batch_size

        with torch.no_grad():
            # Calculate the sparsities, and add it to a list, calculate sparsity metrics
            act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            n_frac_active_tokens += batch_size
            feature_sparsity = act_freq_scores / n_frac_active_tokens

            if use_wandb and ((n_training_steps + 1) % wandb_log_frequency == 0):
                # metrics for currents acts
                l0 = (feature_acts > 0).float().sum(-1).mean()
                current_learning_rate = optimizer.param_groups[0]["lr"]

                per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
                total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
                explained_variance = 1 - per_token_l2_loss / total_variance

                wandb.log(
                    {
                        # losses
                        "losses/mse_loss": mse_loss.item(),
                        "losses/l1_loss": l1_loss.item()
                        / sparse_autoencoder.l1_coefficient,  # normalize by l1 coefficient
                        "losses/ghost_grad_loss": ghost_grad_loss.item(),
                        "losses/overall_loss": loss.item(),
                        # variance explained
                        "metrics/explained_variance": explained_variance.mean().item(),
                        "metrics/explained_variance_std": explained_variance.std().item(),
                        "metrics/l0": l0.item(),
                        # sparsity
                        "sparsity/mean_passes_since_fired": n_forward_passes_since_fired.mean().item(),
                        "sparsity/dead_features": ghost_grad_neuron_mask.sum().item(),
                        "details/n_training_tokens": n_training_tokens,
                        "details/current_learning_rate": current_learning_rate,
                    },
                    step=n_training_steps,
                )

            # record loss frequently, but not all the time.
            if use_wandb and ((n_training_steps + 1) % (wandb_log_frequency * 10) == 0):
                sparse_autoencoder.eval()
                run_evals(sparse_autoencoder, activation_store, model, n_training_steps)
                sparse_autoencoder.train()

            pbar.set_description(
                f"{n_training_steps}| MSE Loss {mse_loss.item():.3f} | L1 {l1_loss.item():.3f}"
            )
            pbar.update(batch_size)

        loss.backward()
        sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()

        # checkpoint if at checkpoint frequency
        if n_checkpoints > 0 and n_training_tokens > checkpoint_thresholds[0]:
            sparse_autoencoder.set_decoder_norm_to_unit_norm()
            path = f"{sparse_autoencoder.cfg.checkpoint_path}/{n_training_tokens}_{sparse_autoencoder.get_name()}.pt"
            log_feature_sparsity_path = f"{sparse_autoencoder.cfg.checkpoint_path}/{n_training_tokens}_{sparse_autoencoder.get_name()}_log_feature_sparsity.pt"
            sparse_autoencoder.save_model(path)
            log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()
            torch.save(log_feature_sparsity, log_feature_sparsity_path)
            checkpoint_thresholds.pop(0)
            if len(checkpoint_thresholds) == 0:
                n_checkpoints = 0
            if sparse_autoencoder.cfg.log_to_wandb:
                model_artifact = wandb.Artifact(
                    f"{sparse_autoencoder.get_name()}",
                    type="model",
                    metadata=dict(sparse_autoencoder.cfg.__dict__),
                )
                model_artifact.add_file(path)
                wandb.log_artifact(model_artifact)

                sparsity_artifact = wandb.Artifact(
                    f"{sparse_autoencoder.get_name()}_log_feature_sparsity",
                    type="log_feature_sparsity",
                    metadata=dict(sparse_autoencoder.cfg.__dict__),
                )
                sparsity_artifact.add_file(log_feature_sparsity_path)
                wandb.log_artifact(sparsity_artifact)

        n_training_steps += 1

    # save sae to checkpoints folder
    path = f"{sparse_autoencoder.cfg.checkpoint_path}/final_{sparse_autoencoder.get_name()}.pt"
    sparse_autoencoder.set_decoder_norm_to_unit_norm()
    sparse_autoencoder.save_model(path)

    if sparse_autoencoder.cfg.log_to_wandb:
        model_artifact = wandb.Artifact(
            f"{sparse_autoencoder.get_name()}",
            type="model",
            metadata=dict(sparse_autoencoder.cfg.__dict__),
        )
        model_artifact.add_file(path)
        wandb.log_artifact(model_artifact, aliases=["final_model"])

    log_feature_sparsity_path = f"{sparse_autoencoder.cfg.checkpoint_path}/final_{sparse_autoencoder.get_name()}_log_feature_sparsity.pt"
    torch.save(log_feature_sparsity, log_feature_sparsity_path)
    if sparse_autoencoder.cfg.log_to_wandb:
        sparsity_artifact = wandb.Artifact(
            f"{sparse_autoencoder.get_name()}_log_feature_sparsity",
            type="log_feature_sparsity",
            metadata=dict(sparse_autoencoder.cfg.__dict__),
        )
        sparsity_artifact.add_file(log_feature_sparsity_path)
        wandb.log_artifact(sparsity_artifact)

    return sparse_autoencoder
