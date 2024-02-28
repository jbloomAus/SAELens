import torch
from torch.optim import Adam
from tqdm import tqdm
from transformer_lens import HookedTransformer

import wandb
from sae_training.activations_store import ActivationsStore
from sae_training.evals import run_evals
from sae_training.geom_median.src.geom_median.torch import compute_geometric_median
from sae_training.optim import get_scheduler
from sae_training.sae_group import SAEGroup


def train_sae_on_language_model(
    model: HookedTransformer,
    sparse_autoencoders: SAEGroup,
    activation_store: ActivationsStore,
    batch_size: int = 1024,
    n_checkpoints: int = 0,
    feature_sampling_window: int = 1000,  # how many training steps between resampling the features / considiring neurons dead
    dead_feature_threshold: float = 1e-8,  # how infrequently a feature has to be active to be considered dead
    use_wandb: bool = False,
    wandb_log_frequency: int = 50,
):
    total_training_tokens = sparse_autoencoders.cfg.total_training_tokens
    total_training_steps = total_training_tokens // batch_size
    n_training_steps = 0
    n_training_tokens = 0

    if n_checkpoints > 0:
        checkpoint_thresholds = list(
            range(0, total_training_tokens, total_training_tokens // n_checkpoints)
        )[1:]

    # things to store for each sae:
    # act_freq_scores, n_forward_passes_since_fired, n_frac_active_tokens, optimizer, scheduler,
    num_saes = len(sparse_autoencoders)
    # track active features

    act_freq_scores = [
        torch.zeros(
            sparse_autoencoders.cfg.d_sae, device=sparse_autoencoders.cfg.device
        )
        for _ in range(num_saes)
    ]
    n_forward_passes_since_fired = [
        torch.zeros(
            sparse_autoencoders.cfg.d_sae, device=sparse_autoencoders.cfg.device
        )
        for _ in range(num_saes)
    ]
    n_frac_active_tokens = [0 for _ in range(num_saes)]

    optimizer = [Adam(sae.parameters(), lr=sae.cfg.lr) for sae in sparse_autoencoders]
    scheduler = [
        get_scheduler(
            sae.cfg.lr_scheduler_name,
            optimizer=opt,
            warm_up_steps=sae.cfg.lr_warm_up_steps,
            training_steps=total_training_steps,
            lr_end=sae.cfg.lr / 10,  # heuristic for now.
        )
        for sae, opt in zip(sparse_autoencoders, optimizer)
    ]

    all_layers = sparse_autoencoders.cfg.hook_point_layer
    if not isinstance(all_layers, list):
        all_layers = [all_layers]

    # compute the geometric median of the activations of each layer

    geometric_medians = []
    for layer_id in range(len(all_layers)):
        layer_acts = activation_store.storage_buffer.detach().cpu()[:, layer_id, :]

        median = compute_geometric_median(
            layer_acts, skip_typechecks=True, maxiter=100, per_component=False
        ).median
        geometric_medians.append(median)

    for sae in sparse_autoencoders:
        hyperparams = sae.cfg
        sae_layer_id = all_layers.index(hyperparams.hook_point_layer)

        # extract all activations at a certain layer and use for sae initialization
        sae.initialize_b_dec_with_precalculated(geometric_medians[sae_layer_id])
        sae.train()

    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:
        # Do a training step.
        layer_acts = activation_store.next_batch()
        n_training_tokens += batch_size

        for (
            i,
            (sparse_autoencoder),
        ) in enumerate(sparse_autoencoders):
            hyperparams = sparse_autoencoder.cfg
            layer_id = all_layers.index(hyperparams.hook_point_layer)
            sae_in = layer_acts[:, layer_id, :]

            sparse_autoencoder.train()
            # Make sure the W_dec is still zero-norm
            sparse_autoencoder.set_decoder_norm_to_unit_norm()

            # log and then reset the feature sparsity every feature_sampling_window steps
            if (n_training_steps + 1) % feature_sampling_window == 0:
                feature_sparsity = act_freq_scores[i] / n_frac_active_tokens[i]
                log_feature_sparsity = (
                    torch.log10(feature_sparsity + 1e-10).detach().cpu()
                )

                if use_wandb:
                    suffix = wandb_log_suffix(sparse_autoencoders.cfg, hyperparams)
                    wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
                    wandb.log(
                        {
                            f"metrics/mean_log10_feature_sparsity{suffix}": log_feature_sparsity.mean().item(),
                            f"plots/feature_density_line_chart{suffix}": wandb_histogram,
                            f"sparsity/below_1e-5{suffix}": (feature_sparsity < 1e-5)
                            .sum()
                            .item(),
                            f"sparsity/below_1e-6{suffix}": (feature_sparsity < 1e-6)
                            .sum()
                            .item(),
                        },
                        step=n_training_steps,
                    )

                act_freq_scores[i] = torch.zeros(
                    sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device
                )
                n_frac_active_tokens[i] = 0

            scheduler[i].step()
            optimizer[i].zero_grad()

            ghost_grad_neuron_mask = (
                n_forward_passes_since_fired[i]
                > sparse_autoencoder.cfg.dead_feature_window
            ).bool()

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
            n_forward_passes_since_fired[i] += 1
            n_forward_passes_since_fired[i][did_fire] = 0

            with torch.no_grad():
                # Calculate the sparsities, and add it to a list, calculate sparsity metrics
                act_freq_scores[i] += (feature_acts.abs() > 0).float().sum(0)
                n_frac_active_tokens[i] += batch_size
                feature_sparsity = act_freq_scores[i] / n_frac_active_tokens[i]

                if use_wandb and ((n_training_steps + 1) % wandb_log_frequency == 0):
                    # metrics for currents acts
                    l0 = (feature_acts > 0).float().sum(-1).mean()
                    current_learning_rate = optimizer[i].param_groups[0]["lr"]

                    per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
                    total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
                    explained_variance = 1 - per_token_l2_loss / total_variance

                    suffix = wandb_log_suffix(sparse_autoencoders.cfg, hyperparams)
                    wandb.log(
                        {
                            # losses
                            f"losses/mse_loss{suffix}": mse_loss.item(),
                            f"losses/l1_loss{suffix}": l1_loss.item()
                            / sparse_autoencoder.l1_coefficient,  # normalize by l1 coefficient
                            f"losses/ghost_grad_loss{suffix}": ghost_grad_loss.item(),
                            f"losses/overall_loss{suffix}": loss.item(),
                            # variance explained
                            f"metrics/explained_variance{suffix}": explained_variance.mean().item(),
                            f"metrics/explained_variance_std{suffix}": explained_variance.std().item(),
                            f"metrics/l0{suffix}": l0.item(),
                            # sparsity
                            f"sparsity/mean_passes_since_fired{suffix}": n_forward_passes_since_fired[
                                i
                            ]
                            .mean()
                            .item(),
                            f"sparsity/dead_features{suffix}": ghost_grad_neuron_mask.sum().item(),
                            f"details/n_training_tokens{suffix}": n_training_tokens,
                            f"details/current_learning_rate{suffix}": current_learning_rate,
                        },
                        step=n_training_steps,
                    )

                # record loss frequently, but not all the time.
                if use_wandb and (
                    (n_training_steps + 1) % (wandb_log_frequency * 10) == 0
                ):
                    sparse_autoencoder.eval()
                    suffix = wandb_log_suffix(sparse_autoencoder.cfg, hyperparams)
                    run_evals(
                        sparse_autoencoder,
                        activation_store,
                        model,
                        n_training_steps,
                        suffix=suffix,
                    )
                    sparse_autoencoder.train()

            loss.backward()
            sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
            optimizer[i].step()

        # checkpoint if at checkpoint frequency
        if n_checkpoints > 0 and n_training_tokens > checkpoint_thresholds[0]:
            path = f"{sparse_autoencoders.cfg.checkpoint_path}/{n_training_tokens}_{sparse_autoencoders.get_name()}.pt"
            for sae in sparse_autoencoders:
                sae.set_decoder_norm_to_unit_norm()
            sparse_autoencoders.save_model(path)

            log_feature_sparsity_path = f"{sparse_autoencoders.cfg.checkpoint_path}/{n_training_tokens}_{sparse_autoencoders.get_name()}_log_feature_sparsity.pt"
            log_feature_sparsity = []
            for sae_id in range(len(sparse_autoencoders)):
                feature_sparsity = (
                    act_freq_scores[sae_id] / n_frac_active_tokens[sae_id]
                )
                log_feature_sparsity.append(
                    torch.log10(feature_sparsity + 1e-10).detach().cpu()
                )
            torch.save(log_feature_sparsity, log_feature_sparsity_path)

            checkpoint_thresholds.pop(0)
            if len(checkpoint_thresholds) == 0:
                n_checkpoints = 0
            if sparse_autoencoders.cfg.log_to_wandb:
                model_artifact = wandb.Artifact(
                    f"{sparse_autoencoders.get_name()}",
                    type="model",
                    metadata=dict(sparse_autoencoders.cfg.__dict__),
                )
                model_artifact.add_file(path)
                wandb.log_artifact(model_artifact)

                sparsity_artifact = wandb.Artifact(
                    f"{sparse_autoencoders.get_name()}_log_feature_sparsity",
                    type="log_feature_sparsity",
                    metadata=dict(sparse_autoencoders.cfg.__dict__),
                )
                sparsity_artifact.add_file(log_feature_sparsity_path)
                wandb.log_artifact(sparsity_artifact)

                ###############

        n_training_steps += 1
        pbar.set_description(
            f"{n_training_steps}| MSE Loss {mse_loss.item():.3f} | L1 {l1_loss.item():.3f}"
        )
        pbar.update(batch_size)

    # save sae group to checkpoints folder
    path = f"{sparse_autoencoders.cfg.checkpoint_path}/final_{sparse_autoencoders.get_name()}.pt"
    for sae in sparse_autoencoders:
        sae.set_decoder_norm_to_unit_norm()
    sparse_autoencoders.save_model(path)

    if sparse_autoencoders.cfg.log_to_wandb:
        model_artifact = wandb.Artifact(
            f"{sparse_autoencoders.get_name()}",
            type="model",
            metadata=dict(sparse_autoencoders.cfg.__dict__),
        )
        model_artifact.add_file(path)
        wandb.log_artifact(model_artifact, aliases=["final_model"])

    # need to fix this
    log_feature_sparsity_path = f"{sparse_autoencoders.cfg.checkpoint_path}/final_{sparse_autoencoders.get_name()}_log_feature_sparsity.pt"
    log_feature_sparsity = []
    for sae_id in range(len(sparse_autoencoders)):
        feature_sparsity = act_freq_scores[sae_id] / n_frac_active_tokens[sae_id]
        log_feature_sparsity.append(
            torch.log10(feature_sparsity + 1e-10).detach().cpu()
        )
    torch.save(log_feature_sparsity, log_feature_sparsity_path)

    if sparse_autoencoders.cfg.log_to_wandb:
        sparsity_artifact = wandb.Artifact(
            f"{sparse_autoencoders.get_name()}_log_feature_sparsity",
            type="log_feature_sparsity",
            metadata=dict(sparse_autoencoders.cfg.__dict__),
        )
        sparsity_artifact.add_file(log_feature_sparsity_path)
        wandb.log_artifact(sparsity_artifact)

    return sparse_autoencoders


def wandb_log_suffix(cfg, hyperparams):
    # Create a mapping from cfg list keys to their corresponding hyperparams attributes
    key_mapping = {
        "hook_point_layer": "layer",
        "l1_coefficient": "coeff",
        "lp_norm": "l",
        "lr": "lr",
    }

    # Generate the suffix by iterating over the keys that have list values in cfg
    suffix = "".join(
        f"_{key_mapping.get(key, key)}{getattr(hyperparams, key, '')}"
        for key, value in vars(cfg).items()
        if isinstance(value, list)
    )
    return suffix
