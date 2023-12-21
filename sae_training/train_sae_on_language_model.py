from functools import partial

import numpy as np
import plotly_express as px
import torch
from torch.optim import Adam
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name

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
    steps_before_reset = 0
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


        if (feature_sampling_method.lower()=="anthropic") and ((n_training_steps + 1) % dead_feature_window == 0):
            
            feature_sparsity = act_freq_scores / n_frac_active_tokens
            dead_neuron_indices = (feature_sparsity < sparse_autoencoder.cfg.dead_feature_threshold).nonzero(as_tuple=False)[:, 0]
            sparse_autoencoder.resample_neurons_anthropic(
                dead_neuron_indices, 
                model,
                optimizer, 
                activation_store
            )

            if use_wandb:
                wandb.log(
                    {
                        "metrics/n_resampled_neurons": len(dead_neuron_indices),
                    },
                    step=n_training_steps,
                )
            
            # for now, we'll hardcode this.
            current_lr = scheduler.get_last_lr()[0]
            reduced_lr = current_lr * 0.1
            increment = (current_lr - reduced_lr) / 1000
            optimizer.param_groups[0]['lr'] = reduced_lr
            steps_before_reset = 1000
            

        # Resample dead neurons
        if (feature_sampling_method == "l2") and ((n_training_steps + 1) % dead_feature_window == 0):
            print("no l2 resampling currently. Please use anthropic resampling")
            # # Get the fraction of neurons active in the previous window
            # feature_sparsity = act_freq_scores / n_frac_active_tokens
            # is_dead = (feature_sparsity < sparse_autoencoder.cfg.dead_feature_threshold)
            
            # # if standard resampling <- do this
            # n_resampled_neurons = sparse_autoencoder.resample_neurons(
            #     activation_store.next_batch(), 
            #     feature_sparsity, 
            #     feature_reinit_scale,
            #     optimizer
            # )
            # # for all the dead neurons, set the feature sparsity to the dead feature threshold
            # act_freq_scores[is_dead] = sparse_autoencoder.cfg.dead_feature_threshold * n_frac_active_tokens
            # if n_resampled_neurons > 0:
            #     print(f"Resampled {n_resampled_neurons} neurons")
            # if use_wandb:
            #     wandb.log(
            #         {
            #             "metrics/n_resampled_neurons": n_resampled_neurons,
            #         },
            #         step=n_training_steps,
            #     )
            # n_resampled_neurons = 0


        if (steps_before_reset > 0) and n_training_steps > 0:
            steps_before_reset -= 1
            optimizer.param_groups[0]['lr'] += increment
            if steps_before_reset == 0:
                optimizer.param_groups[0]['lr'] = current_lr
        else:
            scheduler.step()
    
        optimizer.zero_grad()
        
        # Forward and Backward Passes
        sae_in = activation_store.next_batch()
        sae_out, feature_acts, loss, mse_loss, l1_loss = sparse_autoencoder(sae_in)
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
                total_variance = sae_in.pow(2).sum(-1)
                explained_variance = 1 - per_token_l2_loss/total_variance
                
                wandb.log(
                    {
                        # losses
                        "losses/mse_loss": mse_loss.item(),
                        "losses/l1_loss": l1_loss.item(),
                        "losses/overall_loss": loss.item(),
                        # variance explained
                        "metrics/explained_variance": explained_variance.mean().item(),
                        "metrics/explained_variance_std": explained_variance.std().item(),
                        "metrics/l0": l0.item(),
                        # sparsity
                        "sparsity/below_1e-5": (feature_sparsity < 1e-5)
                        .float()
                        .mean()
                        .item(),
                        "sparsity/below_1e-6": (feature_sparsity < 1e-6)
                        .float()
                        .mean()
                        .item(),
                        "sparsity/dead_features": (
                            feature_sparsity < dead_feature_threshold
                        )
                        .float()
                        .mean()
                        .item(),
                        "details/n_training_tokens": n_training_tokens,
                        "details/current_learning_rate": current_learning_rate,
                    },
                    step=n_training_steps,
                )

            # record loss frequently, but not all the time.
            if use_wandb and ((n_training_steps + 1) % (wandb_log_frequency * 10) == 0):
                run_evals(sparse_autoencoder, activation_store, model, n_training_steps)
                
                log_feature_sparsity = torch.log10(feature_sparsity + 1e-10).detach().cpu()
                
                # sparsity_line_chart = px.scatter(
                #     y = log_feature_sparsity,
                #     title="Feature Sparsity",
                #     labels={"y": "log10(sparsity)", "x": "FeatureID"},
                #     range_y=[-8, 0],
                #     marginal_y="histogram",
                # )
                wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
                wandb.log(
                    {   
                        "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
                        "plots/feature_density_line_chart": wandb_histogram,
                    },
                    step=n_training_steps,
                )
                


            pbar.set_description(
                f"{n_training_steps}| MSE Loss {mse_loss.item():.3f} | L1 {l1_loss.item():.3f}"
            )
            pbar.update(batch_size)

        loss.backward()
        sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()


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
def run_evals(sparse_autoencoder: SparseAutoencoder, activation_store: ActivationsStore, model: HookedTransformer, n_training_steps: int):
    
    hook_point = sparse_autoencoder.cfg.hook_point
    hook_point_layer = sparse_autoencoder.cfg.hook_point_layer
    hook_point_head_index = sparse_autoencoder.cfg.hook_point_head_index
    
     ### Evals
    eval_tokens = activation_store.get_batch_tokens()
    
    # Get Reconstruction Score
    recons_score, ntp_loss, recons_loss, zero_abl_loss = get_recons_loss(sparse_autoencoder, model, activation_store, eval_tokens)
    
    # get cache
    _, cache = model.run_with_cache(eval_tokens, prepend_bos=False, names_filter=[get_act_name("pattern", hook_point_layer), hook_point])
    
    # get act
    if sparse_autoencoder.cfg.hook_point_head_index is not None:
        original_act = cache[sparse_autoencoder.cfg.hook_point][:,:,sparse_autoencoder.cfg.hook_point_head_index]
    else:
        original_act = cache[sparse_autoencoder.cfg.hook_point]
        
    sae_out, feature_acts, _, _, _ = sparse_autoencoder(
        original_act
    )
    patterns_original = cache[get_act_name("pattern", hook_point_layer)][:,hook_point_head_index].detach().cpu()
    del cache
    
    if "cuda" in str(model.cfg.device):
        torch.cuda.empty_cache()
    
    l2_norm_in = torch.norm(original_act, dim=-1)
    l2_norm_out = torch.norm(sae_out, dim=-1)
    l2_norm_ratio = l2_norm_out / l2_norm_in
    
    wandb.log(
        {

            # l2 norms
            "metrics/l2_norm": l2_norm_out.mean().item(),
            "metrics/l2_ratio": l2_norm_ratio.mean().item(),
            
            # CE Loss
            "metrics/CE_loss_score": recons_score,
            "metrics/ce_loss_without_sae": ntp_loss,
            "metrics/ce_loss_with_sae": recons_loss,
            "metrics/ce_loss_with_ablation": zero_abl_loss,
            
        },
        step=n_training_steps,
    )

    # get attn when using reconstructed activations
    with model.hooks(fwd_hooks=[(hook_point, partial(replacement_hook, encoder=sparse_autoencoder))]):
        _, new_cache = model.run_with_cache(eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)])
        patterns_reconstructed = new_cache[get_act_name("pattern", hook_point_layer)][:,hook_point_head_index].detach().cpu()
        del new_cache
        
    # get attn when using reconstructed activations
    with model.hooks(fwd_hooks=[(hook_point, partial(zero_ablate_hook))]):
        _, zero_ablation_cache = model.run_with_cache(eval_tokens, names_filter=[get_act_name("pattern", hook_point_layer)])
        patterns_ablation = zero_ablation_cache[get_act_name("pattern", hook_point_layer)][:,hook_point_head_index].detach().cpu()
        del zero_ablation_cache
        
        
    # Visualizations to show L0 / MSE distributions
    # l0 = (feature_acts > 0).float().sum(-1)
    # per_token_l2_loss = (sae_out - original_act).pow(2).sum(dim=-1).squeeze()
    
    # fig = px.scatter(
    #     x = per_token_l2_loss.flatten().cpu().numpy(),
    #     y = l0.flatten().cpu().numpy(),
    #     color = np.arange(per_token_l2_loss.shape[1]).repeat(per_token_l2_loss.shape[0]),
    #     opacity=0.5,
    #     labels = {"color": "position", "x": "MSE Loss", "y": "L0"},
    #     title = "L0 vs MSE Loss",
    #     marginal_x="histogram",
    #     marginal_y="histogram",
    # )
    # wandb.log({"plots/l0_vs_mse_loss": wandb.Plotly(fig)}, step = n_training_steps)
    
    # fig = px.scatter(
    #     x =  per_token_l2_loss.flatten().cpu().numpy(),
    #     y = l2_norm_in.flatten().cpu().numpy(),
    #     color = np.arange(per_token_l2_loss.shape[1]).repeat(per_token_l2_loss.shape[0]),
    #     opacity=0.5,
    #     labels={"color": "position", "x": "MSE Loss", "y": "L2 Norm"},
    #     title = "L2 Norm vs MSE Loss",
    #     marginal_x="histogram",
    #     marginal_y="histogram",
    # )
    # wandb.log({"plots/l2_norm_vs_mse_loss": wandb.Plotly(fig)}, step = n_training_steps)

    # if dealing with a head SAE, do the head metrics.
    if sparse_autoencoder.cfg.hook_point_head_index:
        
        # show patterns before/after
        # fig_patterns_original = px.imshow(patterns_original[0].numpy(), title="original attn scores",
        #     color_continuous_midpoint=0, color_continuous_scale="RdBu")
        # fig_patterns_original.update_layout(coloraxis_showscale=False)         # hide colorbar 
        # wandb.log({"attention/patterns_original": wandb.Plotly(fig_patterns_original)}, step = n_training_steps)
        # fig_patterns_reconstructed = px.imshow(patterns_reconstructed[0].numpy(), title="reconstructed attn scores",
        #         color_continuous_midpoint=0, color_continuous_scale="RdBu")
        # fig_patterns_reconstructed.update_layout(coloraxis_showscale=False)         # hide colorbar
        # wandb.log({"attention/patterns_reconstructed": wandb.Plotly(fig_patterns_reconstructed)}, step = n_training_steps)
        
        kl_result_reconstructed = kl_divergence_attention(patterns_original, patterns_reconstructed)
        kl_result_reconstructed = kl_result_reconstructed.sum(dim=-1).numpy()
        # print(kl_result.mean().item())
        # px.imshow(kl_result, title="KL Divergence", width=800, height=800,
        #       color_continuous_midpoint=0, color_continuous_scale="RdBu").show()
        # px.histogram(kl_result.flatten()).show()
        # px.line(kl_result.mean(0), title="KL Divergence by Position").show()
        
        kl_result_ablation = kl_divergence_attention(patterns_original, patterns_ablation)
        kl_result_ablation = kl_result_ablation.sum(dim=-1).numpy()
        # print(kl_result.mean().item())
        # # px.imshow(kl_result, title="KL Divergence", width=800, height=800,
        # #       color_continuous_midpoint=0, color_continuous_scale="RdBu").show()
        # px.histogram(kl_result.flatten()).show()
        # px.line(kl_result.mean(0), title="KL Divergence by Position").show()
    
        wandb.log(
            {

              "metrics/kldiv_reconstructed": kl_result_reconstructed.mean().item(),
              "metrics/kldiv_ablation": kl_result_ablation.mean().item(),
                
            },
            step=n_training_steps,
        )

@torch.no_grad()
def get_recons_loss(sparse_autoencder, model, activation_store, batch_tokens):
    hook_point = activation_store.cfg.hook_point
    loss = model(batch_tokens, return_type="loss")

    recons_loss = model.run_with_hooks(
        batch_tokens,
        return_type="loss",
        fwd_hooks=[(hook_point, partial(replacement_hook, encoder=sparse_autoencder))],
    )

    zero_abl_loss = model.run_with_hooks(
        batch_tokens, return_type="loss", fwd_hooks=[(hook_point, zero_ablate_hook)]
    )

    score = (zero_abl_loss - recons_loss) / (zero_abl_loss - loss)

    return score, loss, recons_loss, zero_abl_loss


def replacement_hook(mlp_post, hook, encoder):
    activations =  encoder(mlp_post)[0].to(mlp_post.dtype)
    return activations


def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1]).to(mlp_post.dtype)
    return mlp_post


def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.0
    return mlp_post


def kl_divergence_attention(y_true, y_pred):

    # Compute log probabilities for KL divergence
    log_y_true = torch.log2(y_true + 1e-10)
    log_y_pred = torch.log2(y_pred + 1e-10)

    return y_true * (log_y_true - log_y_pred)