import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_lens.sae import SAE


@torch.no_grad()
def get_feature_property_df(sae: SAE, feature_sparsity: torch.Tensor):
    """
    feature_property_df = get_feature_property_df(sae, log_feature_density.cpu())
    """

    W_dec_normalized = (
        sae.W_dec.cpu()
    )  # / sparse_autoencoder.W_dec.cpu().norm(dim=-1, keepdim=True)
    W_enc_normalized = (sae.W_enc.cpu() / sae.W_enc.cpu().norm(dim=-1, keepdim=True)).T

    d_e_projection = (W_dec_normalized * W_enc_normalized).sum(-1)
    b_dec_projection = sae.b_dec.cpu() @ W_dec_normalized.T

    temp_df = pd.DataFrame(
        {
            "log_feature_sparsity": feature_sparsity + 1e-10,
            "d_e_projection": d_e_projection,
            # "d_e_projection_normalized": d_e_projection_normalized,
            "b_enc": sae.b_enc.detach().cpu(),
            "b_dec_projection": b_dec_projection,
            "feature": list(range(sae.cfg.d_sae)),  # type: ignore
            "dead_neuron": (feature_sparsity < -9).cpu(),
        }
    )

    return temp_df


@torch.no_grad()
def get_stats_df(projection: torch.Tensor):
    """
    Returns a dataframe with the mean, std, skewness and kurtosis of the projection
    """
    mean = projection.mean(dim=1, keepdim=True)
    diffs = projection - mean
    var = (diffs**2).mean(dim=1, keepdim=True)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0), dim=1)
    kurtosis = torch.mean(torch.pow(zscores, 4.0), dim=1)

    stats_df = pd.DataFrame(
        {
            "feature": range(len(skews)),
            "mean": mean.numpy().squeeze(),
            "std": std.numpy().squeeze(),
            "skewness": skews.numpy(),
            "kurtosis": kurtosis.numpy(),
        }
    )

    return stats_df


@torch.no_grad()
def get_all_stats_dfs(
    gpt2_small_sparse_autoencoders: dict[str, SAE],  # [hook_point, sae]
    gpt2_small_sae_sparsities: dict[str, torch.Tensor],  # [hook_point, sae]
    model: HookedTransformer,
    cosine_sim: bool = False,
):
    stats_dfs = []
    pbar = tqdm(gpt2_small_sparse_autoencoders.keys())
    for key in pbar:
        layer = int(key.split(".")[1])
        sparse_autoencoder = gpt2_small_sparse_autoencoders[key]
        pbar.set_description(f"Processing layer {sparse_autoencoder.cfg.hook_name}")
        W_U_stats_df_dec, _ = get_W_U_W_dec_stats_df(
            sparse_autoencoder.W_dec.cpu(), model, cosine_sim
        )
        log_feature_sparsity = gpt2_small_sae_sparsities[key].detach().cpu()
        W_U_stats_df_dec["log_feature_sparsity"] = log_feature_sparsity
        W_U_stats_df_dec["layer"] = layer + (1 if "post" in key else 0)
        stats_dfs.append(W_U_stats_df_dec)

    W_U_stats_df_dec_all_layers = pd.concat(stats_dfs, axis=0)
    return W_U_stats_df_dec_all_layers


@torch.no_grad()
def get_W_U_W_dec_stats_df(
    W_dec: torch.Tensor, model: HookedTransformer, cosine_sim: bool = False
) -> tuple[pd.DataFrame, torch.Tensor]:
    W_U = model.W_U.detach().cpu()
    if cosine_sim:
        W_U = W_U / W_U.norm(dim=0, keepdim=True)
    dec_projection_onto_W_U = W_dec @ W_U
    W_U_stats_df = get_stats_df(dec_projection_onto_W_U)
    return W_U_stats_df, dec_projection_onto_W_U
