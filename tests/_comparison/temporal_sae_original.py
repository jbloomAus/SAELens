"""Original TemporalSAE implementation from temporal_sae_setup.ipynb

This is used for comparison testing to ensure the SAELens implementation
matches the original implementation.
"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


def get_attention(query: th.Tensor, key: th.Tensor) -> th.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = th.zeros(L, S, dtype=query.dtype, device=query.device)
    temp_mask = th.ones(L, S, dtype=th.bool, device=query.device).tril(diagonal=0)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(query.dtype)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    return th.softmax(attn_weight, dim=-1)


### Manual Attention Implementation
class ManualAttention(nn.Module):
    """
    Manual implementation to allow tinkering with the attention mechanism.
    """

    def __init__(
        self,
        dimin: int,
        n_heads: int = 4,
        bottleneck_factor: int = 64,
        bias_k: bool = True,
        bias_q: bool = True,
        bias_v: bool = True,
        bias_o: bool = True,
    ) -> None:
        super().__init__()
        assert dimin % (bottleneck_factor * n_heads) == 0

        # attention heads
        self.n_heads = n_heads
        self.n_embds = dimin // bottleneck_factor  # n_heads
        self.dimin = dimin

        # key, query, value projections for all heads, but in a batch
        self.k_ctx = nn.Linear(dimin, self.n_embds, bias=bias_k)
        self.q_target = nn.Linear(dimin, self.n_embds, bias=bias_q)
        self.v_ctx = nn.Linear(dimin, dimin, bias=bias_v)

        # output projection
        self.c_proj = nn.Linear(dimin, dimin, bias=bias_o)

        # Normalize to match scale with representations
        with th.no_grad():
            scaling = 1 / math.sqrt(self.n_embds // self.n_heads)
            self.k_ctx.weight.copy_(
                scaling
                * self.k_ctx.weight
                / (1e-6 + th.linalg.norm(self.k_ctx.weight, dim=1, keepdim=True))
            )
            self.q_target.weight.copy_(
                scaling
                * self.q_target.weight
                / (1e-6 + th.linalg.norm(self.q_target.weight, dim=1, keepdim=True))
            )

            scaling = 1 / math.sqrt(self.dimin // self.n_heads)
            self.v_ctx.weight.copy_(
                scaling
                * self.v_ctx.weight
                / (1e-6 + th.linalg.norm(self.v_ctx.weight, dim=1, keepdim=True))
            )

            scaling = 1 / math.sqrt(self.dimin)
            self.c_proj.weight.copy_(
                scaling
                * self.c_proj.weight
                / (1e-6 + th.linalg.norm(self.c_proj.weight, dim=1, keepdim=True))
            )

    def forward(
        self, x_ctx: th.Tensor, x_target: th.Tensor, get_attn_map: bool = False
    ) -> tuple[th.Tensor, th.Tensor | None]:
        """
        Compute projective attention output
        """
        # Compute key and value projections from context representations
        k = self.k_ctx(x_ctx)
        v = self.v_ctx(x_ctx)

        # Compute query projection from target representations
        q = self.q_target(x_target)

        # Split into heads
        B, T, _ = x_ctx.size()
        k = k.view(B, T, self.n_heads, self.n_embds // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.n_embds // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.dimin // self.n_heads).transpose(1, 2)

        # Attn map
        attn_map: th.Tensor | None = None
        if get_attn_map:
            attn_map = get_attention(query=q, key=k)
            th.cuda.empty_cache()

        # Scaled dot-product attention
        attn_output = th.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0, is_causal=True
        )

        # Reshape, project back to original dimension
        d_target = self.c_proj(
            attn_output.transpose(1, 2).contiguous().view(B, T, self.dimin)
        )  # [batch, length, dimin]

        if get_attn_map:
            return d_target, attn_map
        return d_target, None


class TemporalSAE(th.nn.Module):
    def __init__(
        self,
        dimin: int = 2,
        width: int = 5,
        n_heads: int = 8,
        sae_diff_type: str = "relu",
        kval_topk: int | None = None,
        tied_weights: bool = True,
        n_attn_layers: int = 1,
        bottleneck_factor: int = 64,
        activation_scaling_factor: float = 1.0,
    ) -> None:
        """
        dimin: (int)
            input dimension
        width: (int)
            width of the encoder
        n_heads: (int)
            number of attention heads
        sae_diff_type: (str)
            type of sae to express the per-token difference
        kval_topk: (int)
            k in topk sae_diff_type
        n_attn_layers: (int)
            number of attention layers
        """
        super().__init__()
        self.sae_type = "temporal"
        self.width = width
        self.dimin = dimin
        self.eps = 1e-6
        self.lam = 1 / (4 * dimin)
        self.tied_weights = tied_weights
        self.activation_scaling_factor = activation_scaling_factor

        ## Attention parameters
        self.n_attn_layers = n_attn_layers
        self.attn_layers = nn.ModuleList(
            [
                ManualAttention(
                    dimin=width,
                    n_heads=n_heads,
                    bottleneck_factor=bottleneck_factor,
                    bias_k=True,
                    bias_q=True,
                    bias_v=True,
                    bias_o=True,
                )
                for _ in range(n_attn_layers)
            ]
        )

        ## Dictionary parameters
        self.D = nn.Parameter(th.randn((width, dimin)))  # N(0,1) init
        self.b = nn.Parameter(th.zeros((1, dimin)))
        if not tied_weights:
            self.E = nn.Parameter(th.randn((dimin, width)))  # N(0,1) init

        ## SAE-specific parameters
        self.sae_diff_type = sae_diff_type
        self.kval_topk = kval_topk if sae_diff_type == "topk" else None

    def forward(
        self,
        x_input: th.Tensor,
        return_graph: bool = False,
        inf_k: int | None = None,
    ) -> tuple[th.Tensor, dict[str, th.Tensor | None]]:
        B, L, _ = x_input.size()
        E = self.D.T if self.tied_weights else self.E

        ### Define context and target ###
        x_input = x_input * self.activation_scaling_factor
        x_input = x_input - self.b

        ### Tracking variables ###
        attn_graphs = []

        ### Predictable part ###
        z_pred = th.zeros(
            (B, L, self.width), device=x_input.device, dtype=x_input.dtype
        )
        for attn_layer in self.attn_layers:
            z_input = F.relu(th.matmul(x_input * self.lam, E))  # [batch, length, width]
            z_ctx = th.cat(
                (th.zeros_like(z_input[:, :1, :]), z_input[:, :-1, :].clone()), dim=1
            )  # [batch, length, width]

            # Compute codes using attention
            z_pred_, attn_graphs_ = attn_layer(
                z_ctx, z_input, get_attn_map=return_graph
            )

            # Take back to input space
            z_pred_ = F.relu(z_pred_)
            Dz_pred_ = th.matmul(z_pred_, self.D)
            Dz_norm_ = Dz_pred_.norm(dim=-1, keepdim=True) + self.eps

            # Compute projection
            proj_scale = (Dz_pred_ * x_input).sum(dim=-1, keepdim=True) / Dz_norm_.pow(
                2
            )

            # Add the projection to the reconstructed
            z_pred = z_pred + (z_pred_ * proj_scale)

            # Remove the projection from the input
            x_input = x_input - proj_scale * Dz_pred_  # [batch, length, width]

            # Add the attention graph if return_graph is True
            if return_graph:
                attn_graphs.append(attn_graphs_)

        ### Novel part (identified using the residual target signal) ###
        z_novel: th.Tensor
        if self.sae_diff_type == "relu":
            z_novel = F.relu(th.matmul(x_input * self.lam, E))

        elif self.sae_diff_type == "topk":
            kval = self.kval_topk if inf_k is None else inf_k
            assert (
                kval is not None
            ), "kval_topk must be set when using topk sae_diff_type"
            z_novel = F.relu(th.matmul(x_input * self.lam, E))
            _, topk_indices = th.topk(z_novel, kval, dim=-1)
            mask = th.zeros_like(z_novel)
            mask.scatter_(-1, topk_indices, 1)
            z_novel = z_novel * mask

        else:  # self.sae_diff_type == "nullify"
            z_novel = th.zeros_like(z_pred)

        ### Reconstruction ###
        x_recons = (
            th.matmul(z_novel + z_pred, self.D) + self.b
        )  # [batch, length, dimin]
        x_recons = x_recons / self.activation_scaling_factor

        ### Compute the predicted vs. novel reconstructions, sans the bias (allows to check context / dictionary's value) ###
        with th.no_grad():
            x_pred_recons = th.matmul(z_pred, self.D) / self.activation_scaling_factor
            x_novel_recons = th.matmul(z_novel, self.D) / self.activation_scaling_factor

        ### Return the dictionary ###
        results_dict = {
            "novel_codes": z_novel,
            "novel_recons": x_novel_recons,
            "pred_codes": z_pred,
            "pred_recons": x_pred_recons,
            "attn_graphs": th.stack(attn_graphs, dim=1) if return_graph else None,
        }

        return x_recons, results_dict
