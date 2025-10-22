"""Verify that SAELens TemporalSAE gives same outputs as original implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import math
import os
from pathlib import Path

# Import SAELens version
from sae_lens import SAE

print("=" * 80)
print("TEMPORAL SAE EQUIVALENCE VERIFICATION")
print("=" * 80)

# ============================================================================
# Original Implementation from notebook
# ============================================================================

def get_attention(query, key):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    attn_bias.to(query.dtype)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight


class ManualAttention(nn.Module):
    def __init__(self, dimin, n_heads=4, bottleneck_factor=64, bias_k=True, bias_q=True, bias_v=True, bias_o=True):
        super().__init__()
        assert dimin % (bottleneck_factor * n_heads) == 0

        self.n_heads = n_heads
        self.n_embds = dimin // bottleneck_factor
        self.dimin = dimin

        self.k_ctx = nn.Linear(dimin, self.n_embds, bias=bias_k)
        self.q_target = nn.Linear(dimin, self.n_embds, bias=bias_q)
        self.v_ctx = nn.Linear(dimin, dimin, bias=bias_v)
        self.c_proj = nn.Linear(dimin, dimin, bias=bias_o)

        with torch.no_grad():
            scaling = 1 / math.sqrt(self.n_embds // self.n_heads)
            self.k_ctx.weight.copy_(scaling * self.k_ctx.weight / (1e-6 + torch.linalg.norm(self.k_ctx.weight, dim=1, keepdim=True)))
            self.q_target.weight.copy_(scaling * self.q_target.weight / (1e-6 + torch.linalg.norm(self.q_target.weight, dim=1, keepdim=True)))

            scaling = 1 / math.sqrt(self.dimin // self.n_heads)
            self.v_ctx.weight.copy_(scaling * self.v_ctx.weight / (1e-6 + torch.linalg.norm(self.v_ctx.weight, dim=1, keepdim=True)))

            scaling = 1 / math.sqrt(self.dimin)
            self.c_proj.weight.copy_(scaling * self.c_proj.weight / (1e-6 + torch.linalg.norm(self.c_proj.weight, dim=1, keepdim=True)))

    def forward(self, x_ctx, x_target, get_attn_map=False):
        k = self.k_ctx(x_ctx)
        v = self.v_ctx(x_ctx)
        q = self.q_target(x_target)

        B, T, _ = x_ctx.size()
        k = k.view(B, T, self.n_heads, self.n_embds // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.n_embds // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.dimin // self.n_heads).transpose(1, 2)

        if get_attn_map:
            attn_map = get_attention(query=q, key=k)
            torch.cuda.empty_cache()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0, is_causal=True
        )

        d_target = self.c_proj(attn_output.transpose(1, 2).contiguous().view(B, T, self.dimin))

        if get_attn_map:
            return d_target, attn_map
        else:
            return d_target, None


class OriginalTemporalSAE(torch.nn.Module):
    def __init__(self, dimin=2, width=5, n_heads=8, sae_diff_type='relu', kval_topk=None, tied_weights=True,
        n_attn_layers=1, bottleneck_factor=64):
        super(OriginalTemporalSAE, self).__init__()
        self.sae_type = 'temporal'
        self.width = width
        self.dimin = dimin
        self.eps = 1e-6
        self.lam = 1 / (4 * dimin)
        self.tied_weights = tied_weights

        self.n_attn_layers = n_attn_layers
        self.attn_layers = nn.ModuleList([
            ManualAttention(dimin=width, n_heads=n_heads, bottleneck_factor=bottleneck_factor,
                            bias_k=True, bias_q=True, bias_v=True, bias_o=True)
            for _ in range(n_attn_layers)
        ])

        self.D = nn.Parameter(torch.randn((width, dimin)))
        self.b = nn.Parameter(torch.zeros((1, dimin)))
        if not tied_weights:
            self.E = nn.Parameter(torch.randn((dimin, width)))

        self.sae_diff_type = sae_diff_type
        self.kval_topk = kval_topk if sae_diff_type == 'topk' else None

    def forward(self, x_input, return_graph=False, inf_k=None):
        B, L, _ = x_input.size()
        E = self.D.T if self.tied_weights else self.E

        x_input = x_input - self.b

        attn_graphs = []

        z_pred = torch.zeros((B, L, self.width), device=x_input.device, dtype=x_input.dtype)
        for attn_layer in self.attn_layers:
            z_input = F.relu(torch.matmul(x_input * self.lam, E))
            z_ctx = torch.cat((torch.zeros_like(z_input[:, :1, :]), z_input[:, :-1, :].clone()), dim=1)

            z_pred_, attn_graphs_ = attn_layer(z_ctx, z_input, get_attn_map=return_graph)

            z_pred_ = F.relu(z_pred_)
            Dz_pred_ = torch.matmul(z_pred_, self.D)
            Dz_norm_ = (Dz_pred_.norm(dim=-1, keepdim=True) + self.eps)

            proj_scale = (Dz_pred_ * x_input).sum(dim=-1, keepdim=True) / Dz_norm_.pow(2)

            z_pred = z_pred + (z_pred_ * proj_scale)
            x_input = x_input - proj_scale * Dz_pred_

            if return_graph:
                attn_graphs.append(attn_graphs_)

        if self.sae_diff_type=='relu':
            z_novel = F.relu(torch.matmul(x_input * self.lam, E))
        elif self.sae_diff_type=='topk':
            kval = self.kval_topk if inf_k is None else inf_k
            z_novel = F.relu(torch.matmul(x_input * self.lam, E))
            _, topk_indices = torch.topk(z_novel, kval, dim=-1)
            mask = torch.zeros_like(z_novel)
            mask.scatter_(-1, topk_indices, 1)
            z_novel = z_novel * mask
        elif self.sae_diff_type=='nullify':
            z_novel = torch.zeros_like(z_pred)

        x_recons = torch.matmul(z_novel + z_pred, self.D) + self.b

        with torch.no_grad():
            x_pred_recons = torch.matmul(z_pred, self.D)
            x_novel_recons = torch.matmul(z_novel, self.D)

        results_dict = {
            'novel_codes': z_novel,
            'novel_recons': x_novel_recons,
            'pred_codes': z_pred,
            'pred_recons': x_pred_recons,
            'attn_graphs': torch.stack(attn_graphs, dim=1) if return_graph else None
        }

        return x_recons, results_dict

    @classmethod
    def from_pretrained(cls, folder_path, dtype, device, **kwargs):
        config_path = os.path.join(folder_path, 'conf.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_args = {
            'dimin': config['llm']['dimin'],
            'width': int(config['llm']['dimin'] * config['sae']['exp_factor']),
            'n_heads': config['sae']['n_heads'],
            'sae_diff_type': config['sae']['sae_diff_type'],
            'kval_topk': config['sae']['kval_topk'],
            'tied_weights': config['sae']['tied_weights'],
            'n_attn_layers': config['sae']['n_attn_layers'],
            'bottleneck_factor': config['sae']['bottleneck_factor'],
        }
        model_args.update(kwargs)

        autoencoder = cls(**model_args)

        ckpt_path = os.path.join(folder_path, 'latest_ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        if 'sae' in checkpoint:
            autoencoder.load_state_dict(checkpoint['sae'])
        else:
            autoencoder.load_state_dict(checkpoint)

        autoencoder = autoencoder.to(device=device, dtype=dtype)
        return autoencoder


# ============================================================================
# Load both implementations
# ============================================================================

print("\n" + "=" * 80)
print("LOADING MODELS")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

print(f"Device: {device}")
print(f"Dtype: {dtype}")

# Load original implementation
sae_dir = Path("temporal_saes_weights/temporal")
original_sae = OriginalTemporalSAE.from_pretrained(
    folder_path=sae_dir,
    device=device,
    dtype=dtype
)
original_sae.eval()
print("✓ Loaded original TemporalSAE")

# Load SAELens implementation
saelens_sae = SAE.from_pretrained(
    release="temporal-sae-gemma-2-2b",
    sae_id="blocks.12.hook_resid_post",
    device=device
)
saelens_sae.eval()
print("✓ Loaded SAELens TemporalSAE")

# ============================================================================
# Compare configurations
# ============================================================================

print("\n" + "=" * 80)
print("CONFIGURATION COMPARISON")
print("=" * 80)

configs_match = True
config_checks = [
    ("d_in/dimin", original_sae.dimin, saelens_sae.cfg.d_in),
    ("d_sae/width", original_sae.width, saelens_sae.cfg.d_sae),
    ("n_heads", original_sae.attn_layers[0].n_heads, saelens_sae.cfg.n_heads),
    ("n_attn_layers", original_sae.n_attn_layers, saelens_sae.cfg.n_attn_layers),
    ("sae_diff_type", original_sae.sae_diff_type, saelens_sae.cfg.sae_diff_type),
    ("kval_topk", original_sae.kval_topk, saelens_sae.cfg.kval_topk),
    ("tied_weights", original_sae.tied_weights, saelens_sae.cfg.tied_weights),
]

for name, orig_val, saelens_val in config_checks:
    match = orig_val == saelens_val
    status = "✓" if match else "✗"
    print(f"{status} {name}: original={orig_val}, saelens={saelens_val}")
    if not match:
        configs_match = False

if configs_match:
    print("\n✓ All configurations match!")
else:
    print("\n✗ Configuration mismatch detected!")

# ============================================================================
# Compare weights
# ============================================================================

print("\n" + "=" * 80)
print("WEIGHT COMPARISON")
print("=" * 80)

weights_match = True

# Check D (decoder)
d_diff = (original_sae.D - saelens_sae.D).abs().max().item()
print(f"D (decoder) max diff: {d_diff:.2e}")
if d_diff > 1e-6:
    print("  ✗ Decoder weights differ!")
    weights_match = False
else:
    print("  ✓ Decoder weights match")

# Check b (bias)
b_diff = (original_sae.b - saelens_sae.b).abs().max().item()
print(f"b (bias) max diff: {b_diff:.2e}")
if b_diff > 1e-6:
    print("  ✗ Bias weights differ!")
    weights_match = False
else:
    print("  ✓ Bias weights match")

# Check E (encoder) if not tied
if not original_sae.tied_weights:
    e_diff = (original_sae.E - saelens_sae.E).abs().max().item()
    print(f"E (encoder) max diff: {e_diff:.2e}")
    if e_diff > 1e-6:
        print("  ✗ Encoder weights differ!")
        weights_match = False
    else:
        print("  ✓ Encoder weights match")

# Check attention layers
for i, (orig_attn, saelens_attn) in enumerate(zip(original_sae.attn_layers, saelens_sae.attn_layers)):
    print(f"\nAttention layer {i}:")
    for param_name in ['k_ctx', 'q_target', 'v_ctx', 'c_proj']:
        orig_param = getattr(orig_attn, param_name)
        saelens_param = getattr(saelens_attn, param_name)

        weight_diff = (orig_param.weight - saelens_param.weight).abs().max().item()
        print(f"  {param_name}.weight max diff: {weight_diff:.2e}")
        if weight_diff > 1e-6:
            print(f"    ✗ {param_name} weights differ!")
            weights_match = False

        if orig_param.bias is not None:
            bias_diff = (orig_param.bias - saelens_param.bias).abs().max().item()
            print(f"  {param_name}.bias max diff: {bias_diff:.2e}")
            if bias_diff > 1e-6:
                print(f"    ✗ {param_name} bias differs!")
                weights_match = False

if weights_match:
    print("\n✓ All weights match!")
else:
    print("\n✗ Weight mismatch detected!")

# ============================================================================
# Compare outputs
# ============================================================================

print("\n" + "=" * 80)
print("OUTPUT COMPARISON")
print("=" * 80)

# Create test input
torch.manual_seed(42)
batch_size = 2
seq_len = 32
x = torch.randn(batch_size, seq_len, original_sae.dimin, device=device, dtype=dtype)

print(f"Test input shape: {x.shape}")

# Run original implementation
with torch.no_grad():
    orig_recons, orig_results = original_sae(x, return_graph=False)
    orig_novel_codes = orig_results['novel_codes']
    orig_pred_codes = orig_results['pred_codes']

print("\nOriginal implementation:")
print(f"  Reconstruction shape: {orig_recons.shape}")
print(f"  Novel codes shape: {orig_novel_codes.shape}")
print(f"  Novel codes L0: {(orig_novel_codes != 0).sum(dim=-1).float().mean().item():.1f}")

# Run SAELens implementation
with torch.no_grad():
    # Full forward pass
    saelens_recons = saelens_sae.forward(x)

    # Get novel codes via encode
    saelens_novel_codes = saelens_sae.encode(x)

print("\nSAELens implementation:")
print(f"  Reconstruction shape: {saelens_recons.shape}")
print(f"  Novel codes shape: {saelens_novel_codes.shape}")
print(f"  Novel codes L0: {(saelens_novel_codes != 0).sum(dim=-1).float().mean().item():.1f}")

# Compare outputs
print("\n" + "-" * 80)
print("Numerical comparison:")
print("-" * 80)

recons_diff = (orig_recons - saelens_recons).abs()
print(f"Reconstruction difference:")
print(f"  Max: {recons_diff.max().item():.2e}")
print(f"  Mean: {recons_diff.mean().item():.2e}")
print(f"  Std: {recons_diff.std().item():.2e}")

novel_diff = (orig_novel_codes - saelens_novel_codes).abs()
print(f"\nNovel codes difference:")
print(f"  Max: {novel_diff.max().item():.2e}")
print(f"  Mean: {novel_diff.mean().item():.2e}")
print(f"  Std: {novel_diff.std().item():.2e}")

# Check if outputs match within tolerance
recons_match = recons_diff.max().item() < 1e-5
novel_match = novel_diff.max().item() < 1e-5

print("\n" + "=" * 80)
print("FINAL VERDICT")
print("=" * 80)

all_match = configs_match and weights_match and recons_match and novel_match

if all_match:
    print("✓ SUCCESS: SAELens implementation matches original implementation!")
    print("  - Configurations match")
    print("  - Weights match")
    print("  - Outputs match (within tolerance)")
else:
    print("✗ MISMATCH DETECTED:")
    if not configs_match:
        print("  - Configuration mismatch")
    if not weights_match:
        print("  - Weight mismatch")
    if not recons_match:
        print(f"  - Reconstruction mismatch (max diff: {recons_diff.max().item():.2e})")
    if not novel_match:
        print(f"  - Novel codes mismatch (max diff: {novel_diff.max().item():.2e})")

print("=" * 80)
