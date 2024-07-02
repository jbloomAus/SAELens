
# Importable SAEs

This is a list of  SAEs importable from the SAELens package. Click on each link for more details.
- [Importable SAEs](#importable-saes)
  - [gpt2-small-res-jb](#gpt2-small-res-jb)
  - [gpt2-small-hook-z-kk](#gpt2-small-hook-z-kk)
  - [gpt2-small-mlp-tm](#gpt2-small-mlp-tm)
  - [gpt2-small-res-jb-feature-splitting](#gpt2-small-res-jb-feature-splitting)
  - [gemma-2b-res-jb](#gemma-2b-res-jb)
  - [gemma-2b-it-res-jb](#gemma-2b-it-res-jb)
  - [mistral-7b-res-wg](#mistral-7b-res-wg)

*This file contains the contents of `sae_lens/pretrained_saes.yaml` in Markdown*

## gpt2-small-res-jb

- **repo_id**: "jbloom/GPT2-Small-SAEs-Reformatted"
- **model**: "gpt2-small"
- **conversion_func**: null

| ID | Path | Variance Explained | L0 |
|----|------|--------------------|----|
| blocks.0.hook_resid_pre | blocks.0.hook_resid_pre | 0.999 | 10.0 |
| blocks.1.hook_resid_pre | blocks.1.hook_resid_pre | 0.999 | 10.0 |
| blocks.2.hook_resid_pre | blocks.2.hook_resid_pre | 0.999 | 18.0 |
| blocks.3.hook_resid_pre | blocks.3.hook_resid_pre | 0.999 | 23.0 |
| blocks.4.hook_resid_pre | blocks.4.hook_resid_pre | 0.900 | 31.0 |
| blocks.5.hook_resid_pre | blocks.5.hook_resid_pre | 0.900 | 41.0 |
| blocks.6.hook_resid_pre | blocks.6.hook_resid_pre | 0.900 | 51.0 |
| blocks.7.hook_resid_pre | blocks.7.hook_resid_pre | 0.900 | 54.0 |
| blocks.8.hook_resid_pre | blocks.8.hook_resid_pre | 0.900 | 60.0 |
| blocks.9.hook_resid_pre | blocks.9.hook_resid_pre | 0.77 | 70.0 |
| blocks.10.hook_resid_pre | blocks.10.hook_resid_pre | 0.77 | 52.0 |
| blocks.11.hook_resid_pre | blocks.11.hook_resid_pre | 0.77 | 56.0 |
| blocks.11.hook_resid_post | blocks.11.hook_resid_post | 0.77 | 70.0 |

## gpt2-small-hook-z-kk

- **repo_id**: "ckkissane/attn-saes-gpt2-small-all-layers"
- **model**: "gpt2-small"
- **conversion_func**: "connor_rob_hook_z"

| ID | Path | Variance Explained | L0 |
|----|------|--------------------|----|
| blocks.0.hook_z | gpt2-small_L0_Hcat_z_lr1.20e-03_l11.80e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9.pt | 0.13 | 3.0 |
| blocks.1.hook_z | gpt2-small_L1_Hcat_z_lr1.20e-03_l18.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v5.pt | 0.42 | 23.0 |
| blocks.2.hook_z | gpt2-small_L2_Hcat_z_lr1.20e-03_l11.00e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v4.pt | 0.40 | 16.0 |
| blocks.3.hook_z | gpt2-small_L3_Hcat_z_lr1.20e-03_l19.00e-01_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9.pt | 0.43 | 15.0 |
| blocks.4.hook_z | gpt2-small_L4_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v7.pt | 0.27 | 14.0 |
| blocks.5.hook_z | gpt2-small_L5_Hcat_z_lr1.20e-03_l11.00e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9.pt | 0.13 | 17.0 |
| blocks.6.hook_z | gpt2-small_L6_Hcat_z_lr1.20e-03_l11.10e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9.pt | 0.0 | 17.0 |
| blocks.7.hook_z | gpt2-small_L7_Hcat_z_lr1.20e-03_l11.10e+00_ds49152_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9.pt | -7.55 | 19.0 |
| blocks.8.hook_z | gpt2-small_L8_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v6.pt | -0.22 | 23.0 |
| blocks.9.hook_z | gpt2-small_L9_Hcat_z_lr1.20e-03_l11.20e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9.pt | -0.54 | 23.0 |
| blocks.10.hook_z | gpt2-small_L10_Hcat_z_lr1.20e-03_l11.30e+00_ds24576_bs4096_dc1.00e-05_rsanthropic_rie25000_nr4_v9.pt | -0.27 | 14.0 |
| blocks.11.hook_z | gpt2-small_L11_Hcat_z_lr1.20e-03_l13.00e+00_ds24576_bs4096_dc3.16e-06_rsanthropic_rie25000_nr4_v9.pt | -0.7 | 9.4 |

## gpt2-small-mlp-tm

- **repo_id**: "tommmcgrath/gpt2-small-mlp-out-saes"
- **model**: "gpt2-small"
- **conversion_func**: null

| ID | Path | Variance Explained | L0 |
|----|------|--------------------|----|
| blocks.0.hook_mlp_out | sae_group_gpt2_blocks.0.hook_mlp_out_24576:v1 | 0.999 | 15.0 |
| blocks.1.hook_mlp_out | sae_group_gpt2_blocks.1.hook_mlp_out_24576:v0 | -0.20 | 21.0 |
| blocks.2.hook_mlp_out | sae_group_gpt2_blocks.2.hook_mlp_out_24576:v0 | 0.55 | 137.0 |
| blocks.3.hook_mlp_out | sae_group_gpt2_blocks.3.hook_mlp_out_24576:v0 | 0.41 | 54.0 |
| blocks.4.hook_mlp_out | sae_group_gpt2_blocks.4.hook_mlp_out_24576:v0 | 0.44 | 74.0 |
| blocks.5.hook_mlp_out | sae_group_gpt2_blocks.5.hook_mlp_out_24576:v0 | 0.52 | 76.0 |
| blocks.6.hook_mlp_out | sae_group_gpt2_blocks.6.hook_mlp_out_24576:v0 | 0.508 | 40.0 |
| blocks.7.hook_mlp_out | sae_group_gpt2_blocks.7.hook_mlp_out_24576:v0 | 0.53 | 52.0 |
| blocks.8.hook_mlp_out | sae_group_gpt2_blocks.8.hook_mlp_out_24576:v1 | 0.46 | 36.0 |
| blocks.9.hook_mlp_out | sae_group_gpt2_blocks.9.hook_mlp_out_24576:v0 | 0.37 | 49.0 |
| blocks.10.hook_mlp_out | sae_group_gpt2_blocks.10.hook_mlp_out_24576:v0 | -0.44 | 167.0 |
| blocks.11.hook_mlp_out | sae_group_gpt2_blocks.11.hook_mlp_out_24576:v2 | 0.05 | 170.0 |

## gpt2-small-res-jb-feature-splitting

- **repo_id**: "jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8"
- **model**: "gpt2-small"
- **conversion_func**: null

| ID | Path | Variance Explained | L0 |
|----|------|--------------------|----|
| blocks.8.hook_resid_pre_768 | blocks.8.hook_resid_pre_768 | 0.61 | 36.0 |
| blocks.8.hook_resid_pre_1536 | blocks.8.hook_resid_pre_1536 | 0.67 | 39.0 |
| blocks.8.hook_resid_pre_3072 | blocks.8.hook_resid_pre_3072 | 0.72 | 41.0 |
| blocks.8.hook_resid_pre_6144 | blocks.8.hook_resid_pre_6144 | 0.76 | 43.0 |
| blocks.8.hook_resid_pre_12288 | blocks.8.hook_resid_pre_12288 | 0.77 | 43.0 |
| blocks.8.hook_resid_pre_24576 | blocks.8.hook_resid_pre_24576 | 0.79 | 40.0 |
| blocks.8.hook_resid_pre_49152 | blocks.8.hook_resid_pre_49152 | 0.81 | 40.0 |
| blocks.8.hook_resid_pre_98304 | blocks.8.hook_resid_pre_98304 | 0.82 | 43.0 |

## gemma-2b-res-jb

- **repo_id**: "jbloom/Gemma-2b-Residual-Stream-SAEs"
- **model**: "gemma-2b"
- **conversion_func**: null

| ID | Path | Variance Explained | L0 |
|----|------|--------------------|----|
| blocks.0.hook_resid_post | gemma_2b_blocks.0.hook_resid_post_16384_anthropic | 0.999 | 47.0 |
| blocks.6.hook_resid_post | gemma_2b_blocks.6.hook_resid_post_16384_anthropic_fast_lr | 0.71 | 56.0 |
| blocks.12.hook_resid_post | gemma_2b_blocks.12.hook_resid_post_16384 | -3.6 | 62.0 |

## gemma-2b-it-res-jb

- **repo_id**: "jbloom/Gemma-2b-IT-Residual-Stream-SAEs"
- **model**: "gemma-2b-it"
- **conversion_func**: null

| ID | Path | Variance Explained | L0 |
|----|------|--------------------|----|
| blocks.12.hook_resid_post | gemma_2b_it_blocks.12.hook_resid_post_16384 | 0.57 | 61.0 |

## mistral-7b-res-wg

- **repo_id**: "JoshEngels/Mistral-7B-Residual-Stream-SAEs"
- **model**: "mistral-7b"
- **conversion_func**: "mistral_7b_josh_engels_loader"

| ID | Path | Variance Explained | L0 |
|----|------|--------------------|----|
| blocks.8.hook_resid_pre | mistral_7b_layer_8 | 0.74 | 82 |
| blocks.16.hook_resid_pre | mistral_7b_layer_16 | 0.85 | 74 |
| blocks.24.hook_resid_pre | mistral_7b_layer_24 | 0.72 | 75 |