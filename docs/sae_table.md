# Pretrained SAEs

This is a list of SAEs importable from the SAELens package. Click on each link for more details.

*This file contains the contents of `sae_lens/pretrained_saes.yaml` in Markdown*

## [gpt2-small-res-jb](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted)

- **Huggingface Repo**: jbloom/GPT2-Small-SAEs-Reformatted
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)
    - [Dashboards](https://www.neuronpedia.org/gpt2sm-res-jb)
    - [Publication](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream)

| hook_name                 |   hook_layer |   d_sae |   context_size | normalize_activations   |
|:--------------------------|-------------:|--------:|---------------:|:------------------------|
| blocks.0.hook_resid_pre   |            0 |   24576 |            128 | none                    |
| blocks.1.hook_resid_pre   |            1 |   24576 |            128 | none                    |
| blocks.2.hook_resid_pre   |            2 |   24576 |            128 | none                    |
| blocks.3.hook_resid_pre   |            3 |   24576 |            128 | none                    |
| blocks.4.hook_resid_pre   |            4 |   24576 |            128 | none                    |
| blocks.5.hook_resid_pre   |            5 |   24576 |            128 | none                    |
| blocks.6.hook_resid_pre   |            6 |   24576 |            128 | none                    |
| blocks.7.hook_resid_pre   |            7 |   24576 |            128 | none                    |
| blocks.8.hook_resid_pre   |            8 |   24576 |            128 | none                    |
| blocks.9.hook_resid_pre   |            9 |   24576 |            128 | none                    |
| blocks.10.hook_resid_pre  |           10 |   24576 |            128 | none                    |
| blocks.11.hook_resid_pre  |           11 |   24576 |            128 | none                    |
| blocks.11.hook_resid_post |           11 |   24576 |            128 | none                    |

## [gpt2-small-hook-z-kk](https://huggingface.co/ckkissane/attn-saes-gpt2-small-all-layers)

- **Huggingface Repo**: ckkissane/attn-saes-gpt2-small-all-layers
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)
    - [Dashboards](https://www.neuronpedia.org/gpt2sm-kk)
    - [Publication](https://www.lesswrong.com/posts/FSTRedtjuHa4Gfdbr/attention-saes-scale-to-gpt-2-small)

| hook_name             |   hook_layer |   d_sae |   context_size | normalize_activations   |
|:----------------------|-------------:|--------:|---------------:|:------------------------|
| blocks.0.attn.hook_z  |            0 |   24576 |            128 | none                    |
| blocks.1.attn.hook_z  |            1 |   24576 |            128 | none                    |
| blocks.2.attn.hook_z  |            2 |   24576 |            128 | none                    |
| blocks.3.attn.hook_z  |            3 |   24576 |            128 | none                    |
| blocks.4.attn.hook_z  |            4 |   24576 |            128 | none                    |
| blocks.5.attn.hook_z  |            5 |   49152 |            128 | none                    |
| blocks.6.attn.hook_z  |            6 |   24576 |            128 | none                    |
| blocks.7.attn.hook_z  |            7 |   49152 |            128 | none                    |
| blocks.8.attn.hook_z  |            8 |   24576 |            128 | none                    |
| blocks.9.attn.hook_z  |            9 |   24576 |            128 | none                    |
| blocks.10.attn.hook_z |           10 |   24576 |            128 | none                    |
| blocks.11.attn.hook_z |           11 |   24576 |            128 | none                    |

## [gpt2-small-mlp-tm](https://huggingface.co/tommmcgrath/gpt2-small-mlp-out-saes)

- **Huggingface Repo**: tommmcgrath/gpt2-small-mlp-out-saes
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)

| hook_name              |   hook_layer |   d_sae |   context_size | normalize_activations    |
|:-----------------------|-------------:|--------:|---------------:|:-------------------------|
| blocks.0.hook_mlp_out  |            0 |   24576 |            512 | expected_average_only_in |
| blocks.1.hook_mlp_out  |            1 |   24576 |            512 | expected_average_only_in |
| blocks.2.hook_mlp_out  |            2 |   24576 |            512 | expected_average_only_in |
| blocks.3.hook_mlp_out  |            3 |   24576 |            512 | expected_average_only_in |
| blocks.4.hook_mlp_out  |            4 |   24576 |            512 | expected_average_only_in |
| blocks.5.hook_mlp_out  |            5 |   24576 |            512 | expected_average_only_in |
| blocks.6.hook_mlp_out  |            6 |   24576 |            512 | expected_average_only_in |
| blocks.7.hook_mlp_out  |            7 |   24576 |            512 | expected_average_only_in |
| blocks.8.hook_mlp_out  |            8 |   24576 |            512 | expected_average_only_in |
| blocks.9.hook_mlp_out  |            9 |   24576 |            512 | expected_average_only_in |
| blocks.10.hook_mlp_out |           10 |   24576 |            512 | expected_average_only_in |
| blocks.11.hook_mlp_out |           11 |   24576 |            512 | expected_average_only_in |

## [gpt2-small-res-jb-feature-splitting](https://huggingface.co/jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8)

- **Huggingface Repo**: jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)
    - [Dashboards](https://www.neuronpedia.org/gpt2sm-rfs-jb)

| hook_name               |   hook_layer |   d_sae |   context_size | normalize_activations   |
|:------------------------|-------------:|--------:|---------------:|:------------------------|
| blocks.8.hook_resid_pre |            8 |     768 |            128 | none                    |
| blocks.8.hook_resid_pre |            8 |    1536 |            128 | none                    |
| blocks.8.hook_resid_pre |            8 |    3072 |            128 | none                    |
| blocks.8.hook_resid_pre |            8 |    6144 |            128 | none                    |
| blocks.8.hook_resid_pre |            8 |   12288 |            128 | none                    |
| blocks.8.hook_resid_pre |            8 |   24576 |            128 | none                    |
| blocks.8.hook_resid_pre |            8 |   49152 |            128 | none                    |
| blocks.8.hook_resid_pre |            8 |   98304 |            128 | none                    |

## [gpt2-small-resid-post-v5-32k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs
- **model**: gpt2-small

| hook_name                 |   hook_layer |   d_sae |   context_size | normalize_activations   |
|:--------------------------|-------------:|--------:|---------------:|:------------------------|
| blocks.0.hook_resid_post  |            0 |   32768 |             64 | layer_norm              |
| blocks.1.hook_resid_post  |            1 |   32768 |             64 | layer_norm              |
| blocks.2.hook_resid_post  |            2 |   32768 |             64 | layer_norm              |
| blocks.3.hook_resid_post  |            3 |   32768 |             64 | layer_norm              |
| blocks.4.hook_resid_post  |            4 |   32768 |             64 | layer_norm              |
| blocks.5.hook_resid_post  |            5 |   32768 |             64 | layer_norm              |
| blocks.6.hook_resid_post  |            6 |   32768 |             64 | layer_norm              |
| blocks.7.hook_resid_post  |            7 |   32768 |             64 | layer_norm              |
| blocks.8.hook_resid_post  |            8 |   32768 |             64 | layer_norm              |
| blocks.9.hook_resid_post  |            9 |   32768 |             64 | layer_norm              |
| blocks.10.hook_resid_post |           10 |   32768 |             64 | layer_norm              |
| blocks.11.hook_resid_post |           11 |   32768 |             64 | layer_norm              |

## [gpt2-small-resid-post-v5-128k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs
- **model**: gpt2-small

| hook_name                 |   hook_layer |   d_sae |   context_size | normalize_activations   |
|:--------------------------|-------------:|--------:|---------------:|:------------------------|
| blocks.0.hook_resid_post  |            0 |  131072 |             64 | layer_norm              |
| blocks.1.hook_resid_post  |            1 |  131072 |             64 | layer_norm              |
| blocks.2.hook_resid_post  |            2 |  131072 |             64 | layer_norm              |
| blocks.3.hook_resid_post  |            3 |  131072 |             64 | layer_norm              |
| blocks.4.hook_resid_post  |            4 |  131072 |             64 | layer_norm              |
| blocks.5.hook_resid_post  |            5 |  131072 |             64 | layer_norm              |
| blocks.6.hook_resid_post  |            6 |  131072 |             64 | layer_norm              |
| blocks.7.hook_resid_post  |            7 |  131072 |             64 | layer_norm              |
| blocks.8.hook_resid_post  |            8 |  131072 |             64 | layer_norm              |
| blocks.9.hook_resid_post  |            9 |  131072 |             64 | layer_norm              |
| blocks.10.hook_resid_post |           10 |  131072 |             64 | layer_norm              |
| blocks.11.hook_resid_post |           11 |  131072 |             64 | layer_norm              |

## [gemma-2b-res-jb](https://huggingface.co/jbloom/Gemma-2b-Residual-Stream-SAEs)

- **Huggingface Repo**: jbloom/Gemma-2b-Residual-Stream-SAEs
- **model**: gemma-2b
- **Additional Links**:
    - [Model](https://huggingface.co/google/gemma-2b)
    - [Dashboards](https://www.neuronpedia.org/gemma2b-res-jb)

| hook_name                 |   hook_layer |   d_sae |   context_size | normalize_activations    |
|:--------------------------|-------------:|--------:|---------------:|:-------------------------|
| blocks.0.hook_resid_post  |            0 |   16384 |           1024 | none                     |
| blocks.6.hook_resid_post  |            6 |   16384 |           1024 | none                     |
| blocks.12.hook_resid_post |           12 |   16384 |           1024 | expected_average_only_in |

## [gemma-2b-it-res-jb](https://huggingface.co/jbloom/Gemma-2b-IT-Residual-Stream-SAEs)

- **Huggingface Repo**: jbloom/Gemma-2b-IT-Residual-Stream-SAEs
- **model**: gemma-2b-it
- **Additional Links**:
    - [Model](https://huggingface.co/google/gemma-2b-it)
    - [Dashboards](https://www.neuronpedia.org/gemma2bit-res-jb)

| hook_name                 |   hook_layer |   d_sae |   context_size | normalize_activations   |
|:--------------------------|-------------:|--------:|---------------:|:------------------------|
| blocks.12.hook_resid_post |           12 |   16384 |           1024 | none                    |

## [mistral-7b-res-wg](https://huggingface.co/JoshEngels/Mistral-7B-Residual-Stream-SAEs)

- **Huggingface Repo**: JoshEngels/Mistral-7B-Residual-Stream-SAEs
- **model**: mistral-7b

| hook_name                |   hook_layer |   d_sae |   context_size | normalize_activations   |
|:-------------------------|-------------:|--------:|---------------:|:------------------------|
| blocks.8.hook_resid_pre  |            8 |   65536 |            256 | none                    |
| blocks.16.hook_resid_pre |           16 |   65536 |            256 | none                    |
| blocks.24.hook_resid_pre |           24 |   65536 |            256 | none                    |

