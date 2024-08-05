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

| hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:--------------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_resid_pre   |            0 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.1.hook_resid_pre   |            1 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.2.hook_resid_pre   |            2 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.3.hook_resid_pre   |            3 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.4.hook_resid_pre   |            4 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.5.hook_resid_pre   |            5 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.6.hook_resid_pre   |            6 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.7.hook_resid_pre   |            7 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre   |            8 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.9.hook_resid_pre   |            9 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.10.hook_resid_pre  |           10 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.11.hook_resid_pre  |           11 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.11.hook_resid_post |           11 |   24576 |            128 | Skylion007/openwebtext | none                    |

## [gpt2-small-hook-z-kk](https://huggingface.co/ckkissane/attn-saes-gpt2-small-all-layers)

- **Huggingface Repo**: ckkissane/attn-saes-gpt2-small-all-layers
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)
    - [Dashboards](https://www.neuronpedia.org/gpt2sm-kk)
    - [Publication](https://www.lesswrong.com/posts/FSTRedtjuHa4Gfdbr/attention-saes-scale-to-gpt-2-small)

| hook_name             |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:----------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.attn.hook_z  |            0 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.1.attn.hook_z  |            1 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.2.attn.hook_z  |            2 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.3.attn.hook_z  |            3 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.4.attn.hook_z  |            4 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.5.attn.hook_z  |            5 |   49152 |            128 | Skylion007/openwebtext | none                    |
| blocks.6.attn.hook_z  |            6 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.7.attn.hook_z  |            7 |   49152 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.attn.hook_z  |            8 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.9.attn.hook_z  |            9 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.10.attn.hook_z |           10 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.11.attn.hook_z |           11 |   24576 |            128 | Skylion007/openwebtext | none                    |

## [gpt2-small-mlp-tm](https://huggingface.co/tommmcgrath/gpt2-small-mlp-out-saes)

- **Huggingface Repo**: tommmcgrath/gpt2-small-mlp-out-saes
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)

| hook_name              |   hook_layer |   d_sae |   context_size | dataset_path                                          | normalize_activations    |
|:-----------------------|-------------:|--------:|---------------:|:------------------------------------------------------|:-------------------------|
| blocks.0.hook_mlp_out  |            0 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.1.hook_mlp_out  |            1 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.2.hook_mlp_out  |            2 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.3.hook_mlp_out  |            3 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.4.hook_mlp_out  |            4 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.5.hook_mlp_out  |            5 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.6.hook_mlp_out  |            6 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.7.hook_mlp_out  |            7 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.8.hook_mlp_out  |            8 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.9.hook_mlp_out  |            9 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.10.hook_mlp_out |           10 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.11.hook_mlp_out |           11 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |

## [gpt2-small-res-jb-feature-splitting](https://huggingface.co/jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8)

- **Huggingface Repo**: jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)
    - [Dashboards](https://www.neuronpedia.org/gpt2sm-rfs-jb)

| hook_name               |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:------------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.8.hook_resid_pre |            8 |     768 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre |            8 |    1536 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre |            8 |    3072 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre |            8 |    6144 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre |            8 |   12288 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre |            8 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre |            8 |   49152 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre |            8 |   98304 |            128 | Skylion007/openwebtext | none                    |

## [gpt2-small-resid-post-v5-32k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs
- **model**: gpt2-small

| hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path                                          | normalize_activations   |
|:--------------------------|-------------:|--------:|---------------:|:------------------------------------------------------|:------------------------|
| blocks.0.hook_resid_post  |            0 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.1.hook_resid_post  |            1 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.2.hook_resid_post  |            2 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.3.hook_resid_post  |            3 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.4.hook_resid_post  |            4 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.5.hook_resid_post  |            5 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.6.hook_resid_post  |            6 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.7.hook_resid_post  |            7 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.8.hook_resid_post  |            8 |   32768 |             64 | Skylion007/openwebtext                                | layer_norm              |
| blocks.9.hook_resid_post  |            9 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.10.hook_resid_post |           10 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.11.hook_resid_post |           11 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |

## [gpt2-small-resid-post-v5-128k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs
- **model**: gpt2-small

| hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path                                          | normalize_activations   |
|:--------------------------|-------------:|--------:|---------------:|:------------------------------------------------------|:------------------------|
| blocks.0.hook_resid_post  |            0 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.1.hook_resid_post  |            1 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.2.hook_resid_post  |            2 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.3.hook_resid_post  |            3 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.4.hook_resid_post  |            4 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.5.hook_resid_post  |            5 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.6.hook_resid_post  |            6 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.7.hook_resid_post  |            7 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.8.hook_resid_post  |            8 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.9.hook_resid_post  |            9 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.10.hook_resid_post |           10 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.11.hook_resid_post |           11 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |

## [gemma-2b-res-jb](https://huggingface.co/jbloom/Gemma-2b-Residual-Stream-SAEs)

- **Huggingface Repo**: jbloom/Gemma-2b-Residual-Stream-SAEs
- **model**: gemma-2b
- **Additional Links**:
    - [Model](https://huggingface.co/google/gemma-2b)
    - [Dashboards](https://www.neuronpedia.org/gemma2b-res-jb)

| hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path                      | normalize_activations    |
|:--------------------------|-------------:|--------:|---------------:|:----------------------------------|:-------------------------|
| blocks.0.hook_resid_post  |            0 |   16384 |           1024 | HuggingFaceFW/fineweb             | none                     |
| blocks.6.hook_resid_post  |            6 |   16384 |           1024 | HuggingFaceFW/fineweb             | none                     |
| blocks.10.hook_resid_post |           10 |   16384 |           1024 | ctigges/openwebtext-gemma-1024-cl | none                     |
| blocks.12.hook_resid_post |           12 |   16384 |           1024 | HuggingFaceFW/fineweb             | expected_average_only_in |
| blocks.17.hook_resid_post |           17 |   16384 |           1024 | ctigges/openwebtext-gemma-1024-cl | none                     |

## [gemma-2b-it-res-jb](https://huggingface.co/jbloom/Gemma-2b-IT-Residual-Stream-SAEs)

- **Huggingface Repo**: jbloom/Gemma-2b-IT-Residual-Stream-SAEs
- **model**: gemma-2b-it
- **Additional Links**:
    - [Model](https://huggingface.co/google/gemma-2b-it)
    - [Dashboards](https://www.neuronpedia.org/gemma2bit-res-jb)

| hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path              | normalize_activations   |
|:--------------------------|-------------:|--------:|---------------:|:--------------------------|:------------------------|
| blocks.12.hook_resid_post |           12 |   16384 |           1024 | chanind/openwebtext-gemma | none                    |

## [mistral-7b-res-wg](https://huggingface.co/JoshEngels/Mistral-7B-Residual-Stream-SAEs)

- **Huggingface Repo**: JoshEngels/Mistral-7B-Residual-Stream-SAEs
- **model**: mistral-7b

| hook_name                |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:-------------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| blocks.8.hook_resid_pre  |            8 |   65536 |            256 | monology/pile-uncopyrighted | none                    |
| blocks.16.hook_resid_pre |           16 |   65536 |            256 | monology/pile-uncopyrighted | none                    |
| blocks.24.hook_resid_pre |           24 |   65536 |            256 | monology/pile-uncopyrighted | none                    |

## [gpt2-small-resid-mid-v5-32k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-resid-mid-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-32k-resid-mid-SAEs
- **model**: gpt2-small

| hook_name                |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:-------------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_resid_mid  |            0 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.1.hook_resid_mid  |            1 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.2.hook_resid_mid  |            2 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.3.hook_resid_mid  |            3 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.4.hook_resid_mid  |            4 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.5.hook_resid_mid  |            5 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.6.hook_resid_mid  |            6 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.7.hook_resid_mid  |            7 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.8.hook_resid_mid  |            8 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.9.hook_resid_mid  |            9 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.10.hook_resid_mid |           10 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.11.hook_resid_mid |           11 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |

## [gpt2-small-resid-mid-v5-128k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-128k-resid-mid-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-128k-resid-mid-SAEs
- **model**: gpt2-small

| hook_name                |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:-------------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_resid_mid  |            0 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.1.hook_resid_mid  |            1 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.2.hook_resid_mid  |            2 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.3.hook_resid_mid  |            3 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.4.hook_resid_mid  |            4 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.5.hook_resid_mid  |            5 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.6.hook_resid_mid  |            6 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.7.hook_resid_mid  |            7 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.8.hook_resid_mid  |            8 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.9.hook_resid_mid  |            9 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.10.hook_resid_mid |           10 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.11.hook_resid_mid |           11 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |

## [gpt2-small-mlp-out-v5-32k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs
- **model**: gpt2-small

| hook_name              |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:-----------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_mlp_out  |            0 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.1.hook_mlp_out  |            1 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.2.hook_mlp_out  |            2 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.3.hook_mlp_out  |            3 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.4.hook_mlp_out  |            4 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.5.hook_mlp_out  |            5 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.6.hook_mlp_out  |            6 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.7.hook_mlp_out  |            7 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.8.hook_mlp_out  |            8 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.9.hook_mlp_out  |            9 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.10.hook_mlp_out |           10 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.11.hook_mlp_out |           11 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |

## [gpt2-small-mlp-out-v5-128k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-128k-mlp-out-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-128k-mlp-out-SAEs
- **model**: gpt2-small

| hook_name              |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:-----------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_mlp_out  |            0 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.1.hook_mlp_out  |            1 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.2.hook_mlp_out  |            2 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.3.hook_mlp_out  |            3 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.4.hook_mlp_out  |            4 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.5.hook_mlp_out  |            5 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.6.hook_mlp_out  |            6 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.7.hook_mlp_out  |            7 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.8.hook_mlp_out  |            8 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.9.hook_mlp_out  |            9 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.10.hook_mlp_out |           10 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.11.hook_mlp_out |           11 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |

## [gpt2-small-attn-out-v5-32k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-attn-out-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-32k-attn-out-SAEs
- **model**: gpt2-small

| hook_name               |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:------------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_attn_out  |            0 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.1.hook_attn_out  |            1 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.2.hook_attn_out  |            2 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.3.hook_attn_out  |            3 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.4.hook_attn_out  |            4 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.5.hook_attn_out  |            5 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.6.hook_attn_out  |            6 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.7.hook_attn_out  |            7 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.8.hook_attn_out  |            8 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.9.hook_attn_out  |            9 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.10.hook_attn_out |           10 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.11.hook_attn_out |           11 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |

## [gpt2-small-attn-out-v5-128k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-128k-attn-out-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-128k-attn-out-SAEs
- **model**: gpt2-small

| hook_name               |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:------------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_attn_out  |            0 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.1.hook_attn_out  |            1 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.2.hook_attn_out  |            2 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.3.hook_attn_out  |            3 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.4.hook_attn_out  |            4 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.5.hook_attn_out  |            5 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.6.hook_attn_out  |            6 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.7.hook_attn_out  |            7 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.8.hook_attn_out  |            8 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.9.hook_attn_out  |            9 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.10.hook_attn_out |           10 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.11.hook_attn_out |           11 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |

## [gemma-scope-2b-pt-res-canonical](https://huggingface.co/google/gemma-scope-2b-pt-res)

- **Huggingface Repo**: google/gemma-scope-2b-pt-res
- **model**: gemma-2-2b
- **Additional Links**:
    - [Model](https://huggingface.co/google/gemma-2-2b)
    - [Dashboards](https://www.neuronpedia.org/gemma-2-2b/gemmascope-res-16k)
    - [Publication](https://huggingface.co/google/gemma-scope)

| hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:--------------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| blocks.0.hook_resid_post  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_resid_post  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_resid_post  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_resid_post  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_resid_post  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_resid_post  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_resid_post  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_resid_post |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_resid_post |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_resid_post |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_resid_post |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_resid_post |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_resid_post |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_resid_post |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_resid_post |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_resid_post  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_resid_post  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_resid_post  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_resid_post  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_resid_post  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_resid_post  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_resid_post  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_resid_post  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_resid_post  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_resid_post |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_resid_post |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_resid_post |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_resid_post |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_resid_post |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_resid_post |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_resid_post |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_resid_post |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-2b-pt-res](https://huggingface.co/google/gemma-scope-2b-pt-res)

- **Huggingface Repo**: google/gemma-scope-2b-pt-res
- **model**: gemma-2-2b

| hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:--------------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| blocks.0.hook_resid_post  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_resid_post  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_resid_post  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_resid_post  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_resid_post  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_resid_post  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_resid_post  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_resid_post  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_resid_post  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_resid_post  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_resid_post  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_resid_post  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_resid_post  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_resid_post  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_resid_post  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_resid_post  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_resid_post  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_resid_post  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_resid_post  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_resid_post  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_resid_post  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_resid_post  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_resid_post  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_resid_post  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_resid_post  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_resid_post  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_resid_post  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_resid_post  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_resid_post  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_resid_post  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_resid_post  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_resid_post  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_resid_post  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_resid_post  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_resid_post  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_resid_post |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_resid_post |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_resid_post |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_resid_post |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_resid_post |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_resid_post |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_resid_post |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_resid_post |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_resid_post |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_resid_post |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_resid_post |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_resid_post |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_resid_post |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_resid_post |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_resid_post |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_resid_post |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_resid_post |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_resid_post |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_resid_post |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_resid_post |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_resid_post |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_resid_post |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_resid_post |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_resid_post |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_resid_post |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_resid_post |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_resid_post |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_resid_post |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_resid_post |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_resid_post |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_resid_post |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_resid_post |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_resid_post |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_resid_post |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_resid_post |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_resid_post |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_resid_post |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_resid_post |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_resid_post |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_resid_post |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_resid_post  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_resid_post  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_resid_post  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_resid_post  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_resid_post  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_resid_post  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_resid_post  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_resid_post  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_resid_post  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_resid_post  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_resid_post  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_resid_post  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_resid_post  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_resid_post  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_resid_post  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_resid_post  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_resid_post  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_resid_post  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_resid_post  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_resid_post  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_resid_post  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_resid_post  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_resid_post  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_resid_post  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_resid_post  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_resid_post  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_resid_post  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_resid_post  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_resid_post  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_resid_post  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_resid_post  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_resid_post  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_resid_post  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_resid_post  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_resid_post  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_resid_post  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_resid_post  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_resid_post  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_resid_post  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_resid_post  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_resid_post  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_resid_post  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_resid_post  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_resid_post  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_resid_post  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_resid_post  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_resid_post |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_resid_post |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_resid_post |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_resid_post |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_resid_post |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_resid_post |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_resid_post |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_resid_post |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_resid_post |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_resid_post |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_resid_post |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_resid_post |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_resid_post |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_resid_post |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_resid_post |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_resid_post |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_resid_post |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_resid_post |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_resid_post |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_resid_post |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_resid_post |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_resid_post |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_resid_post |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_resid_post |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_resid_post |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_resid_post |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_resid_post |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_resid_post |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_resid_post |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_resid_post |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_resid_post |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_resid_post |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_resid_post |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_resid_post |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_resid_post |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_resid_post |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_resid_post |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_resid_post |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_resid_post |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_resid_post |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_resid_post |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_resid_post |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-2b-pt-mlp-canonical](https://huggingface.co/google/gemma-scope-2b-pt-mlp)

- **Huggingface Repo**: google/gemma-scope-2b-pt-mlp
- **model**: gemma-2-2b

| hook_name              |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:-----------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| blocks.0.hook_mlp_out  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_mlp_out  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_mlp_out  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_mlp_out  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_mlp_out  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_mlp_out  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_mlp_out  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_mlp_out |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_mlp_out |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_mlp_out |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_mlp_out |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_mlp_out |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_mlp_out |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_mlp_out |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_mlp_out |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_mlp_out |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_mlp_out |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_mlp_out |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_mlp_out |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_mlp_out  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_mlp_out  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_mlp_out  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_mlp_out  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_mlp_out  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_mlp_out  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_mlp_out  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_mlp_out |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_mlp_out |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_mlp_out |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_mlp_out |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_mlp_out |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_mlp_out |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_mlp_out |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_mlp_out |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_mlp_out |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_mlp_out |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_mlp_out |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_mlp_out |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-2b-pt-mlp](https://huggingface.co/google/gemma-scope-2b-pt-mlp)

- **Huggingface Repo**: google/gemma-scope-2b-pt-mlp
- **model**: gemma-2-2b

| hook_name              |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:-----------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| blocks.0.hook_mlp_out  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_mlp_out  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_mlp_out  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_mlp_out  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_mlp_out  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_mlp_out  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_mlp_out  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_mlp_out  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_mlp_out  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_mlp_out  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_mlp_out  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_mlp_out  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_mlp_out  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_mlp_out  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_mlp_out  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_mlp_out  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_mlp_out  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_mlp_out  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_mlp_out  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_mlp_out  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_mlp_out  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_mlp_out  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_mlp_out  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_mlp_out  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_mlp_out  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_mlp_out  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_mlp_out  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_mlp_out  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_mlp_out  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_mlp_out  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_mlp_out  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_mlp_out  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_mlp_out  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_mlp_out  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_mlp_out  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_mlp_out |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_mlp_out |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_mlp_out |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_mlp_out |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_mlp_out |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_mlp_out |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_mlp_out |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_mlp_out |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_mlp_out |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_mlp_out |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_mlp_out |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_mlp_out |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_mlp_out |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_mlp_out |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_mlp_out |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_mlp_out |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_mlp_out |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_mlp_out |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_mlp_out |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_mlp_out |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_mlp_out |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_mlp_out |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_mlp_out |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_mlp_out |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_mlp_out |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_mlp_out |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_mlp_out |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_mlp_out |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_mlp_out |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_mlp_out |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_mlp_out |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_mlp_out |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_mlp_out |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_mlp_out |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_mlp_out |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_mlp_out |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_mlp_out |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_mlp_out |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_mlp_out |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_mlp_out |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_mlp_out |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_mlp_out |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_mlp_out |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_mlp_out |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_mlp_out |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_mlp_out |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_mlp_out |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_mlp_out |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_mlp_out |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_mlp_out |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_mlp_out |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_mlp_out |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_mlp_out |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_mlp_out |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_mlp_out |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_mlp_out |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_mlp_out |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_mlp_out |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_mlp_out |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_mlp_out |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_mlp_out  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_mlp_out  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_mlp_out  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_mlp_out  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.hook_mlp_out  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_mlp_out  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_mlp_out  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_mlp_out  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_mlp_out  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_mlp_out  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_mlp_out  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_mlp_out  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_mlp_out  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_mlp_out  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_mlp_out  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_mlp_out  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_mlp_out  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_mlp_out  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_mlp_out  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_mlp_out  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_mlp_out  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_mlp_out  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_mlp_out  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_mlp_out  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_mlp_out  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_mlp_out  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_mlp_out  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_mlp_out  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_mlp_out  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_mlp_out  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_mlp_out  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_mlp_out  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_mlp_out  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_mlp_out  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_mlp_out  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_mlp_out |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_mlp_out |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_mlp_out |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_mlp_out |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_mlp_out |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_mlp_out |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_mlp_out |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_mlp_out |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_mlp_out |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_mlp_out |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_mlp_out |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_mlp_out |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_mlp_out |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_mlp_out |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_mlp_out |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_mlp_out |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_mlp_out |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_mlp_out |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_mlp_out |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_mlp_out |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_mlp_out |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_mlp_out |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_mlp_out |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_mlp_out |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_mlp_out |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_mlp_out |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_mlp_out |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_mlp_out |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_mlp_out |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_mlp_out |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_mlp_out |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_mlp_out |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_mlp_out |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_mlp_out |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_mlp_out |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_mlp_out |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_mlp_out |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_mlp_out |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_mlp_out |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_mlp_out |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_mlp_out |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_mlp_out |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_mlp_out |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_mlp_out |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_mlp_out |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_mlp_out |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_mlp_out |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_mlp_out |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_mlp_out |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_mlp_out |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_mlp_out |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_mlp_out |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_mlp_out |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_mlp_out |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_mlp_out |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_mlp_out |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_mlp_out |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_mlp_out |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_mlp_out |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_mlp_out |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-2b-pt-att](https://huggingface.co/google/gemma-scope-2b-pt-att)

- **Huggingface Repo**: google/gemma-scope-2b-pt-att
- **model**: gemma-2-2b

| hook_name             |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:----------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| blocks.0.attn.hook_z  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.attn.hook_z  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.attn.hook_z  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.attn.hook_z  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.attn.hook_z  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.attn.hook_z  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.attn.hook_z  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.attn.hook_z  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.attn.hook_z  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.attn.hook_z  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.attn.hook_z  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.attn.hook_z  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.attn.hook_z  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.attn.hook_z  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.attn.hook_z  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.attn.hook_z  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.attn.hook_z  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.attn.hook_z  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.attn.hook_z  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.attn.hook_z  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.attn.hook_z  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.attn.hook_z  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.attn.hook_z  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.attn.hook_z  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.attn.hook_z  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.attn.hook_z  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.attn.hook_z  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.attn.hook_z  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.attn.hook_z  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.attn.hook_z  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.attn.hook_z  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.attn.hook_z  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.attn.hook_z  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.attn.hook_z  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.attn.hook_z  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.attn.hook_z  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.attn.hook_z  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.attn.hook_z  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.attn.hook_z  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.attn.hook_z  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.attn.hook_z  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.attn.hook_z  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.attn.hook_z  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.attn.hook_z  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.attn.hook_z  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.attn.hook_z  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.attn.hook_z  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.attn.hook_z  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.attn.hook_z  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.attn.hook_z  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.attn.hook_z |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.attn.hook_z |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.attn.hook_z |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.attn.hook_z |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.attn.hook_z |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.attn.hook_z |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.attn.hook_z |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.attn.hook_z |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.attn.hook_z |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.attn.hook_z |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.attn.hook_z |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.attn.hook_z |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.attn.hook_z |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.attn.hook_z |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.attn.hook_z |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.attn.hook_z |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.attn.hook_z |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.attn.hook_z |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.attn.hook_z |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.attn.hook_z |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.attn.hook_z |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.attn.hook_z |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.attn.hook_z |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.attn.hook_z |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.attn.hook_z |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.attn.hook_z |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.attn.hook_z |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.attn.hook_z |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.attn.hook_z |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.attn.hook_z |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.attn.hook_z |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.attn.hook_z |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.attn.hook_z |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.attn.hook_z |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.attn.hook_z |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.attn.hook_z |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.attn.hook_z |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.attn.hook_z |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.attn.hook_z |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.attn.hook_z |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.attn.hook_z |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.attn.hook_z |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.attn.hook_z |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.attn.hook_z |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.attn.hook_z |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.attn.hook_z |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.attn.hook_z |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.attn.hook_z |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.attn.hook_z |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.attn.hook_z |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.attn.hook_z |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.attn.hook_z |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.attn.hook_z |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.attn.hook_z |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.attn.hook_z |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.attn.hook_z |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.attn.hook_z |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.attn.hook_z |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.attn.hook_z |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.attn.hook_z |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.attn.hook_z |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.attn.hook_z |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.attn.hook_z |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.attn.hook_z |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.attn.hook_z |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.attn.hook_z |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.attn.hook_z |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.attn.hook_z |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.attn.hook_z |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.attn.hook_z |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.attn.hook_z |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.attn.hook_z |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.attn.hook_z |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.attn.hook_z |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.attn.hook_z |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.attn.hook_z |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.attn.hook_z |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.attn.hook_z |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.attn.hook_z |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.attn.hook_z |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.attn.hook_z  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.attn.hook_z  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.attn.hook_z  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.attn.hook_z  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.attn.hook_z  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.attn.hook_z  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.attn.hook_z  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.attn.hook_z  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.attn.hook_z  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.attn.hook_z  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.attn.hook_z  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.attn.hook_z  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.attn.hook_z  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.attn.hook_z  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.attn.hook_z  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.attn.hook_z  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.attn.hook_z  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.attn.hook_z  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.attn.hook_z  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.attn.hook_z  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.attn.hook_z  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.attn.hook_z  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.attn.hook_z  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.attn.hook_z  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.attn.hook_z  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.attn.hook_z  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.attn.hook_z  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.attn.hook_z  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.attn.hook_z  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.attn.hook_z  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.attn.hook_z  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.attn.hook_z  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.attn.hook_z  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.attn.hook_z  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.attn.hook_z  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.attn.hook_z  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.attn.hook_z  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.attn.hook_z  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.attn.hook_z  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.attn.hook_z  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.attn.hook_z  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.attn.hook_z  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.attn.hook_z  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.attn.hook_z  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.attn.hook_z  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.attn.hook_z  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.attn.hook_z  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.attn.hook_z  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.attn.hook_z  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.attn.hook_z  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.attn.hook_z |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.attn.hook_z |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.attn.hook_z |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.attn.hook_z |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.attn.hook_z |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.attn.hook_z |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.attn.hook_z |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.attn.hook_z |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.attn.hook_z |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.attn.hook_z |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.attn.hook_z |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.attn.hook_z |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.attn.hook_z |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.attn.hook_z |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.attn.hook_z |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.attn.hook_z |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.attn.hook_z |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.attn.hook_z |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.attn.hook_z |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.attn.hook_z |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.attn.hook_z |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.attn.hook_z |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.attn.hook_z |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.attn.hook_z |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.attn.hook_z |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.attn.hook_z |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.attn.hook_z |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.attn.hook_z |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.attn.hook_z |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.attn.hook_z |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.attn.hook_z |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.attn.hook_z |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.attn.hook_z |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.attn.hook_z |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.attn.hook_z |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.attn.hook_z |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.attn.hook_z |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.attn.hook_z |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.attn.hook_z |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.attn.hook_z |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.attn.hook_z |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.attn.hook_z |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.attn.hook_z |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.attn.hook_z |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.attn.hook_z |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.attn.hook_z |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.attn.hook_z |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.attn.hook_z |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.attn.hook_z |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.attn.hook_z |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.attn.hook_z |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.attn.hook_z |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.attn.hook_z |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.attn.hook_z |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.attn.hook_z |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.attn.hook_z |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.attn.hook_z |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.attn.hook_z |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.attn.hook_z |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.attn.hook_z |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.attn.hook_z |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.attn.hook_z |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.attn.hook_z |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.attn.hook_z |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.attn.hook_z |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.attn.hook_z |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.attn.hook_z |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.attn.hook_z |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.attn.hook_z |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.attn.hook_z |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.attn.hook_z |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.attn.hook_z |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.attn.hook_z |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.attn.hook_z |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.attn.hook_z |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.attn.hook_z |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.attn.hook_z |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.attn.hook_z |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.attn.hook_z |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.attn.hook_z |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-9b-pt-att](https://huggingface.co/google/gemma-scope-9b-pt-att)

- **Huggingface Repo**: google/gemma-scope-9b-pt-att
- **model**: gemma-2-2b

| hook_name             |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:----------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| blocks.0.attn.hook_z  |            0 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.attn.hook_z  |            1 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.attn.hook_z  |            2 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.attn.hook_z  |            3 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.attn.hook_z  |            4 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.attn.hook_z  |            5 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.attn.hook_z  |            6 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.attn.hook_z  |            6 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.attn.hook_z  |            7 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.attn.hook_z  |            8 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.attn.hook_z  |            9 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.attn.hook_z |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.attn.hook_z |           11 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.attn.hook_z |           12 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.attn.hook_z |           13 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.attn.hook_z |           14 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.attn.hook_z |           15 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.attn.hook_z |           16 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.attn.hook_z |           17 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.attn.hook_z |           18 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.attn.hook_z |           19 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.attn.hook_z |           20 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.attn.hook_z |           21 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.attn.hook_z |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.attn.hook_z |           23 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.attn.hook_z |           24 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.attn.hook_z |           25 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.26.attn.hook_z |           26 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.27.attn.hook_z |           27 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.28.attn.hook_z |           28 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.29.attn.hook_z |           29 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.30.attn.hook_z |           30 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.31.attn.hook_z |           31 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.32.attn.hook_z |           32 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.33.attn.hook_z |           33 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.34.attn.hook_z |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.35.attn.hook_z |           35 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.36.attn.hook_z |           36 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.37.attn.hook_z |           37 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.38.attn.hook_z |           38 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.39.attn.hook_z |           39 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.40.attn.hook_z |           40 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.41.attn.hook_z |           41 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.0.attn.hook_z  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.attn.hook_z  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.attn.hook_z  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.attn.hook_z  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.attn.hook_z  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.attn.hook_z  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.attn.hook_z  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.attn.hook_z  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.attn.hook_z  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.attn.hook_z  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.attn.hook_z |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.attn.hook_z |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.attn.hook_z |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.attn.hook_z |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.attn.hook_z |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.attn.hook_z |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.attn.hook_z |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.attn.hook_z |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.attn.hook_z |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.attn.hook_z |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.attn.hook_z |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.attn.hook_z |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.attn.hook_z |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.attn.hook_z |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.attn.hook_z |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.attn.hook_z |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.26.attn.hook_z |           26 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.27.attn.hook_z |           27 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.28.attn.hook_z |           28 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.29.attn.hook_z |           29 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.30.attn.hook_z |           30 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.31.attn.hook_z |           31 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.32.attn.hook_z |           32 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.33.attn.hook_z |           33 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.34.attn.hook_z |           34 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.35.attn.hook_z |           35 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.36.attn.hook_z |           36 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.37.attn.hook_z |           37 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.37.attn.hook_z |           37 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.38.attn.hook_z |           38 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.39.attn.hook_z |           39 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.40.attn.hook_z |           40 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.40.attn.hook_z |           40 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.41.attn.hook_z |           41 |   16384 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-9b-pt-mlp](https://huggingface.co/google/gemma-scope-9b-pt-mlp)

- **Huggingface Repo**: google/gemma-scope-9b-pt-mlp
- **model**: gemma-2-2b

| hook_name              |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:-----------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| blocks.0.hook_mlp_out  |            0 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.1.hook_mlp_out  |            1 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.2.hook_mlp_out  |            2 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.4.hook_mlp_out  |            4 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.5.hook_mlp_out  |            5 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.6.hook_mlp_out  |            6 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.7.hook_mlp_out  |            7 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.8.hook_mlp_out  |            8 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.9.hook_mlp_out  |            9 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.11.hook_mlp_out |           11 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.12.hook_mlp_out |           12 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.13.hook_mlp_out |           13 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.14.hook_mlp_out |           14 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.15.hook_mlp_out |           15 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.16.hook_mlp_out |           16 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.17.hook_mlp_out |           17 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.18.hook_mlp_out |           18 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.19.hook_mlp_out |           19 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.21.hook_mlp_out |           21 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_mlp_out |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.23.hook_mlp_out |           23 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.24.hook_mlp_out |           24 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.25.hook_mlp_out |           25 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.26.hook_mlp_out |           26 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.27.hook_mlp_out |           27 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.28.hook_mlp_out |           28 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.29.hook_mlp_out |           29 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.30.hook_mlp_out |           30 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.31.hook_mlp_out |           31 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.32.hook_mlp_out |           32 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.33.hook_mlp_out |           33 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.34.hook_mlp_out |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.35.hook_mlp_out |           35 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.36.hook_mlp_out |           36 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.37.hook_mlp_out |           37 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.38.hook_mlp_out |           38 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.39.hook_mlp_out |           39 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.40.hook_mlp_out |           40 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.41.hook_mlp_out |           41 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.26.hook_mlp_out |           26 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.26.hook_mlp_out |           26 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.31.hook_mlp_out |           31 |   16384 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-27b-pt-res](https://huggingface.co/google/gemma-scope-27b-pt-res)

- **Huggingface Repo**: google/gemma-scope-27b-pt-res
- **model**: gemma-2-2b

| hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:--------------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| blocks.10.hook_resid_post |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.10.hook_resid_post |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.22.hook_resid_post |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.34.hook_resid_post |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.34.hook_resid_post |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.34.hook_resid_post |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.34.hook_resid_post |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.34.hook_resid_post |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| blocks.34.hook_resid_post |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |

