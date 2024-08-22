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

| id                        | architecture   | neuronpedia          | hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:--------------------------|:---------------|:---------------------|:--------------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_resid_pre   | standard       | gpt2-small/0-res-jb  | blocks.0.hook_resid_pre   |            0 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.1.hook_resid_pre   | standard       | gpt2-small/1-res-jb  | blocks.1.hook_resid_pre   |            1 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.2.hook_resid_pre   | standard       | gpt2-small/2-res-jb  | blocks.2.hook_resid_pre   |            2 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.3.hook_resid_pre   | standard       | gpt2-small/3-res-jb  | blocks.3.hook_resid_pre   |            3 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.4.hook_resid_pre   | standard       | gpt2-small/4-res-jb  | blocks.4.hook_resid_pre   |            4 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.5.hook_resid_pre   | standard       | gpt2-small/0-res-jb  | blocks.5.hook_resid_pre   |            5 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.6.hook_resid_pre   | standard       | gpt2-small/6-res-jb  | blocks.6.hook_resid_pre   |            6 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.7.hook_resid_pre   | standard       | gpt2-small/7-res-jb  | blocks.7.hook_resid_pre   |            7 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre   | standard       | gpt2-small/8-res-jb  | blocks.8.hook_resid_pre   |            8 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.9.hook_resid_pre   | standard       | gpt2-small/9-res-jb  | blocks.9.hook_resid_pre   |            9 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.10.hook_resid_pre  | standard       | gpt2-small/10-res-jb | blocks.10.hook_resid_pre  |           10 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.11.hook_resid_pre  | standard       | gpt2-small/11-res-jb | blocks.11.hook_resid_pre  |           11 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.11.hook_resid_post | standard       | gpt2-small/11-res-jb | blocks.11.hook_resid_post |           11 |   24576 |            128 | Skylion007/openwebtext | none                    |

## [gpt2-small-hook-z-kk](https://huggingface.co/ckkissane/attn-saes-gpt2-small-all-layers)

- **Huggingface Repo**: ckkissane/attn-saes-gpt2-small-all-layers
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)
    - [Dashboards](https://www.neuronpedia.org/gpt2sm-kk)
    - [Publication](https://www.lesswrong.com/posts/FSTRedtjuHa4Gfdbr/attention-saes-scale-to-gpt-2-small)

| id               | architecture   | neuronpedia          | hook_name             |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:-----------------|:---------------|:---------------------|:----------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_z  | standard       | gpt2-small/0-att-kk  | blocks.0.attn.hook_z  |            0 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.1.hook_z  | standard       | gpt2-small/1-att-kk  | blocks.1.attn.hook_z  |            1 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.2.hook_z  | standard       | gpt2-small/2-att-kk  | blocks.2.attn.hook_z  |            2 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.3.hook_z  | standard       | gpt2-small/3-att-kk  | blocks.3.attn.hook_z  |            3 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.4.hook_z  | standard       | gpt2-small/4-att-kk  | blocks.4.attn.hook_z  |            4 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.5.hook_z  | standard       | gpt2-small/5-att-kk  | blocks.5.attn.hook_z  |            5 |   49152 |            128 | Skylion007/openwebtext | none                    |
| blocks.6.hook_z  | standard       | gpt2-small/6-att-kk  | blocks.6.attn.hook_z  |            6 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.7.hook_z  | standard       | gpt2-small/7-att-kk  | blocks.7.attn.hook_z  |            7 |   49152 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_z  | standard       | gpt2-small/8-att-kk  | blocks.8.attn.hook_z  |            8 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.9.hook_z  | standard       | gpt2-small/9-att-kk  | blocks.9.attn.hook_z  |            9 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.10.hook_z | standard       | gpt2-small/10-att-kk | blocks.10.attn.hook_z |           10 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.11.hook_z | standard       | gpt2-small/11-att-kk | blocks.11.attn.hook_z |           11 |   24576 |            128 | Skylion007/openwebtext | none                    |

## [gpt2-small-mlp-tm](https://huggingface.co/tommmcgrath/gpt2-small-mlp-out-saes)

- **Huggingface Repo**: tommmcgrath/gpt2-small-mlp-out-saes
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)

| id                     | architecture   | neuronpedia   | hook_name              |   hook_layer |   d_sae |   context_size | dataset_path                                          | normalize_activations    |
|:-----------------------|:---------------|:--------------|:-----------------------|-------------:|--------:|---------------:|:------------------------------------------------------|:-------------------------|
| blocks.0.hook_mlp_out  | standard       |               | blocks.0.hook_mlp_out  |            0 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.1.hook_mlp_out  | standard       |               | blocks.1.hook_mlp_out  |            1 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.2.hook_mlp_out  | standard       |               | blocks.2.hook_mlp_out  |            2 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.3.hook_mlp_out  | standard       |               | blocks.3.hook_mlp_out  |            3 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.4.hook_mlp_out  | standard       |               | blocks.4.hook_mlp_out  |            4 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.5.hook_mlp_out  | standard       |               | blocks.5.hook_mlp_out  |            5 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.6.hook_mlp_out  | standard       |               | blocks.6.hook_mlp_out  |            6 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.7.hook_mlp_out  | standard       |               | blocks.7.hook_mlp_out  |            7 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.8.hook_mlp_out  | standard       |               | blocks.8.hook_mlp_out  |            8 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.9.hook_mlp_out  | standard       |               | blocks.9.hook_mlp_out  |            9 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.10.hook_mlp_out | standard       |               | blocks.10.hook_mlp_out |           10 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |
| blocks.11.hook_mlp_out | standard       |               | blocks.11.hook_mlp_out |           11 |   24576 |            512 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | expected_average_only_in |

## [gpt2-small-res-jb-feature-splitting](https://huggingface.co/jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8)

- **Huggingface Repo**: jbloom/GPT2-Small-Feature-Splitting-Experiment-Layer-8
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)
    - [Dashboards](https://www.neuronpedia.org/gpt2sm-rfs-jb)

| id                            | architecture   | neuronpedia                 | hook_name               |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:------------------------------|:---------------|:----------------------------|:------------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.8.hook_resid_pre_768   | standard       | gpt2-small/8-res_fs768-jb   | blocks.8.hook_resid_pre |            8 |     768 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre_1536  | standard       | gpt2-small/8-res_fs1536-jb  | blocks.8.hook_resid_pre |            8 |    1536 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre_3072  | standard       | gpt2-small/8-res_fs3072-jb  | blocks.8.hook_resid_pre |            8 |    3072 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre_6144  | standard       | gpt2-small/8-res_fs6144-jb  | blocks.8.hook_resid_pre |            8 |    6144 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre_12288 | standard       | gpt2-small/8-res_fs12288-jb | blocks.8.hook_resid_pre |            8 |   12288 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre_24576 | standard       | gpt2-small/8-res_fs24576-jb | blocks.8.hook_resid_pre |            8 |   24576 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre_49152 | standard       | gpt2-small/8-res_fs49152-jb | blocks.8.hook_resid_pre |            8 |   49152 |            128 | Skylion007/openwebtext | none                    |
| blocks.8.hook_resid_pre_98304 | standard       | gpt2-small/8-res_fs98304-jb | blocks.8.hook_resid_pre |            8 |   98304 |            128 | Skylion007/openwebtext | none                    |

## [gpt2-small-resid-post-v5-32k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs
- **model**: gpt2-small

| id                        | architecture   | neuronpedia                    | hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path                                          | normalize_activations   |
|:--------------------------|:---------------|:-------------------------------|:--------------------------|-------------:|--------:|---------------:|:------------------------------------------------------|:------------------------|
| blocks.0.hook_resid_post  | standard       | gpt2-small/0-res_post_32k-oai  | blocks.0.hook_resid_post  |            0 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.1.hook_resid_post  | standard       | gpt2-small/1-res_post_32k-oai  | blocks.1.hook_resid_post  |            1 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.2.hook_resid_post  | standard       | gpt2-small/2-res_post_32k-oai  | blocks.2.hook_resid_post  |            2 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.3.hook_resid_post  | standard       | gpt2-small/3-res_post_32k-oai  | blocks.3.hook_resid_post  |            3 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.4.hook_resid_post  | standard       | gpt2-small/4-res_post_32k-oai  | blocks.4.hook_resid_post  |            4 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.5.hook_resid_post  | standard       | gpt2-small/5-res_post_32k-oai  | blocks.5.hook_resid_post  |            5 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.6.hook_resid_post  | standard       | gpt2-small/6-res_post_32k-oai  | blocks.6.hook_resid_post  |            6 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.7.hook_resid_post  | standard       | gpt2-small/7-res_post_32k-oai  | blocks.7.hook_resid_post  |            7 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.8.hook_resid_post  | standard       | gpt2-small/8-res_post_32k-oai  | blocks.8.hook_resid_post  |            8 |   32768 |             64 | Skylion007/openwebtext                                | layer_norm              |
| blocks.9.hook_resid_post  | standard       | gpt2-small/9-res_post_32k-oai  | blocks.9.hook_resid_post  |            9 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.10.hook_resid_post | standard       | gpt2-small/10-res_post_32k-oai | blocks.10.hook_resid_post |           10 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.11.hook_resid_post | standard       | gpt2-small/11-res_post_32k-oai | blocks.11.hook_resid_post |           11 |   32768 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |

## [gpt2-small-resid-post-v5-128k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs
- **model**: gpt2-small

| id                        | architecture   | neuronpedia                     | hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path                                          | normalize_activations   |
|:--------------------------|:---------------|:--------------------------------|:--------------------------|-------------:|--------:|---------------:|:------------------------------------------------------|:------------------------|
| blocks.0.hook_resid_post  | standard       | gpt2-small/0-res_post_128k-oai  | blocks.0.hook_resid_post  |            0 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.1.hook_resid_post  | standard       | gpt2-small/1-res_post_128k-oai  | blocks.1.hook_resid_post  |            1 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.2.hook_resid_post  | standard       | gpt2-small/2-res_post_128k-oai  | blocks.2.hook_resid_post  |            2 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.3.hook_resid_post  | standard       | gpt2-small/3-res_post_128k-oai  | blocks.3.hook_resid_post  |            3 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.4.hook_resid_post  | standard       | gpt2-small/4-res_post_128k-oai  | blocks.4.hook_resid_post  |            4 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.5.hook_resid_post  | standard       | gpt2-small/5-res_post_128k-oai  | blocks.5.hook_resid_post  |            5 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.6.hook_resid_post  | standard       | gpt2-small/6-res_post_128k-oai  | blocks.6.hook_resid_post  |            6 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.7.hook_resid_post  | standard       | gpt2-small/7-res_post_128k-oai  | blocks.7.hook_resid_post  |            7 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.8.hook_resid_post  | standard       | gpt2-small/8-res_post_128k-oai  | blocks.8.hook_resid_post  |            8 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.9.hook_resid_post  | standard       | gpt2-small/9-res_post_128k-oai  | blocks.9.hook_resid_post  |            9 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.10.hook_resid_post | standard       | gpt2-small/10-res_post_128k-oai | blocks.10.hook_resid_post |           10 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |
| blocks.11.hook_resid_post | standard       | gpt2-small/11-res_post_128k-oai | blocks.11.hook_resid_post |           11 |  131072 |             64 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | layer_norm              |

## [gemma-2b-res-jb](https://huggingface.co/jbloom/Gemma-2b-Residual-Stream-SAEs)

- **Huggingface Repo**: jbloom/Gemma-2b-Residual-Stream-SAEs
- **model**: gemma-2b
- **Additional Links**:
    - [Model](https://huggingface.co/google/gemma-2b)
    - [Dashboards](https://www.neuronpedia.org/gemma2b-res-jb)

| id                        | architecture   | neuronpedia        | hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path                      | normalize_activations    |
|:--------------------------|:---------------|:-------------------|:--------------------------|-------------:|--------:|---------------:|:----------------------------------|:-------------------------|
| blocks.0.hook_resid_post  | standard       | gemma-2b/0-res-jb  | blocks.0.hook_resid_post  |            0 |   16384 |           1024 | HuggingFaceFW/fineweb             | none                     |
| blocks.6.hook_resid_post  | standard       | gemma-2b/6-res-jb  | blocks.6.hook_resid_post  |            6 |   16384 |           1024 | HuggingFaceFW/fineweb             | none                     |
| blocks.10.hook_resid_post | standard       | gemma-2b/10-res-jb | blocks.10.hook_resid_post |           10 |   16384 |           1024 | ctigges/openwebtext-gemma-1024-cl | none                     |
| blocks.12.hook_resid_post | standard       | gemma-2b/12-res-jb | blocks.12.hook_resid_post |           12 |   16384 |           1024 | HuggingFaceFW/fineweb             | expected_average_only_in |
| blocks.17.hook_resid_post | standard       |                    | blocks.17.hook_resid_post |           17 |   16384 |           1024 | ctigges/openwebtext-gemma-1024-cl | none                     |

## [gemma-2b-it-res-jb](https://huggingface.co/jbloom/Gemma-2b-IT-Residual-Stream-SAEs)

- **Huggingface Repo**: jbloom/Gemma-2b-IT-Residual-Stream-SAEs
- **model**: gemma-2b-it
- **Additional Links**:
    - [Model](https://huggingface.co/google/gemma-2b-it)
    - [Dashboards](https://www.neuronpedia.org/gemma2bit-res-jb)

| id                        | architecture   | neuronpedia           | hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path              | normalize_activations   |
|:--------------------------|:---------------|:----------------------|:--------------------------|-------------:|--------:|---------------:|:--------------------------|:------------------------|
| blocks.12.hook_resid_post | standard       | gemma-2b-it/12-res-jb | blocks.12.hook_resid_post |           12 |   16384 |           1024 | chanind/openwebtext-gemma | none                    |

## [mistral-7b-res-wg](https://huggingface.co/JoshEngels/Mistral-7B-Residual-Stream-SAEs)

- **Huggingface Repo**: JoshEngels/Mistral-7B-Residual-Stream-SAEs
- **model**: mistral-7b

| id                       | architecture   | neuronpedia   | hook_name                |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:-------------------------|:---------------|:--------------|:-------------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| blocks.8.hook_resid_pre  | standard       |               | blocks.8.hook_resid_pre  |            8 |   65536 |            256 | monology/pile-uncopyrighted | none                    |
| blocks.16.hook_resid_pre | standard       |               | blocks.16.hook_resid_pre |           16 |   65536 |            256 | monology/pile-uncopyrighted | none                    |
| blocks.24.hook_resid_pre | standard       |               | blocks.24.hook_resid_pre |           24 |   65536 |            256 | monology/pile-uncopyrighted | none                    |

## [gpt2-small-resid-mid-v5-32k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-resid-mid-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-32k-resid-mid-SAEs
- **model**: gpt2-small

| id                       | architecture   | neuronpedia   | hook_name                |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:-------------------------|:---------------|:--------------|:-------------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_resid_mid  | standard       |               | blocks.0.hook_resid_mid  |            0 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.1.hook_resid_mid  | standard       |               | blocks.1.hook_resid_mid  |            1 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.2.hook_resid_mid  | standard       |               | blocks.2.hook_resid_mid  |            2 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.3.hook_resid_mid  | standard       |               | blocks.3.hook_resid_mid  |            3 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.4.hook_resid_mid  | standard       |               | blocks.4.hook_resid_mid  |            4 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.5.hook_resid_mid  | standard       |               | blocks.5.hook_resid_mid  |            5 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.6.hook_resid_mid  | standard       |               | blocks.6.hook_resid_mid  |            6 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.7.hook_resid_mid  | standard       |               | blocks.7.hook_resid_mid  |            7 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.8.hook_resid_mid  | standard       |               | blocks.8.hook_resid_mid  |            8 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.9.hook_resid_mid  | standard       |               | blocks.9.hook_resid_mid  |            9 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.10.hook_resid_mid | standard       |               | blocks.10.hook_resid_mid |           10 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.11.hook_resid_mid | standard       |               | blocks.11.hook_resid_mid |           11 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |

## [gpt2-small-resid-mid-v5-128k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-128k-resid-mid-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-128k-resid-mid-SAEs
- **model**: gpt2-small

| id                       | architecture   | neuronpedia   | hook_name                |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:-------------------------|:---------------|:--------------|:-------------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_resid_mid  | standard       |               | blocks.0.hook_resid_mid  |            0 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.1.hook_resid_mid  | standard       |               | blocks.1.hook_resid_mid  |            1 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.2.hook_resid_mid  | standard       |               | blocks.2.hook_resid_mid  |            2 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.3.hook_resid_mid  | standard       |               | blocks.3.hook_resid_mid  |            3 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.4.hook_resid_mid  | standard       |               | blocks.4.hook_resid_mid  |            4 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.5.hook_resid_mid  | standard       |               | blocks.5.hook_resid_mid  |            5 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.6.hook_resid_mid  | standard       |               | blocks.6.hook_resid_mid  |            6 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.7.hook_resid_mid  | standard       |               | blocks.7.hook_resid_mid  |            7 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.8.hook_resid_mid  | standard       |               | blocks.8.hook_resid_mid  |            8 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.9.hook_resid_mid  | standard       |               | blocks.9.hook_resid_mid  |            9 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.10.hook_resid_mid | standard       |               | blocks.10.hook_resid_mid |           10 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.11.hook_resid_mid | standard       |               | blocks.11.hook_resid_mid |           11 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |

## [gpt2-small-mlp-out-v5-32k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs
- **model**: gpt2-small

| id                     | architecture   | neuronpedia                   | hook_name              |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:-----------------------|:---------------|:------------------------------|:-----------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_mlp_out  | standard       | gpt2-small/0-res_mlp_32k-oai  | blocks.0.hook_mlp_out  |            0 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.1.hook_mlp_out  | standard       | gpt2-small/1-res_mlp_32k-oai  | blocks.1.hook_mlp_out  |            1 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.2.hook_mlp_out  | standard       | gpt2-small/2-res_mlp_32k-oai  | blocks.2.hook_mlp_out  |            2 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.3.hook_mlp_out  | standard       | gpt2-small/3-res_mlp_32k-oai  | blocks.3.hook_mlp_out  |            3 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.4.hook_mlp_out  | standard       | gpt2-small/4-res_mlp_32k-oai  | blocks.4.hook_mlp_out  |            4 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.5.hook_mlp_out  | standard       | gpt2-small/5-res_mlp_32k-oai  | blocks.5.hook_mlp_out  |            5 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.6.hook_mlp_out  | standard       | gpt2-small/6-res_mlp_32k-oai  | blocks.6.hook_mlp_out  |            6 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.7.hook_mlp_out  | standard       | gpt2-small/7-res_mlp_32k-oai  | blocks.7.hook_mlp_out  |            7 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.8.hook_mlp_out  | standard       | gpt2-small/8-res_mlp_32k-oai  | blocks.8.hook_mlp_out  |            8 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.9.hook_mlp_out  | standard       | gpt2-small/9-res_mlp_32k-oai  | blocks.9.hook_mlp_out  |            9 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.10.hook_mlp_out | standard       | gpt2-small/10-res_mlp_32k-oai | blocks.10.hook_mlp_out |           10 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.11.hook_mlp_out | standard       | gpt2-small/11-res_mlp_32k-oai | blocks.11.hook_mlp_out |           11 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |

## [gpt2-small-mlp-out-v5-128k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-128k-mlp-out-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-128k-mlp-out-SAEs
- **model**: gpt2-small

| id                     | architecture   | neuronpedia                    | hook_name              |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:-----------------------|:---------------|:-------------------------------|:-----------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_mlp_out  | standard       | gpt2-small/0-res_mlp_128k-oai  | blocks.0.hook_mlp_out  |            0 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.1.hook_mlp_out  | standard       | gpt2-small/1-res_mlp_128k-oai  | blocks.1.hook_mlp_out  |            1 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.2.hook_mlp_out  | standard       | gpt2-small/2-res_mlp_128k-oai  | blocks.2.hook_mlp_out  |            2 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.3.hook_mlp_out  | standard       | gpt2-small/3-res_mlp_128k-oai  | blocks.3.hook_mlp_out  |            3 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.4.hook_mlp_out  | standard       | gpt2-small/4-res_mlp_128k-oai  | blocks.4.hook_mlp_out  |            4 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.5.hook_mlp_out  | standard       | gpt2-small/5-res_mlp_128k-oai  | blocks.5.hook_mlp_out  |            5 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.6.hook_mlp_out  | standard       | gpt2-small/6-res_mlp_128k-oai  | blocks.6.hook_mlp_out  |            6 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.7.hook_mlp_out  | standard       | gpt2-small/7-res_mlp_128k-oai  | blocks.7.hook_mlp_out  |            7 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.8.hook_mlp_out  | standard       | gpt2-small/8-res_mlp_128k-oai  | blocks.8.hook_mlp_out  |            8 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.9.hook_mlp_out  | standard       | gpt2-small/9-res_mlp_128k-oai  | blocks.9.hook_mlp_out  |            9 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.10.hook_mlp_out | standard       | gpt2-small/10-res_mlp_128k-oai | blocks.10.hook_mlp_out |           10 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.11.hook_mlp_out | standard       | gpt2-small/11-res_mlp_128k-oai | blocks.11.hook_mlp_out |           11 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |

## [gpt2-small-attn-out-v5-32k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-32k-attn-out-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-32k-attn-out-SAEs
- **model**: gpt2-small

| id                      | architecture   | neuronpedia                   | hook_name               |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:------------------------|:---------------|:------------------------------|:------------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_attn_out  | standard       | gpt2-small/0-res_att_32k-oai  | blocks.0.hook_attn_out  |            0 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.1.hook_attn_out  | standard       | gpt2-small/1-res_att_32k-oai  | blocks.1.hook_attn_out  |            1 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.2.hook_attn_out  | standard       | gpt2-small/2-res_att_32k-oai  | blocks.2.hook_attn_out  |            2 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.3.hook_attn_out  | standard       | gpt2-small/3-res_att_32k-oai  | blocks.3.hook_attn_out  |            3 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.4.hook_attn_out  | standard       | gpt2-small/4-res_att_32k-oai  | blocks.4.hook_attn_out  |            4 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.5.hook_attn_out  | standard       | gpt2-small/5-res_att_32k-oai  | blocks.5.hook_attn_out  |            5 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.6.hook_attn_out  | standard       | gpt2-small/6-res_att_32k-oai  | blocks.6.hook_attn_out  |            6 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.7.hook_attn_out  | standard       | gpt2-small/7-res_att_32k-oai  | blocks.7.hook_attn_out  |            7 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.8.hook_attn_out  | standard       | gpt2-small/8-res_att_32k-oai  | blocks.8.hook_attn_out  |            8 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.9.hook_attn_out  | standard       | gpt2-small/9-res_att_32k-oai  | blocks.9.hook_attn_out  |            9 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.10.hook_attn_out | standard       | gpt2-small/10-res_att_32k-oai | blocks.10.hook_attn_out |           10 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.11.hook_attn_out | standard       | gpt2-small/11-res_att_32k-oai | blocks.11.hook_attn_out |           11 |   32768 |             64 | Skylion007/openwebtext | layer_norm              |

## [gpt2-small-attn-out-v5-128k](https://huggingface.co/jbloom/GPT2-Small-OAI-v5-128k-attn-out-SAEs)

- **Huggingface Repo**: jbloom/GPT2-Small-OAI-v5-128k-attn-out-SAEs
- **model**: gpt2-small

| id                      | architecture   | neuronpedia                    | hook_name               |   hook_layer |   d_sae |   context_size | dataset_path           | normalize_activations   |
|:------------------------|:---------------|:-------------------------------|:------------------------|-------------:|--------:|---------------:|:-----------------------|:------------------------|
| blocks.0.hook_attn_out  | standard       | gpt2-small/0-res_att_128k-oai  | blocks.0.hook_attn_out  |            0 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.1.hook_attn_out  | standard       | gpt2-small/1-res_att_128k-oai  | blocks.1.hook_attn_out  |            1 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.2.hook_attn_out  | standard       | gpt2-small/2-res_att_128k-oai  | blocks.2.hook_attn_out  |            2 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.3.hook_attn_out  | standard       | gpt2-small/3-res_att_128k-oai  | blocks.3.hook_attn_out  |            3 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.4.hook_attn_out  | standard       | gpt2-small/4-res_att_128k-oai  | blocks.4.hook_attn_out  |            4 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.5.hook_attn_out  | standard       | gpt2-small/5-res_att_128k-oai  | blocks.5.hook_attn_out  |            5 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.6.hook_attn_out  | standard       | gpt2-small/6-res_att_128k-oai  | blocks.6.hook_attn_out  |            6 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.7.hook_attn_out  | standard       | gpt2-small/7-res_att_128k-oai  | blocks.7.hook_attn_out  |            7 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.8.hook_attn_out  | standard       | gpt2-small/8-res_att_128k-oai  | blocks.8.hook_attn_out  |            8 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.9.hook_attn_out  | standard       | gpt2-small/9-res_att_128k-oai  | blocks.9.hook_attn_out  |            9 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.10.hook_attn_out | standard       | gpt2-small/10-res_att_128k-oai | blocks.10.hook_attn_out |           10 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |
| blocks.11.hook_attn_out | standard       | gpt2-small/11-res_att_128k-oai | blocks.11.hook_attn_out |           11 |  131072 |             64 | Skylion007/openwebtext | layer_norm              |

## [gemma-scope-2b-pt-res-canonical](https://huggingface.co/google/gemma-scope-2b-pt-res)

- **Huggingface Repo**: google/gemma-scope-2b-pt-res
- **model**: gemma-2-2b
- **Additional Links**:
    - [Model](https://huggingface.co/google/gemma-2-2b)
    - [Dashboards](https://www.neuronpedia.org/gemma-2-2b/gemmascope-res-16k)
    - [Publication](https://huggingface.co/google/gemma-scope)

| id                            | architecture   | neuronpedia   | hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:------------------------------|:---------------|:--------------|:--------------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| layer_0/width_16k/canonical   | jumprelu       |               | blocks.0.hook_resid_post  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/canonical   | jumprelu       |               | blocks.1.hook_resid_post  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/canonical   | jumprelu       |               | blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/canonical   | jumprelu       |               | blocks.3.hook_resid_post  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/canonical   | jumprelu       |               | blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/canonical   | jumprelu       |               | blocks.5.hook_resid_post  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/canonical   | jumprelu       |               | blocks.6.hook_resid_post  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/canonical   | jumprelu       |               | blocks.7.hook_resid_post  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/canonical   | jumprelu       |               | blocks.8.hook_resid_post  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/canonical   | jumprelu       |               | blocks.9.hook_resid_post  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/canonical  | jumprelu       |               | blocks.10.hook_resid_post |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/canonical  | jumprelu       |               | blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/canonical  | jumprelu       |               | blocks.12.hook_resid_post |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/canonical  | jumprelu       |               | blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/canonical  | jumprelu       |               | blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/canonical  | jumprelu       |               | blocks.15.hook_resid_post |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/canonical  | jumprelu       |               | blocks.16.hook_resid_post |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/canonical  | jumprelu       |               | blocks.17.hook_resid_post |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/canonical  | jumprelu       |               | blocks.18.hook_resid_post |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/canonical  | jumprelu       |               | blocks.19.hook_resid_post |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/canonical  | jumprelu       |               | blocks.20.hook_resid_post |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/canonical  | jumprelu       |               | blocks.21.hook_resid_post |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/canonical  | jumprelu       |               | blocks.22.hook_resid_post |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/canonical  | jumprelu       |               | blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/canonical  | jumprelu       |               | blocks.24.hook_resid_post |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/canonical  | jumprelu       |               | blocks.25.hook_resid_post |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_1m/canonical    | jumprelu       |               | blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_1m/canonical   | jumprelu       |               | blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_1m/canonical   | jumprelu       |               | blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_262k/canonical | jumprelu       |               | blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_32k/canonical  | jumprelu       |               | blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_524k/canonical | jumprelu       |               | blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/canonical   | jumprelu       |               | blocks.0.hook_resid_post  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/canonical   | jumprelu       |               | blocks.1.hook_resid_post  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/canonical   | jumprelu       |               | blocks.2.hook_resid_post  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/canonical   | jumprelu       |               | blocks.3.hook_resid_post  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/canonical   | jumprelu       |               | blocks.4.hook_resid_post  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/canonical   | jumprelu       |               | blocks.5.hook_resid_post  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/canonical   | jumprelu       |               | blocks.6.hook_resid_post  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/canonical   | jumprelu       |               | blocks.7.hook_resid_post  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/canonical   | jumprelu       |               | blocks.8.hook_resid_post  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/canonical   | jumprelu       |               | blocks.9.hook_resid_post  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/canonical  | jumprelu       |               | blocks.10.hook_resid_post |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/canonical  | jumprelu       |               | blocks.11.hook_resid_post |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/canonical  | jumprelu       |               | blocks.12.hook_resid_post |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/canonical  | jumprelu       |               | blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/canonical  | jumprelu       |               | blocks.14.hook_resid_post |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/canonical  | jumprelu       |               | blocks.15.hook_resid_post |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/canonical  | jumprelu       |               | blocks.16.hook_resid_post |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/canonical  | jumprelu       |               | blocks.17.hook_resid_post |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/canonical  | jumprelu       |               | blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/canonical  | jumprelu       |               | blocks.19.hook_resid_post |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/canonical  | jumprelu       |               | blocks.20.hook_resid_post |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/canonical  | jumprelu       |               | blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/canonical  | jumprelu       |               | blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/canonical  | jumprelu       |               | blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/canonical  | jumprelu       |               | blocks.24.hook_resid_post |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/canonical  | jumprelu       |               | blocks.25.hook_resid_post |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-2b-pt-res](https://huggingface.co/google/gemma-scope-2b-pt-res)

- **Huggingface Repo**: google/gemma-scope-2b-pt-res
- **model**: gemma-2-2b

| id                                 | architecture   | neuronpedia   | hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:-----------------------------------|:---------------|:--------------|:--------------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| layer_0/width_16k/average_l0_105   | jumprelu       |               | blocks.0.hook_resid_post  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_16k/average_l0_13    | jumprelu       |               | blocks.0.hook_resid_post  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_16k/average_l0_226   | jumprelu       |               | blocks.0.hook_resid_post  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_16k/average_l0_25    | jumprelu       |               | blocks.0.hook_resid_post  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_16k/average_l0_46    | jumprelu       |               | blocks.0.hook_resid_post  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_10    | jumprelu       |               | blocks.1.hook_resid_post  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_102   | jumprelu       |               | blocks.1.hook_resid_post  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_20    | jumprelu       |               | blocks.1.hook_resid_post  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_250   | jumprelu       |               | blocks.1.hook_resid_post  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_40    | jumprelu       |               | blocks.1.hook_resid_post  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_13    | jumprelu       |               | blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_141   | jumprelu       |               | blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_142   | jumprelu       |               | blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_24    | jumprelu       |               | blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_304   | jumprelu       |               | blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_53    | jumprelu       |               | blocks.2.hook_resid_post  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_14    | jumprelu       |               | blocks.3.hook_resid_post  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_142   | jumprelu       |               | blocks.3.hook_resid_post  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_28    | jumprelu       |               | blocks.3.hook_resid_post  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_315   | jumprelu       |               | blocks.3.hook_resid_post  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_59    | jumprelu       |               | blocks.3.hook_resid_post  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_124   | jumprelu       |               | blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_125   | jumprelu       |               | blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_17    | jumprelu       |               | blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_281   | jumprelu       |               | blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_31    | jumprelu       |               | blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_60    | jumprelu       |               | blocks.4.hook_resid_post  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_143   | jumprelu       |               | blocks.5.hook_resid_post  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_18    | jumprelu       |               | blocks.5.hook_resid_post  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_309   | jumprelu       |               | blocks.5.hook_resid_post  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_34    | jumprelu       |               | blocks.5.hook_resid_post  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_68    | jumprelu       |               | blocks.5.hook_resid_post  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_144   | jumprelu       |               | blocks.6.hook_resid_post  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_19    | jumprelu       |               | blocks.6.hook_resid_post  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_301   | jumprelu       |               | blocks.6.hook_resid_post  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_36    | jumprelu       |               | blocks.6.hook_resid_post  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_70    | jumprelu       |               | blocks.6.hook_resid_post  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_137   | jumprelu       |               | blocks.7.hook_resid_post  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_20    | jumprelu       |               | blocks.7.hook_resid_post  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_285   | jumprelu       |               | blocks.7.hook_resid_post  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_36    | jumprelu       |               | blocks.7.hook_resid_post  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_69    | jumprelu       |               | blocks.7.hook_resid_post  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_142   | jumprelu       |               | blocks.8.hook_resid_post  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_20    | jumprelu       |               | blocks.8.hook_resid_post  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_301   | jumprelu       |               | blocks.8.hook_resid_post  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_37    | jumprelu       |               | blocks.8.hook_resid_post  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_71    | jumprelu       |               | blocks.8.hook_resid_post  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_151   | jumprelu       |               | blocks.9.hook_resid_post  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_21    | jumprelu       |               | blocks.9.hook_resid_post  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_340   | jumprelu       |               | blocks.9.hook_resid_post  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_37    | jumprelu       |               | blocks.9.hook_resid_post  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_73    | jumprelu       |               | blocks.9.hook_resid_post  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_166  | jumprelu       |               | blocks.10.hook_resid_post |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_21   | jumprelu       |               | blocks.10.hook_resid_post |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_39   | jumprelu       |               | blocks.10.hook_resid_post |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_395  | jumprelu       |               | blocks.10.hook_resid_post |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_77   | jumprelu       |               | blocks.10.hook_resid_post |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_168  | jumprelu       |               | blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_22   | jumprelu       |               | blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_393  | jumprelu       |               | blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_41   | jumprelu       |               | blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_79   | jumprelu       |               | blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_80   | jumprelu       |               | blocks.11.hook_resid_post |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_176  | jumprelu       |               | blocks.12.hook_resid_post |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_22   | jumprelu       |               | blocks.12.hook_resid_post |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_41   | jumprelu       |               | blocks.12.hook_resid_post |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_445  | jumprelu       |               | blocks.12.hook_resid_post |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_82   | jumprelu       |               | blocks.12.hook_resid_post |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_173  | jumprelu       |               | blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_23   | jumprelu       |               | blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_403  | jumprelu       |               | blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_43   | jumprelu       |               | blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_83   | jumprelu       |               | blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_84   | jumprelu       |               | blocks.13.hook_resid_post |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_173  | jumprelu       |               | blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_23   | jumprelu       |               | blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_388  | jumprelu       |               | blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_43   | jumprelu       |               | blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_83   | jumprelu       |               | blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_84   | jumprelu       |               | blocks.14.hook_resid_post |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_150  | jumprelu       |               | blocks.15.hook_resid_post |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_23   | jumprelu       |               | blocks.15.hook_resid_post |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_308  | jumprelu       |               | blocks.15.hook_resid_post |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_41   | jumprelu       |               | blocks.15.hook_resid_post |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_78   | jumprelu       |               | blocks.15.hook_resid_post |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_154  | jumprelu       |               | blocks.16.hook_resid_post |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_23   | jumprelu       |               | blocks.16.hook_resid_post |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_335  | jumprelu       |               | blocks.16.hook_resid_post |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_42   | jumprelu       |               | blocks.16.hook_resid_post |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_78   | jumprelu       |               | blocks.16.hook_resid_post |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_150  | jumprelu       |               | blocks.17.hook_resid_post |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_23   | jumprelu       |               | blocks.17.hook_resid_post |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_304  | jumprelu       |               | blocks.17.hook_resid_post |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_42   | jumprelu       |               | blocks.17.hook_resid_post |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_77   | jumprelu       |               | blocks.17.hook_resid_post |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_138  | jumprelu       |               | blocks.18.hook_resid_post |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_23   | jumprelu       |               | blocks.18.hook_resid_post |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_280  | jumprelu       |               | blocks.18.hook_resid_post |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_40   | jumprelu       |               | blocks.18.hook_resid_post |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_74   | jumprelu       |               | blocks.18.hook_resid_post |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_137  | jumprelu       |               | blocks.19.hook_resid_post |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_23   | jumprelu       |               | blocks.19.hook_resid_post |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_279  | jumprelu       |               | blocks.19.hook_resid_post |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_40   | jumprelu       |               | blocks.19.hook_resid_post |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_73   | jumprelu       |               | blocks.19.hook_resid_post |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_139  | jumprelu       |               | blocks.20.hook_resid_post |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_22   | jumprelu       |               | blocks.20.hook_resid_post |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_294  | jumprelu       |               | blocks.20.hook_resid_post |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_38   | jumprelu       |               | blocks.20.hook_resid_post |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_71   | jumprelu       |               | blocks.20.hook_resid_post |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_139  | jumprelu       |               | blocks.21.hook_resid_post |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_22   | jumprelu       |               | blocks.21.hook_resid_post |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_301  | jumprelu       |               | blocks.21.hook_resid_post |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_38   | jumprelu       |               | blocks.21.hook_resid_post |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_70   | jumprelu       |               | blocks.21.hook_resid_post |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_147  | jumprelu       |               | blocks.22.hook_resid_post |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_21   | jumprelu       |               | blocks.22.hook_resid_post |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_349  | jumprelu       |               | blocks.22.hook_resid_post |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_38   | jumprelu       |               | blocks.22.hook_resid_post |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_72   | jumprelu       |               | blocks.22.hook_resid_post |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_157  | jumprelu       |               | blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_21   | jumprelu       |               | blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_38   | jumprelu       |               | blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_404  | jumprelu       |               | blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_74   | jumprelu       |               | blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_75   | jumprelu       |               | blocks.23.hook_resid_post |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_158  | jumprelu       |               | blocks.24.hook_resid_post |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_20   | jumprelu       |               | blocks.24.hook_resid_post |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_38   | jumprelu       |               | blocks.24.hook_resid_post |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_457  | jumprelu       |               | blocks.24.hook_resid_post |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_73   | jumprelu       |               | blocks.24.hook_resid_post |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_116  | jumprelu       |               | blocks.25.hook_resid_post |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_16   | jumprelu       |               | blocks.25.hook_resid_post |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_28   | jumprelu       |               | blocks.25.hook_resid_post |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_285  | jumprelu       |               | blocks.25.hook_resid_post |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_55   | jumprelu       |               | blocks.25.hook_resid_post |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_1m/average_l0_114    | jumprelu       |               | blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_1m/average_l0_13     | jumprelu       |               | blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_1m/average_l0_21     | jumprelu       |               | blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_1m/average_l0_36     | jumprelu       |               | blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_1m/average_l0_63     | jumprelu       |               | blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_1m/average_l0_9      | jumprelu       |               | blocks.5.hook_resid_post  |            5 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_1m/average_l0_107   | jumprelu       |               | blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_1m/average_l0_19    | jumprelu       |               | blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_1m/average_l0_207   | jumprelu       |               | blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_1m/average_l0_26    | jumprelu       |               | blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_1m/average_l0_58    | jumprelu       |               | blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_1m/average_l0_73    | jumprelu       |               | blocks.12.hook_resid_post |           12 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_1m/average_l0_157   | jumprelu       |               | blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_1m/average_l0_16    | jumprelu       |               | blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_1m/average_l0_18    | jumprelu       |               | blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_1m/average_l0_29    | jumprelu       |               | blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_1m/average_l0_50    | jumprelu       |               | blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_1m/average_l0_88    | jumprelu       |               | blocks.19.hook_resid_post |           19 | 1048576 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_262k/average_l0_11  | jumprelu       |               | blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_262k/average_l0_121 | jumprelu       |               | blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_262k/average_l0_21  | jumprelu       |               | blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_262k/average_l0_243 | jumprelu       |               | blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_262k/average_l0_36  | jumprelu       |               | blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_262k/average_l0_67  | jumprelu       |               | blocks.12.hook_resid_post |           12 |  262144 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_32k/average_l0_12   | jumprelu       |               | blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_32k/average_l0_155  | jumprelu       |               | blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_32k/average_l0_22   | jumprelu       |               | blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_32k/average_l0_360  | jumprelu       |               | blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_32k/average_l0_40   | jumprelu       |               | blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_32k/average_l0_76   | jumprelu       |               | blocks.12.hook_resid_post |           12 |   32768 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_524k/average_l0_115 | jumprelu       |               | blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_524k/average_l0_22  | jumprelu       |               | blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_524k/average_l0_227 | jumprelu       |               | blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_524k/average_l0_29  | jumprelu       |               | blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_524k/average_l0_46  | jumprelu       |               | blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_524k/average_l0_65  | jumprelu       |               | blocks.12.hook_resid_post |           12 |  524288 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_11    | jumprelu       |               | blocks.0.hook_resid_post  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_17    | jumprelu       |               | blocks.0.hook_resid_post  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_27    | jumprelu       |               | blocks.0.hook_resid_post  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_43    | jumprelu       |               | blocks.0.hook_resid_post  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_73    | jumprelu       |               | blocks.0.hook_resid_post  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_121   | jumprelu       |               | blocks.1.hook_resid_post  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_16    | jumprelu       |               | blocks.1.hook_resid_post  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_30    | jumprelu       |               | blocks.1.hook_resid_post  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_54    | jumprelu       |               | blocks.1.hook_resid_post  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_9     | jumprelu       |               | blocks.1.hook_resid_post  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_11    | jumprelu       |               | blocks.2.hook_resid_post  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_169   | jumprelu       |               | blocks.2.hook_resid_post  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_20    | jumprelu       |               | blocks.2.hook_resid_post  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_37    | jumprelu       |               | blocks.2.hook_resid_post  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_77    | jumprelu       |               | blocks.2.hook_resid_post  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_13    | jumprelu       |               | blocks.3.hook_resid_post  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_193   | jumprelu       |               | blocks.3.hook_resid_post  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_23    | jumprelu       |               | blocks.3.hook_resid_post  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_42    | jumprelu       |               | blocks.3.hook_resid_post  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_89    | jumprelu       |               | blocks.3.hook_resid_post  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_14    | jumprelu       |               | blocks.4.hook_resid_post  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_177   | jumprelu       |               | blocks.4.hook_resid_post  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_25    | jumprelu       |               | blocks.4.hook_resid_post  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_46    | jumprelu       |               | blocks.4.hook_resid_post  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_89    | jumprelu       |               | blocks.4.hook_resid_post  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_105   | jumprelu       |               | blocks.5.hook_resid_post  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_17    | jumprelu       |               | blocks.5.hook_resid_post  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_211   | jumprelu       |               | blocks.5.hook_resid_post  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_29    | jumprelu       |               | blocks.5.hook_resid_post  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_53    | jumprelu       |               | blocks.5.hook_resid_post  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_107   | jumprelu       |               | blocks.6.hook_resid_post  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_17    | jumprelu       |               | blocks.6.hook_resid_post  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_208   | jumprelu       |               | blocks.6.hook_resid_post  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_30    | jumprelu       |               | blocks.6.hook_resid_post  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_56    | jumprelu       |               | blocks.6.hook_resid_post  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_107   | jumprelu       |               | blocks.7.hook_resid_post  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_18    | jumprelu       |               | blocks.7.hook_resid_post  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_203   | jumprelu       |               | blocks.7.hook_resid_post  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_31    | jumprelu       |               | blocks.7.hook_resid_post  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_57    | jumprelu       |               | blocks.7.hook_resid_post  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_111   | jumprelu       |               | blocks.8.hook_resid_post  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_19    | jumprelu       |               | blocks.8.hook_resid_post  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_213   | jumprelu       |               | blocks.8.hook_resid_post  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_33    | jumprelu       |               | blocks.8.hook_resid_post  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_59    | jumprelu       |               | blocks.8.hook_resid_post  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_118   | jumprelu       |               | blocks.9.hook_resid_post  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_19    | jumprelu       |               | blocks.9.hook_resid_post  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_240   | jumprelu       |               | blocks.9.hook_resid_post  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_34    | jumprelu       |               | blocks.9.hook_resid_post  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_61    | jumprelu       |               | blocks.9.hook_resid_post  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_128  | jumprelu       |               | blocks.10.hook_resid_post |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_20   | jumprelu       |               | blocks.10.hook_resid_post |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_265  | jumprelu       |               | blocks.10.hook_resid_post |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_36   | jumprelu       |               | blocks.10.hook_resid_post |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_66   | jumprelu       |               | blocks.10.hook_resid_post |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_134  | jumprelu       |               | blocks.11.hook_resid_post |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_21   | jumprelu       |               | blocks.11.hook_resid_post |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_273  | jumprelu       |               | blocks.11.hook_resid_post |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_37   | jumprelu       |               | blocks.11.hook_resid_post |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_70   | jumprelu       |               | blocks.11.hook_resid_post |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_141  | jumprelu       |               | blocks.12.hook_resid_post |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_21   | jumprelu       |               | blocks.12.hook_resid_post |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_297  | jumprelu       |               | blocks.12.hook_resid_post |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_38   | jumprelu       |               | blocks.12.hook_resid_post |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_72   | jumprelu       |               | blocks.12.hook_resid_post |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_142  | jumprelu       |               | blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_22   | jumprelu       |               | blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_288  | jumprelu       |               | blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_40   | jumprelu       |               | blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_74   | jumprelu       |               | blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_75   | jumprelu       |               | blocks.13.hook_resid_post |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_144  | jumprelu       |               | blocks.14.hook_resid_post |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_21   | jumprelu       |               | blocks.14.hook_resid_post |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_284  | jumprelu       |               | blocks.14.hook_resid_post |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_40   | jumprelu       |               | blocks.14.hook_resid_post |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_73   | jumprelu       |               | blocks.14.hook_resid_post |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_127  | jumprelu       |               | blocks.15.hook_resid_post |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_21   | jumprelu       |               | blocks.15.hook_resid_post |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_240  | jumprelu       |               | blocks.15.hook_resid_post |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_38   | jumprelu       |               | blocks.15.hook_resid_post |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_68   | jumprelu       |               | blocks.15.hook_resid_post |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_128  | jumprelu       |               | blocks.16.hook_resid_post |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_21   | jumprelu       |               | blocks.16.hook_resid_post |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_244  | jumprelu       |               | blocks.16.hook_resid_post |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_38   | jumprelu       |               | blocks.16.hook_resid_post |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_69   | jumprelu       |               | blocks.16.hook_resid_post |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_125  | jumprelu       |               | blocks.17.hook_resid_post |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_21   | jumprelu       |               | blocks.17.hook_resid_post |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_233  | jumprelu       |               | blocks.17.hook_resid_post |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_38   | jumprelu       |               | blocks.17.hook_resid_post |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_68   | jumprelu       |               | blocks.17.hook_resid_post |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_116  | jumprelu       |               | blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_117  | jumprelu       |               | blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_21   | jumprelu       |               | blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_216  | jumprelu       |               | blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_36   | jumprelu       |               | blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_64   | jumprelu       |               | blocks.18.hook_resid_post |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_115  | jumprelu       |               | blocks.19.hook_resid_post |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_21   | jumprelu       |               | blocks.19.hook_resid_post |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_216  | jumprelu       |               | blocks.19.hook_resid_post |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_35   | jumprelu       |               | blocks.19.hook_resid_post |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_63   | jumprelu       |               | blocks.19.hook_resid_post |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_114  | jumprelu       |               | blocks.20.hook_resid_post |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_20   | jumprelu       |               | blocks.20.hook_resid_post |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_221  | jumprelu       |               | blocks.20.hook_resid_post |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_34   | jumprelu       |               | blocks.20.hook_resid_post |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_61   | jumprelu       |               | blocks.20.hook_resid_post |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_111  | jumprelu       |               | blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_112  | jumprelu       |               | blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_20   | jumprelu       |               | blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_225  | jumprelu       |               | blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_33   | jumprelu       |               | blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_61   | jumprelu       |               | blocks.21.hook_resid_post |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_116  | jumprelu       |               | blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_117  | jumprelu       |               | blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_20   | jumprelu       |               | blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_248  | jumprelu       |               | blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_33   | jumprelu       |               | blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_62   | jumprelu       |               | blocks.22.hook_resid_post |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_123  | jumprelu       |               | blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_124  | jumprelu       |               | blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_20   | jumprelu       |               | blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_272  | jumprelu       |               | blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_35   | jumprelu       |               | blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_64   | jumprelu       |               | blocks.23.hook_resid_post |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_124  | jumprelu       |               | blocks.24.hook_resid_post |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_19   | jumprelu       |               | blocks.24.hook_resid_post |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_273  | jumprelu       |               | blocks.24.hook_resid_post |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_34   | jumprelu       |               | blocks.24.hook_resid_post |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_63   | jumprelu       |               | blocks.24.hook_resid_post |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_15   | jumprelu       |               | blocks.25.hook_resid_post |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_197  | jumprelu       |               | blocks.25.hook_resid_post |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_26   | jumprelu       |               | blocks.25.hook_resid_post |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_48   | jumprelu       |               | blocks.25.hook_resid_post |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_93   | jumprelu       |               | blocks.25.hook_resid_post |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-2b-pt-mlp-canonical](https://huggingface.co/google/gemma-scope-2b-pt-mlp)

- **Huggingface Repo**: google/gemma-scope-2b-pt-mlp
- **model**: gemma-2-2b

| id                           | architecture   | neuronpedia   | hook_name              |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:-----------------------------|:---------------|:--------------|:-----------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| layer_0/width_16k/canonical  | jumprelu       |               | blocks.0.hook_mlp_out  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/canonical  | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/canonical  | jumprelu       |               | blocks.2.hook_mlp_out  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/canonical  | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/canonical  | jumprelu       |               | blocks.4.hook_mlp_out  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/canonical  | jumprelu       |               | blocks.5.hook_mlp_out  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/canonical  | jumprelu       |               | blocks.6.hook_mlp_out  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/canonical  | jumprelu       |               | blocks.7.hook_mlp_out  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/canonical  | jumprelu       |               | blocks.8.hook_mlp_out  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/canonical  | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/canonical | jumprelu       |               | blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/canonical | jumprelu       |               | blocks.11.hook_mlp_out |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/canonical | jumprelu       |               | blocks.12.hook_mlp_out |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/canonical | jumprelu       |               | blocks.13.hook_mlp_out |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/canonical | jumprelu       |               | blocks.14.hook_mlp_out |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/canonical | jumprelu       |               | blocks.15.hook_mlp_out |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/canonical | jumprelu       |               | blocks.16.hook_mlp_out |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/canonical | jumprelu       |               | blocks.17.hook_mlp_out |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/canonical | jumprelu       |               | blocks.18.hook_mlp_out |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/canonical | jumprelu       |               | blocks.19.hook_mlp_out |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/canonical | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/canonical | jumprelu       |               | blocks.21.hook_mlp_out |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/canonical | jumprelu       |               | blocks.22.hook_mlp_out |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/canonical | jumprelu       |               | blocks.23.hook_mlp_out |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/canonical | jumprelu       |               | blocks.24.hook_mlp_out |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/canonical | jumprelu       |               | blocks.25.hook_mlp_out |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/canonical  | jumprelu       |               | blocks.0.hook_mlp_out  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/canonical  | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/canonical  | jumprelu       |               | blocks.2.hook_mlp_out  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/canonical  | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/canonical  | jumprelu       |               | blocks.4.hook_mlp_out  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/canonical  | jumprelu       |               | blocks.5.hook_mlp_out  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/canonical  | jumprelu       |               | blocks.6.hook_mlp_out  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/canonical  | jumprelu       |               | blocks.7.hook_mlp_out  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/canonical  | jumprelu       |               | blocks.8.hook_mlp_out  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/canonical  | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/canonical | jumprelu       |               | blocks.10.hook_mlp_out |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/canonical | jumprelu       |               | blocks.11.hook_mlp_out |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/canonical | jumprelu       |               | blocks.12.hook_mlp_out |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/canonical | jumprelu       |               | blocks.13.hook_mlp_out |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/canonical | jumprelu       |               | blocks.14.hook_mlp_out |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/canonical | jumprelu       |               | blocks.15.hook_mlp_out |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/canonical | jumprelu       |               | blocks.16.hook_mlp_out |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/canonical | jumprelu       |               | blocks.17.hook_mlp_out |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/canonical | jumprelu       |               | blocks.18.hook_mlp_out |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/canonical | jumprelu       |               | blocks.19.hook_mlp_out |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/canonical | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/canonical | jumprelu       |               | blocks.21.hook_mlp_out |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/canonical | jumprelu       |               | blocks.22.hook_mlp_out |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/canonical | jumprelu       |               | blocks.23.hook_mlp_out |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/canonical | jumprelu       |               | blocks.24.hook_mlp_out |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/canonical | jumprelu       |               | blocks.25.hook_mlp_out |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-2b-pt-mlp](https://huggingface.co/google/gemma-scope-2b-pt-mlp)

- **Huggingface Repo**: google/gemma-scope-2b-pt-mlp
- **model**: gemma-2-2b

| id                                | architecture   | neuronpedia   | hook_name              |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:----------------------------------|:---------------|:--------------|:-----------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| layer_0/width_16k/average_l0_119  | jumprelu       |               | blocks.0.hook_mlp_out  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_16k/average_l0_16   | jumprelu       |               | blocks.0.hook_mlp_out  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_16k/average_l0_30   | jumprelu       |               | blocks.0.hook_mlp_out  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_16k/average_l0_60   | jumprelu       |               | blocks.0.hook_mlp_out  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_16k/average_l0_9    | jumprelu       |               | blocks.0.hook_mlp_out  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_105  | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_12   | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_239  | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_24   | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_50   | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_19   | jumprelu       |               | blocks.2.hook_mlp_out  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_213  | jumprelu       |               | blocks.2.hook_mlp_out  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_41   | jumprelu       |               | blocks.2.hook_mlp_out  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_434  | jumprelu       |               | blocks.2.hook_mlp_out  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_95   | jumprelu       |               | blocks.2.hook_mlp_out  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_195  | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_21   | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_377  | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_44   | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_95   | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_18   | jumprelu       |               | blocks.4.hook_mlp_out  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_198  | jumprelu       |               | blocks.4.hook_mlp_out  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_38   | jumprelu       |               | blocks.4.hook_mlp_out  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_433  | jumprelu       |               | blocks.4.hook_mlp_out  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_85   | jumprelu       |               | blocks.4.hook_mlp_out  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_114  | jumprelu       |               | blocks.5.hook_mlp_out  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_23   | jumprelu       |               | blocks.5.hook_mlp_out  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_269  | jumprelu       |               | blocks.5.hook_mlp_out  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_48   | jumprelu       |               | blocks.5.hook_mlp_out  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_575  | jumprelu       |               | blocks.5.hook_mlp_out  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_133  | jumprelu       |               | blocks.6.hook_mlp_out  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_25   | jumprelu       |               | blocks.6.hook_mlp_out  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_328  | jumprelu       |               | blocks.6.hook_mlp_out  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_55   | jumprelu       |               | blocks.6.hook_mlp_out  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_699  | jumprelu       |               | blocks.6.hook_mlp_out  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_146  | jumprelu       |               | blocks.7.hook_mlp_out  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_28   | jumprelu       |               | blocks.7.hook_mlp_out  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_355  | jumprelu       |               | blocks.7.hook_mlp_out  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_60   | jumprelu       |               | blocks.7.hook_mlp_out  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_731  | jumprelu       |               | blocks.7.hook_mlp_out  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_136  | jumprelu       |               | blocks.8.hook_mlp_out  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_27   | jumprelu       |               | blocks.8.hook_mlp_out  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_351  | jumprelu       |               | blocks.8.hook_mlp_out  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_56   | jumprelu       |               | blocks.8.hook_mlp_out  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_739  | jumprelu       |               | blocks.8.hook_mlp_out  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_216  | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_38   | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_482  | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_861  | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_88   | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_110 | jumprelu       |               | blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_266 | jumprelu       |               | blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_45  | jumprelu       |               | blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_568 | jumprelu       |               | blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_908 | jumprelu       |               | blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_234 | jumprelu       |               | blocks.11.hook_mlp_out |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_42  | jumprelu       |               | blocks.11.hook_mlp_out |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_499 | jumprelu       |               | blocks.11.hook_mlp_out |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_847 | jumprelu       |               | blocks.11.hook_mlp_out |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_98  | jumprelu       |               | blocks.11.hook_mlp_out |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_108 | jumprelu       |               | blocks.12.hook_mlp_out |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_262 | jumprelu       |               | blocks.12.hook_mlp_out |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_44  | jumprelu       |               | blocks.12.hook_mlp_out |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_548 | jumprelu       |               | blocks.12.hook_mlp_out |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_879 | jumprelu       |               | blocks.12.hook_mlp_out |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_112 | jumprelu       |               | blocks.13.hook_mlp_out |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_267 | jumprelu       |               | blocks.13.hook_mlp_out |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_47  | jumprelu       |               | blocks.13.hook_mlp_out |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_553 | jumprelu       |               | blocks.13.hook_mlp_out |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_892 | jumprelu       |               | blocks.13.hook_mlp_out |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_246 | jumprelu       |               | blocks.14.hook_mlp_out |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_41  | jumprelu       |               | blocks.14.hook_mlp_out |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_536 | jumprelu       |               | blocks.14.hook_mlp_out |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_894 | jumprelu       |               | blocks.14.hook_mlp_out |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_97  | jumprelu       |               | blocks.14.hook_mlp_out |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_207 | jumprelu       |               | blocks.15.hook_mlp_out |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_35  | jumprelu       |               | blocks.15.hook_mlp_out |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_492 | jumprelu       |               | blocks.15.hook_mlp_out |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_80  | jumprelu       |               | blocks.15.hook_mlp_out |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_879 | jumprelu       |               | blocks.15.hook_mlp_out |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_185 | jumprelu       |               | blocks.16.hook_mlp_out |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_33  | jumprelu       |               | blocks.16.hook_mlp_out |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_452 | jumprelu       |               | blocks.16.hook_mlp_out |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_72  | jumprelu       |               | blocks.16.hook_mlp_out |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_847 | jumprelu       |               | blocks.16.hook_mlp_out |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_179 | jumprelu       |               | blocks.17.hook_mlp_out |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_31  | jumprelu       |               | blocks.17.hook_mlp_out |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_453 | jumprelu       |               | blocks.17.hook_mlp_out |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_68  | jumprelu       |               | blocks.17.hook_mlp_out |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_853 | jumprelu       |               | blocks.17.hook_mlp_out |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_106 | jumprelu       |               | blocks.18.hook_mlp_out |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_24  | jumprelu       |               | blocks.18.hook_mlp_out |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_292 | jumprelu       |               | blocks.18.hook_mlp_out |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_47  | jumprelu       |               | blocks.18.hook_mlp_out |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_672 | jumprelu       |               | blocks.18.hook_mlp_out |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_109 | jumprelu       |               | blocks.19.hook_mlp_out |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_25  | jumprelu       |               | blocks.19.hook_mlp_out |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_295 | jumprelu       |               | blocks.19.hook_mlp_out |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_50  | jumprelu       |               | blocks.19.hook_mlp_out |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_673 | jumprelu       |               | blocks.19.hook_mlp_out |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_109 | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_24  | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_289 | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_49  | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_658 | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_113 | jumprelu       |               | blocks.21.hook_mlp_out |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_23  | jumprelu       |               | blocks.21.hook_mlp_out |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_279 | jumprelu       |               | blocks.21.hook_mlp_out |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_48  | jumprelu       |               | blocks.21.hook_mlp_out |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_633 | jumprelu       |               | blocks.21.hook_mlp_out |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_121 | jumprelu       |               | blocks.22.hook_mlp_out |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_24  | jumprelu       |               | blocks.22.hook_mlp_out |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_290 | jumprelu       |               | blocks.22.hook_mlp_out |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_51  | jumprelu       |               | blocks.22.hook_mlp_out |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_624 | jumprelu       |               | blocks.22.hook_mlp_out |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_128 | jumprelu       |               | blocks.23.hook_mlp_out |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_27  | jumprelu       |               | blocks.23.hook_mlp_out |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_287 | jumprelu       |               | blocks.23.hook_mlp_out |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_57  | jumprelu       |               | blocks.23.hook_mlp_out |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_627 | jumprelu       |               | blocks.23.hook_mlp_out |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_158 | jumprelu       |               | blocks.24.hook_mlp_out |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_19  | jumprelu       |               | blocks.24.hook_mlp_out |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_35  | jumprelu       |               | blocks.24.hook_mlp_out |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_357 | jumprelu       |               | blocks.24.hook_mlp_out |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_73  | jumprelu       |               | blocks.24.hook_mlp_out |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_126 | jumprelu       |               | blocks.25.hook_mlp_out |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_15  | jumprelu       |               | blocks.25.hook_mlp_out |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_277 | jumprelu       |               | blocks.25.hook_mlp_out |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_29  | jumprelu       |               | blocks.25.hook_mlp_out |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_59  | jumprelu       |               | blocks.25.hook_mlp_out |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_12   | jumprelu       |               | blocks.0.hook_mlp_out  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_21   | jumprelu       |               | blocks.0.hook_mlp_out  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_39   | jumprelu       |               | blocks.0.hook_mlp_out  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_7    | jumprelu       |               | blocks.0.hook_mlp_out  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_72   | jumprelu       |               | blocks.0.hook_mlp_out  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_11   | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_127  | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_20   | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_37   | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_67   | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_134  | jumprelu       |               | blocks.2.hook_mlp_out  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_16   | jumprelu       |               | blocks.2.hook_mlp_out  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_265  | jumprelu       |               | blocks.2.hook_mlp_out  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_31   | jumprelu       |               | blocks.2.hook_mlp_out  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_60   | jumprelu       |               | blocks.2.hook_mlp_out  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_144  | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_18   | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_279  | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_33   | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_68   | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_138  | jumprelu       |               | blocks.4.hook_mlp_out  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_17   | jumprelu       |               | blocks.4.hook_mlp_out  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_299  | jumprelu       |               | blocks.4.hook_mlp_out  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_32   | jumprelu       |               | blocks.4.hook_mlp_out  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_66   | jumprelu       |               | blocks.4.hook_mlp_out  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_186  | jumprelu       |               | blocks.5.hook_mlp_out  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_22   | jumprelu       |               | blocks.5.hook_mlp_out  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_407  | jumprelu       |               | blocks.5.hook_mlp_out  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_43   | jumprelu       |               | blocks.5.hook_mlp_out  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_86   | jumprelu       |               | blocks.5.hook_mlp_out  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_101  | jumprelu       |               | blocks.6.hook_mlp_out  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_224  | jumprelu       |               | blocks.6.hook_mlp_out  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_24   | jumprelu       |               | blocks.6.hook_mlp_out  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_47   | jumprelu       |               | blocks.6.hook_mlp_out  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_515  | jumprelu       |               | blocks.6.hook_mlp_out  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_115  | jumprelu       |               | blocks.7.hook_mlp_out  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_266  | jumprelu       |               | blocks.7.hook_mlp_out  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_28   | jumprelu       |               | blocks.7.hook_mlp_out  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_56   | jumprelu       |               | blocks.7.hook_mlp_out  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_571  | jumprelu       |               | blocks.7.hook_mlp_out  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_110  | jumprelu       |               | blocks.8.hook_mlp_out  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_256  | jumprelu       |               | blocks.8.hook_mlp_out  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_31   | jumprelu       |               | blocks.8.hook_mlp_out  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_547  | jumprelu       |               | blocks.8.hook_mlp_out  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_55   | jumprelu       |               | blocks.8.hook_mlp_out  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_168  | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_38   | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_387  | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_745  | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_77   | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_218 | jumprelu       |               | blocks.10.hook_mlp_out |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_43  | jumprelu       |               | blocks.10.hook_mlp_out |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_474 | jumprelu       |               | blocks.10.hook_mlp_out |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_851 | jumprelu       |               | blocks.10.hook_mlp_out |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_95  | jumprelu       |               | blocks.10.hook_mlp_out |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_200 | jumprelu       |               | blocks.11.hook_mlp_out |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_41  | jumprelu       |               | blocks.11.hook_mlp_out |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_436 | jumprelu       |               | blocks.11.hook_mlp_out |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_771 | jumprelu       |               | blocks.11.hook_mlp_out |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_88  | jumprelu       |               | blocks.11.hook_mlp_out |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_222 | jumprelu       |               | blocks.12.hook_mlp_out |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_44  | jumprelu       |               | blocks.12.hook_mlp_out |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_482 | jumprelu       |               | blocks.12.hook_mlp_out |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_848 | jumprelu       |               | blocks.12.hook_mlp_out |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_96  | jumprelu       |               | blocks.12.hook_mlp_out |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_228 | jumprelu       |               | blocks.13.hook_mlp_out |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_44  | jumprelu       |               | blocks.13.hook_mlp_out |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_480 | jumprelu       |               | blocks.13.hook_mlp_out |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_841 | jumprelu       |               | blocks.13.hook_mlp_out |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_98  | jumprelu       |               | blocks.13.hook_mlp_out |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_204 | jumprelu       |               | blocks.14.hook_mlp_out |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_39  | jumprelu       |               | blocks.14.hook_mlp_out |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_463 | jumprelu       |               | blocks.14.hook_mlp_out |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_816 | jumprelu       |               | blocks.14.hook_mlp_out |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_89  | jumprelu       |               | blocks.14.hook_mlp_out |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_164 | jumprelu       |               | blocks.15.hook_mlp_out |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_35  | jumprelu       |               | blocks.15.hook_mlp_out |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_405 | jumprelu       |               | blocks.15.hook_mlp_out |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_72  | jumprelu       |               | blocks.15.hook_mlp_out |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_754 | jumprelu       |               | blocks.15.hook_mlp_out |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_142 | jumprelu       |               | blocks.16.hook_mlp_out |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_32  | jumprelu       |               | blocks.16.hook_mlp_out |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_348 | jumprelu       |               | blocks.16.hook_mlp_out |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_66  | jumprelu       |               | blocks.16.hook_mlp_out |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_695 | jumprelu       |               | blocks.16.hook_mlp_out |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_136 | jumprelu       |               | blocks.17.hook_mlp_out |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_30  | jumprelu       |               | blocks.17.hook_mlp_out |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_342 | jumprelu       |               | blocks.17.hook_mlp_out |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_61  | jumprelu       |               | blocks.17.hook_mlp_out |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_666 | jumprelu       |               | blocks.17.hook_mlp_out |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_191 | jumprelu       |               | blocks.18.hook_mlp_out |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_24  | jumprelu       |               | blocks.18.hook_mlp_out |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_44  | jumprelu       |               | blocks.18.hook_mlp_out |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_491 | jumprelu       |               | blocks.18.hook_mlp_out |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_88  | jumprelu       |               | blocks.18.hook_mlp_out |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_192 | jumprelu       |               | blocks.19.hook_mlp_out |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_25  | jumprelu       |               | blocks.19.hook_mlp_out |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_45  | jumprelu       |               | blocks.19.hook_mlp_out |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_470 | jumprelu       |               | blocks.19.hook_mlp_out |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_88  | jumprelu       |               | blocks.19.hook_mlp_out |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_189 | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_23  | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_44  | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_446 | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_88  | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_192 | jumprelu       |               | blocks.21.hook_mlp_out |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_23  | jumprelu       |               | blocks.21.hook_mlp_out |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_42  | jumprelu       |               | blocks.21.hook_mlp_out |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_472 | jumprelu       |               | blocks.21.hook_mlp_out |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_86  | jumprelu       |               | blocks.21.hook_mlp_out |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_203 | jumprelu       |               | blocks.22.hook_mlp_out |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_23  | jumprelu       |               | blocks.22.hook_mlp_out |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_46  | jumprelu       |               | blocks.22.hook_mlp_out |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_487 | jumprelu       |               | blocks.22.hook_mlp_out |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_92  | jumprelu       |               | blocks.22.hook_mlp_out |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_102 | jumprelu       |               | blocks.23.hook_mlp_out |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_218 | jumprelu       |               | blocks.23.hook_mlp_out |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_25  | jumprelu       |               | blocks.23.hook_mlp_out |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_49  | jumprelu       |               | blocks.23.hook_mlp_out |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_497 | jumprelu       |               | blocks.23.hook_mlp_out |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_128 | jumprelu       |               | blocks.24.hook_mlp_out |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_18  | jumprelu       |               | blocks.24.hook_mlp_out |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_268 | jumprelu       |               | blocks.24.hook_mlp_out |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_32  | jumprelu       |               | blocks.24.hook_mlp_out |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_62  | jumprelu       |               | blocks.24.hook_mlp_out |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_107 | jumprelu       |               | blocks.25.hook_mlp_out |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_14  | jumprelu       |               | blocks.25.hook_mlp_out |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_215 | jumprelu       |               | blocks.25.hook_mlp_out |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_26  | jumprelu       |               | blocks.25.hook_mlp_out |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_52  | jumprelu       |               | blocks.25.hook_mlp_out |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-2b-pt-att](https://huggingface.co/google/gemma-scope-2b-pt-att)

- **Huggingface Repo**: google/gemma-scope-2b-pt-att
- **model**: gemma-2-2b

| id                                | architecture   | neuronpedia   | hook_name             |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:----------------------------------|:---------------|:--------------|:----------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| layer_0/width_16k/average_l0_104  | jumprelu       |               | blocks.0.attn.hook_z  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_16k/average_l0_12   | jumprelu       |               | blocks.0.attn.hook_z  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_16k/average_l0_18   | jumprelu       |               | blocks.0.attn.hook_z  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_16k/average_l0_30   | jumprelu       |               | blocks.0.attn.hook_z  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_16k/average_l0_57   | jumprelu       |               | blocks.0.attn.hook_z  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_146  | jumprelu       |               | blocks.1.attn.hook_z  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_20   | jumprelu       |               | blocks.1.attn.hook_z  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_251  | jumprelu       |               | blocks.1.attn.hook_z  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_40   | jumprelu       |               | blocks.1.attn.hook_z  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_79   | jumprelu       |               | blocks.1.attn.hook_z  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_174  | jumprelu       |               | blocks.2.attn.hook_z  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_19   | jumprelu       |               | blocks.2.attn.hook_z  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_297  | jumprelu       |               | blocks.2.attn.hook_z  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_43   | jumprelu       |               | blocks.2.attn.hook_z  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_93   | jumprelu       |               | blocks.2.attn.hook_z  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_117  | jumprelu       |               | blocks.3.attn.hook_z  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_219  | jumprelu       |               | blocks.3.attn.hook_z  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_24   | jumprelu       |               | blocks.3.attn.hook_z  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_386  | jumprelu       |               | blocks.3.attn.hook_z  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_55   | jumprelu       |               | blocks.3.attn.hook_z  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_116  | jumprelu       |               | blocks.4.attn.hook_z  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_249  | jumprelu       |               | blocks.4.attn.hook_z  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_26   | jumprelu       |               | blocks.4.attn.hook_z  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_454  | jumprelu       |               | blocks.4.attn.hook_z  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_53   | jumprelu       |               | blocks.4.attn.hook_z  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_135  | jumprelu       |               | blocks.5.attn.hook_z  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_268  | jumprelu       |               | blocks.5.attn.hook_z  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_30   | jumprelu       |               | blocks.5.attn.hook_z  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_449  | jumprelu       |               | blocks.5.attn.hook_z  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_59   | jumprelu       |               | blocks.5.attn.hook_z  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_143  | jumprelu       |               | blocks.6.attn.hook_z  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_292  | jumprelu       |               | blocks.6.attn.hook_z  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_30   | jumprelu       |               | blocks.6.attn.hook_z  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_479  | jumprelu       |               | blocks.6.attn.hook_z  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_61   | jumprelu       |               | blocks.6.attn.hook_z  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_184  | jumprelu       |               | blocks.7.attn.hook_z  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_331  | jumprelu       |               | blocks.7.attn.hook_z  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_46   | jumprelu       |               | blocks.7.attn.hook_z  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_537  | jumprelu       |               | blocks.7.attn.hook_z  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_99   | jumprelu       |               | blocks.7.attn.hook_z  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_129  | jumprelu       |               | blocks.8.attn.hook_z  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_282  | jumprelu       |               | blocks.8.attn.hook_z  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_32   | jumprelu       |               | blocks.8.attn.hook_z  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_482  | jumprelu       |               | blocks.8.attn.hook_z  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_64   | jumprelu       |               | blocks.8.attn.hook_z  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_127  | jumprelu       |               | blocks.9.attn.hook_z  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_270  | jumprelu       |               | blocks.9.attn.hook_z  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_34   | jumprelu       |               | blocks.9.attn.hook_z  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_499  | jumprelu       |               | blocks.9.attn.hook_z  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_64   | jumprelu       |               | blocks.9.attn.hook_z  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_148 | jumprelu       |               | blocks.10.attn.hook_z |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_307 | jumprelu       |               | blocks.10.attn.hook_z |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_36  | jumprelu       |               | blocks.10.attn.hook_z |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_541 | jumprelu       |               | blocks.10.attn.hook_z |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_70  | jumprelu       |               | blocks.10.attn.hook_z |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_170 | jumprelu       |               | blocks.11.attn.hook_z |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_350 | jumprelu       |               | blocks.11.attn.hook_z |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_41  | jumprelu       |               | blocks.11.attn.hook_z |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_593 | jumprelu       |               | blocks.11.attn.hook_z |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_80  | jumprelu       |               | blocks.11.attn.hook_z |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_184 | jumprelu       |               | blocks.12.attn.hook_z |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_328 | jumprelu       |               | blocks.12.attn.hook_z |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_41  | jumprelu       |               | blocks.12.attn.hook_z |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_514 | jumprelu       |               | blocks.12.attn.hook_z |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_85  | jumprelu       |               | blocks.12.attn.hook_z |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_203 | jumprelu       |               | blocks.13.attn.hook_z |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_372 | jumprelu       |               | blocks.13.attn.hook_z |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_43  | jumprelu       |               | blocks.13.attn.hook_z |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_570 | jumprelu       |               | blocks.13.attn.hook_z |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_92  | jumprelu       |               | blocks.13.attn.hook_z |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_161 | jumprelu       |               | blocks.14.attn.hook_z |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_298 | jumprelu       |               | blocks.14.attn.hook_z |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_37  | jumprelu       |               | blocks.14.attn.hook_z |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_468 | jumprelu       |               | blocks.14.attn.hook_z |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_71  | jumprelu       |               | blocks.14.attn.hook_z |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_195 | jumprelu       |               | blocks.15.attn.hook_z |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_342 | jumprelu       |               | blocks.15.attn.hook_z |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_44  | jumprelu       |               | blocks.15.attn.hook_z |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_535 | jumprelu       |               | blocks.15.attn.hook_z |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_98  | jumprelu       |               | blocks.15.attn.hook_z |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_144 | jumprelu       |               | blocks.16.attn.hook_z |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_293 | jumprelu       |               | blocks.16.attn.hook_z |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_37  | jumprelu       |               | blocks.16.attn.hook_z |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_527 | jumprelu       |               | blocks.16.attn.hook_z |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_71  | jumprelu       |               | blocks.16.attn.hook_z |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_176 | jumprelu       |               | blocks.17.attn.hook_z |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_316 | jumprelu       |               | blocks.17.attn.hook_z |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_38  | jumprelu       |               | blocks.17.attn.hook_z |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_509 | jumprelu       |               | blocks.17.attn.hook_z |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_79  | jumprelu       |               | blocks.17.attn.hook_z |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_144 | jumprelu       |               | blocks.18.attn.hook_z |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_292 | jumprelu       |               | blocks.18.attn.hook_z |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_34  | jumprelu       |               | blocks.18.attn.hook_z |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_491 | jumprelu       |               | blocks.18.attn.hook_z |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_72  | jumprelu       |               | blocks.18.attn.hook_z |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_122 | jumprelu       |               | blocks.19.attn.hook_z |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_249 | jumprelu       |               | blocks.19.attn.hook_z |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_28  | jumprelu       |               | blocks.19.attn.hook_z |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_423 | jumprelu       |               | blocks.19.attn.hook_z |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_56  | jumprelu       |               | blocks.19.attn.hook_z |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_141 | jumprelu       |               | blocks.20.attn.hook_z |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_274 | jumprelu       |               | blocks.20.attn.hook_z |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_31  | jumprelu       |               | blocks.20.attn.hook_z |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_446 | jumprelu       |               | blocks.20.attn.hook_z |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_62  | jumprelu       |               | blocks.20.attn.hook_z |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_142 | jumprelu       |               | blocks.21.attn.hook_z |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_301 | jumprelu       |               | blocks.21.attn.hook_z |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_32  | jumprelu       |               | blocks.21.attn.hook_z |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_505 | jumprelu       |               | blocks.21.attn.hook_z |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_65  | jumprelu       |               | blocks.21.attn.hook_z |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_106 | jumprelu       |               | blocks.22.attn.hook_z |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_215 | jumprelu       |               | blocks.22.attn.hook_z |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_22  | jumprelu       |               | blocks.22.attn.hook_z |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_373 | jumprelu       |               | blocks.22.attn.hook_z |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_47  | jumprelu       |               | blocks.22.attn.hook_z |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_161 | jumprelu       |               | blocks.23.attn.hook_z |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_30  | jumprelu       |               | blocks.23.attn.hook_z |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_300 | jumprelu       |               | blocks.23.attn.hook_z |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_474 | jumprelu       |               | blocks.23.attn.hook_z |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_73  | jumprelu       |               | blocks.23.attn.hook_z |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_212 | jumprelu       |               | blocks.24.attn.hook_z |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_372 | jumprelu       |               | blocks.24.attn.hook_z |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_39  | jumprelu       |               | blocks.24.attn.hook_z |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_558 | jumprelu       |               | blocks.24.attn.hook_z |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_96  | jumprelu       |               | blocks.24.attn.hook_z |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_177 | jumprelu       |               | blocks.25.attn.hook_z |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_313 | jumprelu       |               | blocks.25.attn.hook_z |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_35  | jumprelu       |               | blocks.25.attn.hook_z |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_492 | jumprelu       |               | blocks.25.attn.hook_z |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_77  | jumprelu       |               | blocks.25.attn.hook_z |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_10   | jumprelu       |               | blocks.0.attn.hook_z  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_16   | jumprelu       |               | blocks.0.attn.hook_z  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_24   | jumprelu       |               | blocks.0.attn.hook_z  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_43   | jumprelu       |               | blocks.0.attn.hook_z  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/average_l0_75   | jumprelu       |               | blocks.0.attn.hook_z  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_15   | jumprelu       |               | blocks.1.attn.hook_z  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_181  | jumprelu       |               | blocks.1.attn.hook_z  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_28   | jumprelu       |               | blocks.1.attn.hook_z  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_55   | jumprelu       |               | blocks.1.attn.hook_z  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/average_l0_98   | jumprelu       |               | blocks.1.attn.hook_z  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_125  | jumprelu       |               | blocks.2.attn.hook_z  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_14   | jumprelu       |               | blocks.2.attn.hook_z  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_228  | jumprelu       |               | blocks.2.attn.hook_z  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_28   | jumprelu       |               | blocks.2.attn.hook_z  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/average_l0_59   | jumprelu       |               | blocks.2.attn.hook_z  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_174  | jumprelu       |               | blocks.3.attn.hook_z  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_19   | jumprelu       |               | blocks.3.attn.hook_z  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_320  | jumprelu       |               | blocks.3.attn.hook_z  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_39   | jumprelu       |               | blocks.3.attn.hook_z  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/average_l0_83   | jumprelu       |               | blocks.3.attn.hook_z  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_188  | jumprelu       |               | blocks.4.attn.hook_z  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_22   | jumprelu       |               | blocks.4.attn.hook_z  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_382  | jumprelu       |               | blocks.4.attn.hook_z  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_43   | jumprelu       |               | blocks.4.attn.hook_z  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/average_l0_87   | jumprelu       |               | blocks.4.attn.hook_z  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_227  | jumprelu       |               | blocks.5.attn.hook_z  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_28   | jumprelu       |               | blocks.5.attn.hook_z  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_400  | jumprelu       |               | blocks.5.attn.hook_z  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_51   | jumprelu       |               | blocks.5.attn.hook_z  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/average_l0_99   | jumprelu       |               | blocks.5.attn.hook_z  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_112  | jumprelu       |               | blocks.6.attn.hook_z  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_261  | jumprelu       |               | blocks.6.attn.hook_z  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_30   | jumprelu       |               | blocks.6.attn.hook_z  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_449  | jumprelu       |               | blocks.6.attn.hook_z  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/average_l0_55   | jumprelu       |               | blocks.6.attn.hook_z  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_176  | jumprelu       |               | blocks.7.attn.hook_z  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_311  | jumprelu       |               | blocks.7.attn.hook_z  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_519  | jumprelu       |               | blocks.7.attn.hook_z  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_52   | jumprelu       |               | blocks.7.attn.hook_z  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/average_l0_96   | jumprelu       |               | blocks.7.attn.hook_z  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_112  | jumprelu       |               | blocks.8.attn.hook_z  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_246  | jumprelu       |               | blocks.8.attn.hook_z  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_35   | jumprelu       |               | blocks.8.attn.hook_z  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_454  | jumprelu       |               | blocks.8.attn.hook_z  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/average_l0_56   | jumprelu       |               | blocks.8.attn.hook_z  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_107  | jumprelu       |               | blocks.9.attn.hook_z  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_231  | jumprelu       |               | blocks.9.attn.hook_z  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_31   | jumprelu       |               | blocks.9.attn.hook_z  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_454  | jumprelu       |               | blocks.9.attn.hook_z  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/average_l0_57   | jumprelu       |               | blocks.9.attn.hook_z  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_134 | jumprelu       |               | blocks.10.attn.hook_z |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_292 | jumprelu       |               | blocks.10.attn.hook_z |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_35  | jumprelu       |               | blocks.10.attn.hook_z |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_521 | jumprelu       |               | blocks.10.attn.hook_z |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/average_l0_67  | jumprelu       |               | blocks.10.attn.hook_z |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_154 | jumprelu       |               | blocks.11.attn.hook_z |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_330 | jumprelu       |               | blocks.11.attn.hook_z |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_41  | jumprelu       |               | blocks.11.attn.hook_z |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_576 | jumprelu       |               | blocks.11.attn.hook_z |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/average_l0_75  | jumprelu       |               | blocks.11.attn.hook_z |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_172 | jumprelu       |               | blocks.12.attn.hook_z |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_320 | jumprelu       |               | blocks.12.attn.hook_z |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_39  | jumprelu       |               | blocks.12.attn.hook_z |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_503 | jumprelu       |               | blocks.12.attn.hook_z |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/average_l0_79  | jumprelu       |               | blocks.12.attn.hook_z |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_191 | jumprelu       |               | blocks.13.attn.hook_z |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_363 | jumprelu       |               | blocks.13.attn.hook_z |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_41  | jumprelu       |               | blocks.13.attn.hook_z |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_556 | jumprelu       |               | blocks.13.attn.hook_z |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/average_l0_87  | jumprelu       |               | blocks.13.attn.hook_z |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_138 | jumprelu       |               | blocks.14.attn.hook_z |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_283 | jumprelu       |               | blocks.14.attn.hook_z |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_37  | jumprelu       |               | blocks.14.attn.hook_z |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_453 | jumprelu       |               | blocks.14.attn.hook_z |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/average_l0_66  | jumprelu       |               | blocks.14.attn.hook_z |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_182 | jumprelu       |               | blocks.15.attn.hook_z |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_327 | jumprelu       |               | blocks.15.attn.hook_z |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_42  | jumprelu       |               | blocks.15.attn.hook_z |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_517 | jumprelu       |               | blocks.15.attn.hook_z |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/average_l0_90  | jumprelu       |               | blocks.15.attn.hook_z |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_129 | jumprelu       |               | blocks.16.attn.hook_z |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_260 | jumprelu       |               | blocks.16.attn.hook_z |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_35  | jumprelu       |               | blocks.16.attn.hook_z |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_502 | jumprelu       |               | blocks.16.attn.hook_z |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/average_l0_64  | jumprelu       |               | blocks.16.attn.hook_z |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_157 | jumprelu       |               | blocks.17.attn.hook_z |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_293 | jumprelu       |               | blocks.17.attn.hook_z |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_35  | jumprelu       |               | blocks.17.attn.hook_z |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_489 | jumprelu       |               | blocks.17.attn.hook_z |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/average_l0_70  | jumprelu       |               | blocks.17.attn.hook_z |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_123 | jumprelu       |               | blocks.18.attn.hook_z |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_255 | jumprelu       |               | blocks.18.attn.hook_z |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_29  | jumprelu       |               | blocks.18.attn.hook_z |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_466 | jumprelu       |               | blocks.18.attn.hook_z |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/average_l0_58  | jumprelu       |               | blocks.18.attn.hook_z |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_106 | jumprelu       |               | blocks.19.attn.hook_z |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_220 | jumprelu       |               | blocks.19.attn.hook_z |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_26  | jumprelu       |               | blocks.19.attn.hook_z |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_411 | jumprelu       |               | blocks.19.attn.hook_z |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/average_l0_49  | jumprelu       |               | blocks.19.attn.hook_z |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_102 | jumprelu       |               | blocks.20.attn.hook_z |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_242 | jumprelu       |               | blocks.20.attn.hook_z |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_26  | jumprelu       |               | blocks.20.attn.hook_z |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_419 | jumprelu       |               | blocks.20.attn.hook_z |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/average_l0_49  | jumprelu       |               | blocks.20.attn.hook_z |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_118 | jumprelu       |               | blocks.21.attn.hook_z |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_266 | jumprelu       |               | blocks.21.attn.hook_z |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_29  | jumprelu       |               | blocks.21.attn.hook_z |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_474 | jumprelu       |               | blocks.21.attn.hook_z |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/average_l0_56  | jumprelu       |               | blocks.21.attn.hook_z |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_112 | jumprelu       |               | blocks.22.attn.hook_z |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_196 | jumprelu       |               | blocks.22.attn.hook_z |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_20  | jumprelu       |               | blocks.22.attn.hook_z |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_361 | jumprelu       |               | blocks.22.attn.hook_z |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/average_l0_37  | jumprelu       |               | blocks.22.attn.hook_z |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_140 | jumprelu       |               | blocks.23.attn.hook_z |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_27  | jumprelu       |               | blocks.23.attn.hook_z |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_276 | jumprelu       |               | blocks.23.attn.hook_z |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_457 | jumprelu       |               | blocks.23.attn.hook_z |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/average_l0_56  | jumprelu       |               | blocks.23.attn.hook_z |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_186 | jumprelu       |               | blocks.24.attn.hook_z |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_32  | jumprelu       |               | blocks.24.attn.hook_z |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_347 | jumprelu       |               | blocks.24.attn.hook_z |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_537 | jumprelu       |               | blocks.24.attn.hook_z |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/average_l0_77  | jumprelu       |               | blocks.24.attn.hook_z |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_153 | jumprelu       |               | blocks.25.attn.hook_z |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_290 | jumprelu       |               | blocks.25.attn.hook_z |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_30  | jumprelu       |               | blocks.25.attn.hook_z |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_465 | jumprelu       |               | blocks.25.attn.hook_z |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/average_l0_63  | jumprelu       |               | blocks.25.attn.hook_z |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-2b-pt-att-canonical](https://huggingface.co/google/gemma-scope-2b-pt-att)

- **Huggingface Repo**: google/gemma-scope-2b-pt-att
- **model**: gemma-2-2b

| id                           | architecture   | neuronpedia   | hook_name             |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:-----------------------------|:---------------|:--------------|:----------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| layer_0/width_16k/canonical  | jumprelu       |               | blocks.0.attn.hook_z  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/canonical  | jumprelu       |               | blocks.1.attn.hook_z  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/canonical  | jumprelu       |               | blocks.2.attn.hook_z  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/canonical  | jumprelu       |               | blocks.3.attn.hook_z  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/canonical  | jumprelu       |               | blocks.4.attn.hook_z  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/canonical  | jumprelu       |               | blocks.5.attn.hook_z  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/canonical  | jumprelu       |               | blocks.6.attn.hook_z  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/canonical  | jumprelu       |               | blocks.7.attn.hook_z  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/canonical  | jumprelu       |               | blocks.8.attn.hook_z  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/canonical  | jumprelu       |               | blocks.9.attn.hook_z  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/canonical | jumprelu       |               | blocks.10.attn.hook_z |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/canonical | jumprelu       |               | blocks.11.attn.hook_z |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/canonical | jumprelu       |               | blocks.12.attn.hook_z |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/canonical | jumprelu       |               | blocks.13.attn.hook_z |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/canonical | jumprelu       |               | blocks.14.attn.hook_z |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/canonical | jumprelu       |               | blocks.15.attn.hook_z |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/canonical | jumprelu       |               | blocks.16.attn.hook_z |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/canonical | jumprelu       |               | blocks.17.attn.hook_z |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/canonical | jumprelu       |               | blocks.18.attn.hook_z |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/canonical | jumprelu       |               | blocks.19.attn.hook_z |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/canonical | jumprelu       |               | blocks.20.attn.hook_z |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/canonical | jumprelu       |               | blocks.21.attn.hook_z |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/canonical | jumprelu       |               | blocks.22.attn.hook_z |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/canonical | jumprelu       |               | blocks.23.attn.hook_z |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/canonical | jumprelu       |               | blocks.24.attn.hook_z |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/canonical | jumprelu       |               | blocks.25.attn.hook_z |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_65k/canonical  | jumprelu       |               | blocks.0.attn.hook_z  |            0 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_65k/canonical  | jumprelu       |               | blocks.1.attn.hook_z  |            1 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_65k/canonical  | jumprelu       |               | blocks.2.attn.hook_z  |            2 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_65k/canonical  | jumprelu       |               | blocks.3.attn.hook_z  |            3 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_65k/canonical  | jumprelu       |               | blocks.4.attn.hook_z  |            4 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_65k/canonical  | jumprelu       |               | blocks.5.attn.hook_z  |            5 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_65k/canonical  | jumprelu       |               | blocks.6.attn.hook_z  |            6 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_65k/canonical  | jumprelu       |               | blocks.7.attn.hook_z  |            7 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_65k/canonical  | jumprelu       |               | blocks.8.attn.hook_z  |            8 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_65k/canonical  | jumprelu       |               | blocks.9.attn.hook_z  |            9 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_65k/canonical | jumprelu       |               | blocks.10.attn.hook_z |           10 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_65k/canonical | jumprelu       |               | blocks.11.attn.hook_z |           11 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_65k/canonical | jumprelu       |               | blocks.12.attn.hook_z |           12 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_65k/canonical | jumprelu       |               | blocks.13.attn.hook_z |           13 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_65k/canonical | jumprelu       |               | blocks.14.attn.hook_z |           14 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_65k/canonical | jumprelu       |               | blocks.15.attn.hook_z |           15 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_65k/canonical | jumprelu       |               | blocks.16.attn.hook_z |           16 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_65k/canonical | jumprelu       |               | blocks.17.attn.hook_z |           17 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_65k/canonical | jumprelu       |               | blocks.18.attn.hook_z |           18 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_65k/canonical | jumprelu       |               | blocks.19.attn.hook_z |           19 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_65k/canonical | jumprelu       |               | blocks.20.attn.hook_z |           20 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_65k/canonical | jumprelu       |               | blocks.21.attn.hook_z |           21 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_65k/canonical | jumprelu       |               | blocks.22.attn.hook_z |           22 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_65k/canonical | jumprelu       |               | blocks.23.attn.hook_z |           23 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_65k/canonical | jumprelu       |               | blocks.24.attn.hook_z |           24 |   65536 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_65k/canonical | jumprelu       |               | blocks.25.attn.hook_z |           25 |   65536 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-9b-pt-att](https://huggingface.co/google/gemma-scope-9b-pt-att)

- **Huggingface Repo**: google/gemma-scope-9b-pt-att
- **model**: gemma-2-2b

| id                                 | architecture   | neuronpedia   | hook_name             |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:-----------------------------------|:---------------|:--------------|:----------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| layer_0/width_131k/average_l0_55   | jumprelu       |               | blocks.0.attn.hook_z  |            0 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_131k/average_l0_116  | jumprelu       |               | blocks.1.attn.hook_z  |            1 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_131k/average_l0_11   | jumprelu       |               | blocks.2.attn.hook_z  |            2 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_131k/average_l0_10   | jumprelu       |               | blocks.3.attn.hook_z  |            3 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_131k/average_l0_12   | jumprelu       |               | blocks.4.attn.hook_z  |            4 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_131k/average_l0_12   | jumprelu       |               | blocks.5.attn.hook_z  |            5 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_131k/average_l0_14   | jumprelu       |               | blocks.6.attn.hook_z  |            6 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_131k/average_l0_148  | jumprelu       |               | blocks.6.attn.hook_z  |            6 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_131k/average_l0_106  | jumprelu       |               | blocks.7.attn.hook_z  |            7 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_131k/average_l0_16   | jumprelu       |               | blocks.8.attn.hook_z  |            8 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_131k/average_l0_111  | jumprelu       |               | blocks.9.attn.hook_z  |            9 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_131k/average_l0_16  | jumprelu       |               | blocks.10.attn.hook_z |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_131k/average_l0_104 | jumprelu       |               | blocks.11.attn.hook_z |           11 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_131k/average_l0_110 | jumprelu       |               | blocks.12.attn.hook_z |           12 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_131k/average_l0_126 | jumprelu       |               | blocks.13.attn.hook_z |           13 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_131k/average_l0_131 | jumprelu       |               | blocks.14.attn.hook_z |           14 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_131k/average_l0_130 | jumprelu       |               | blocks.15.attn.hook_z |           15 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_131k/average_l0_140 | jumprelu       |               | blocks.16.attn.hook_z |           16 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_131k/average_l0_191 | jumprelu       |               | blocks.17.attn.hook_z |           17 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_131k/average_l0_133 | jumprelu       |               | blocks.18.attn.hook_z |           18 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_131k/average_l0_152 | jumprelu       |               | blocks.19.attn.hook_z |           19 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_131k/average_l0_125 | jumprelu       |               | blocks.20.attn.hook_z |           20 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_131k/average_l0_150 | jumprelu       |               | blocks.21.attn.hook_z |           21 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_131k/average_l0_115 | jumprelu       |               | blocks.22.attn.hook_z |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_131k/average_l0_134 | jumprelu       |               | blocks.23.attn.hook_z |           23 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_131k/average_l0_130 | jumprelu       |               | blocks.24.attn.hook_z |           24 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_131k/average_l0_115 | jumprelu       |               | blocks.25.attn.hook_z |           25 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_26/width_131k/average_l0_120 | jumprelu       |               | blocks.26.attn.hook_z |           26 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_27/width_131k/average_l0_102 | jumprelu       |               | blocks.27.attn.hook_z |           27 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_28/width_131k/average_l0_115 | jumprelu       |               | blocks.28.attn.hook_z |           28 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_29/width_131k/average_l0_128 | jumprelu       |               | blocks.29.attn.hook_z |           29 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_30/width_131k/average_l0_109 | jumprelu       |               | blocks.30.attn.hook_z |           30 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_31/width_131k/average_l0_117 | jumprelu       |               | blocks.31.attn.hook_z |           31 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_32/width_131k/average_l0_117 | jumprelu       |               | blocks.32.attn.hook_z |           32 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_33/width_131k/average_l0_128 | jumprelu       |               | blocks.33.attn.hook_z |           33 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_34/width_131k/average_l0_15  | jumprelu       |               | blocks.34.attn.hook_z |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_35/width_131k/average_l0_124 | jumprelu       |               | blocks.35.attn.hook_z |           35 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_36/width_131k/average_l0_105 | jumprelu       |               | blocks.36.attn.hook_z |           36 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_37/width_131k/average_l0_124 | jumprelu       |               | blocks.37.attn.hook_z |           37 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_38/width_131k/average_l0_135 | jumprelu       |               | blocks.38.attn.hook_z |           38 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_39/width_131k/average_l0_120 | jumprelu       |               | blocks.39.attn.hook_z |           39 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_40/width_131k/average_l0_144 | jumprelu       |               | blocks.40.attn.hook_z |           40 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_41/width_131k/average_l0_13  | jumprelu       |               | blocks.41.attn.hook_z |           41 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_0/width_16k/average_l0_12    | jumprelu       |               | blocks.0.attn.hook_z  |            0 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_16k/average_l0_147   | jumprelu       |               | blocks.1.attn.hook_z  |            1 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_16k/average_l0_15    | jumprelu       |               | blocks.2.attn.hook_z  |            2 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_102   | jumprelu       |               | blocks.3.attn.hook_z  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_16k/average_l0_126   | jumprelu       |               | blocks.4.attn.hook_z  |            4 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_16k/average_l0_125   | jumprelu       |               | blocks.5.attn.hook_z  |            5 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_16k/average_l0_108   | jumprelu       |               | blocks.6.attn.hook_z  |            6 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_16k/average_l0_70    | jumprelu       |               | blocks.7.attn.hook_z  |            7 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_16k/average_l0_150   | jumprelu       |               | blocks.8.attn.hook_z  |            8 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_16k/average_l0_172   | jumprelu       |               | blocks.9.attn.hook_z  |            9 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_132  | jumprelu       |               | blocks.10.attn.hook_z |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_16k/average_l0_153  | jumprelu       |               | blocks.11.attn.hook_z |           11 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_16k/average_l0_149  | jumprelu       |               | blocks.12.attn.hook_z |           12 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_16k/average_l0_170  | jumprelu       |               | blocks.13.attn.hook_z |           13 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_16k/average_l0_179  | jumprelu       |               | blocks.14.attn.hook_z |           14 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_16k/average_l0_168  | jumprelu       |               | blocks.15.attn.hook_z |           15 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_16k/average_l0_172  | jumprelu       |               | blocks.16.attn.hook_z |           16 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_16k/average_l0_110  | jumprelu       |               | blocks.17.attn.hook_z |           17 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_16k/average_l0_171  | jumprelu       |               | blocks.18.attn.hook_z |           18 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_16k/average_l0_186  | jumprelu       |               | blocks.19.attn.hook_z |           19 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_158  | jumprelu       |               | blocks.20.attn.hook_z |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_16k/average_l0_195  | jumprelu       |               | blocks.21.attn.hook_z |           21 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_16k/average_l0_141  | jumprelu       |               | blocks.22.attn.hook_z |           22 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_16k/average_l0_173  | jumprelu       |               | blocks.23.attn.hook_z |           23 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_16k/average_l0_167  | jumprelu       |               | blocks.24.attn.hook_z |           24 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_16k/average_l0_156  | jumprelu       |               | blocks.25.attn.hook_z |           25 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_26/width_16k/average_l0_159  | jumprelu       |               | blocks.26.attn.hook_z |           26 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_27/width_16k/average_l0_136  | jumprelu       |               | blocks.27.attn.hook_z |           27 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_28/width_16k/average_l0_143  | jumprelu       |               | blocks.28.attn.hook_z |           28 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_29/width_16k/average_l0_171  | jumprelu       |               | blocks.29.attn.hook_z |           29 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_30/width_16k/average_l0_157  | jumprelu       |               | blocks.30.attn.hook_z |           30 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_31/width_16k/average_l0_168  | jumprelu       |               | blocks.31.attn.hook_z |           31 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_32/width_16k/average_l0_158  | jumprelu       |               | blocks.32.attn.hook_z |           32 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_33/width_16k/average_l0_158  | jumprelu       |               | blocks.33.attn.hook_z |           33 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_34/width_16k/average_l0_17   | jumprelu       |               | blocks.34.attn.hook_z |           34 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_35/width_16k/average_l0_14   | jumprelu       |               | blocks.35.attn.hook_z |           35 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_36/width_16k/average_l0_144  | jumprelu       |               | blocks.36.attn.hook_z |           36 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_37/width_16k/average_l0_17   | jumprelu       |               | blocks.37.attn.hook_z |           37 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_37/width_16k/average_l0_172  | jumprelu       |               | blocks.37.attn.hook_z |           37 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_38/width_16k/average_l0_175  | jumprelu       |               | blocks.38.attn.hook_z |           38 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_39/width_16k/average_l0_15   | jumprelu       |               | blocks.39.attn.hook_z |           39 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_40/width_16k/average_l0_18   | jumprelu       |               | blocks.40.attn.hook_z |           40 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_40/width_16k/average_l0_189  | jumprelu       |               | blocks.40.attn.hook_z |           40 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_41/width_16k/average_l0_129  | jumprelu       |               | blocks.41.attn.hook_z |           41 |   16384 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-9b-pt-mlp](https://huggingface.co/google/gemma-scope-9b-pt-mlp)

- **Huggingface Repo**: google/gemma-scope-9b-pt-mlp
- **model**: gemma-2-2b

| id                                 | architecture   | neuronpedia   | hook_name              |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:-----------------------------------|:---------------|:--------------|:-----------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| layer_0/width_131k/average_l0_11   | jumprelu       |               | blocks.0.hook_mlp_out  |            0 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_131k/average_l0_10   | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_1/width_131k/average_l0_106  | jumprelu       |               | blocks.1.hook_mlp_out  |            1 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_2/width_131k/average_l0_12   | jumprelu       |               | blocks.2.hook_mlp_out  |            2 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_131k/average_l0_109  | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_4/width_131k/average_l0_14   | jumprelu       |               | blocks.4.hook_mlp_out  |            4 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_5/width_131k/average_l0_12   | jumprelu       |               | blocks.5.hook_mlp_out  |            5 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_6/width_131k/average_l0_12   | jumprelu       |               | blocks.6.hook_mlp_out  |            6 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_7/width_131k/average_l0_13   | jumprelu       |               | blocks.7.hook_mlp_out  |            7 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_8/width_131k/average_l0_15   | jumprelu       |               | blocks.8.hook_mlp_out  |            8 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_131k/average_l0_12   | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_9/width_131k/average_l0_129  | jumprelu       |               | blocks.9.hook_mlp_out  |            9 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_131k/average_l0_12  | jumprelu       |               | blocks.10.hook_mlp_out |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_11/width_131k/average_l0_120 | jumprelu       |               | blocks.11.hook_mlp_out |           11 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_12/width_131k/average_l0_159 | jumprelu       |               | blocks.12.hook_mlp_out |           12 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_13/width_131k/average_l0_160 | jumprelu       |               | blocks.13.hook_mlp_out |           13 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_14/width_131k/average_l0_174 | jumprelu       |               | blocks.14.hook_mlp_out |           14 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_15/width_131k/average_l0_194 | jumprelu       |               | blocks.15.hook_mlp_out |           15 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_131k/average_l0_17  | jumprelu       |               | blocks.16.hook_mlp_out |           16 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_16/width_131k/average_l0_175 | jumprelu       |               | blocks.16.hook_mlp_out |           16 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_17/width_131k/average_l0_207 | jumprelu       |               | blocks.17.hook_mlp_out |           17 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_18/width_131k/average_l0_174 | jumprelu       |               | blocks.18.hook_mlp_out |           18 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_19/width_131k/average_l0_189 | jumprelu       |               | blocks.19.hook_mlp_out |           19 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_131k/average_l0_20  | jumprelu       |               | blocks.20.hook_mlp_out |           20 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_21/width_131k/average_l0_16  | jumprelu       |               | blocks.21.hook_mlp_out |           21 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_131k/average_l0_17  | jumprelu       |               | blocks.22.hook_mlp_out |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_131k/average_l0_172 | jumprelu       |               | blocks.22.hook_mlp_out |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_23/width_131k/average_l0_146 | jumprelu       |               | blocks.23.hook_mlp_out |           23 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_24/width_131k/average_l0_147 | jumprelu       |               | blocks.24.hook_mlp_out |           24 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_25/width_131k/average_l0_139 | jumprelu       |               | blocks.25.hook_mlp_out |           25 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_26/width_131k/average_l0_110 | jumprelu       |               | blocks.26.hook_mlp_out |           26 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_27/width_131k/average_l0_14  | jumprelu       |               | blocks.27.hook_mlp_out |           27 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_28/width_131k/average_l0_15  | jumprelu       |               | blocks.28.hook_mlp_out |           28 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_29/width_131k/average_l0_15  | jumprelu       |               | blocks.29.hook_mlp_out |           29 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_30/width_131k/average_l0_14  | jumprelu       |               | blocks.30.hook_mlp_out |           30 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_31/width_131k/average_l0_12  | jumprelu       |               | blocks.31.hook_mlp_out |           31 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_32/width_131k/average_l0_12  | jumprelu       |               | blocks.32.hook_mlp_out |           32 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_33/width_131k/average_l0_12  | jumprelu       |               | blocks.33.hook_mlp_out |           33 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_34/width_131k/average_l0_10  | jumprelu       |               | blocks.34.hook_mlp_out |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_35/width_131k/average_l0_10  | jumprelu       |               | blocks.35.hook_mlp_out |           35 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_36/width_131k/average_l0_11  | jumprelu       |               | blocks.36.hook_mlp_out |           36 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_37/width_131k/average_l0_12  | jumprelu       |               | blocks.37.hook_mlp_out |           37 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_38/width_131k/average_l0_11  | jumprelu       |               | blocks.38.hook_mlp_out |           38 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_39/width_131k/average_l0_11  | jumprelu       |               | blocks.39.hook_mlp_out |           39 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_40/width_131k/average_l0_11  | jumprelu       |               | blocks.40.hook_mlp_out |           40 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_41/width_131k/average_l0_14  | jumprelu       |               | blocks.41.hook_mlp_out |           41 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_3/width_16k/average_l0_126   | jumprelu       |               | blocks.3.hook_mlp_out  |            3 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_16k/average_l0_114  | jumprelu       |               | blocks.10.hook_mlp_out |           10 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_146  | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_1522 | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_23   | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_384  | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_56   | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_20/width_16k/average_l0_868  | jumprelu       |               | blocks.20.hook_mlp_out |           20 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_26/width_16k/average_l0_14   | jumprelu       |               | blocks.26.hook_mlp_out |           26 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_26/width_16k/average_l0_142  | jumprelu       |               | blocks.26.hook_mlp_out |           26 |   16384 |           1024 | monology/pile-uncopyrighted |                         |
| layer_31/width_16k/average_l0_12   | jumprelu       |               | blocks.31.hook_mlp_out |           31 |   16384 |           1024 | monology/pile-uncopyrighted |                         |

## [gemma-scope-27b-pt-res](https://huggingface.co/google/gemma-scope-27b-pt-res)

- **Huggingface Repo**: google/gemma-scope-27b-pt-res
- **model**: gemma-2-2b

| id                                 | architecture   | neuronpedia   | hook_name                 |   hook_layer |   d_sae |   context_size | dataset_path                | normalize_activations   |
|:-----------------------------------|:---------------|:--------------|:--------------------------|-------------:|--------:|---------------:|:----------------------------|:------------------------|
| layer_10/width_131k/average_l0_106 | jumprelu       |               | blocks.10.hook_resid_post |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_131k/average_l0_15  | jumprelu       |               | blocks.10.hook_resid_post |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_131k/average_l0_200 | jumprelu       |               | blocks.10.hook_resid_post |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_131k/average_l0_24  | jumprelu       |               | blocks.10.hook_resid_post |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_131k/average_l0_37  | jumprelu       |               | blocks.10.hook_resid_post |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_10/width_131k/average_l0_64  | jumprelu       |               | blocks.10.hook_resid_post |           10 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_131k/average_l0_150 | jumprelu       |               | blocks.22.hook_resid_post |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_131k/average_l0_20  | jumprelu       |               | blocks.22.hook_resid_post |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_131k/average_l0_290 | jumprelu       |               | blocks.22.hook_resid_post |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_131k/average_l0_31  | jumprelu       |               | blocks.22.hook_resid_post |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_131k/average_l0_48  | jumprelu       |               | blocks.22.hook_resid_post |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_22/width_131k/average_l0_82  | jumprelu       |               | blocks.22.hook_resid_post |           22 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_34/width_131k/average_l0_155 | jumprelu       |               | blocks.34.hook_resid_post |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_34/width_131k/average_l0_21  | jumprelu       |               | blocks.34.hook_resid_post |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_34/width_131k/average_l0_333 | jumprelu       |               | blocks.34.hook_resid_post |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_34/width_131k/average_l0_38  | jumprelu       |               | blocks.34.hook_resid_post |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_34/width_131k/average_l0_72  | jumprelu       |               | blocks.34.hook_resid_post |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |
| layer_34/width_131k/average_l0_785 | jumprelu       |               | blocks.34.hook_resid_post |           34 |  131072 |           1024 | monology/pile-uncopyrighted |                         |

## [pythia-70m-deduped-res-sm](https://huggingface.co/ctigges/pythia-70m-deduped__res-sm_processed)

- **Huggingface Repo**: ctigges/pythia-70m-deduped__res-sm_processed
- **model**: pythia-70m-deduped
- **Additional Links**:
    - [Model](https://huggingface.co/EleutherAI/pythia-70m-deduped)
    - [Dashboards](https://www.neuronpedia.org/pythia-70m-deduped)

| id                       | architecture   | neuronpedia                 | hook_name                |   hook_layer |   d_sae |   context_size | dataset_path                     | normalize_activations   |
|:-------------------------|:---------------|:----------------------------|:-------------------------|-------------:|--------:|---------------:|:---------------------------------|:------------------------|
| blocks.0.hook_resid_pre  | standard       | pythia-70m-deduped/e-att-sm | blocks.0.hook_resid_pre  |            0 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.0.hook_resid_post | standard       | pythia-70m-deduped/0-res-sm | blocks.0.hook_resid_post |            0 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.1.hook_resid_post | standard       | pythia-70m-deduped/1-res-sm | blocks.1.hook_resid_post |            1 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.2.hook_resid_post | standard       | pythia-70m-deduped/2-res-sm | blocks.2.hook_resid_post |            2 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.3.hook_resid_post | standard       | pythia-70m-deduped/3-res-sm | blocks.3.hook_resid_post |            3 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.4.hook_resid_post | standard       | pythia-70m-deduped/4-res-sm | blocks.4.hook_resid_post |            4 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.5.hook_resid_post | standard       | pythia-70m-deduped/5-res-sm | blocks.5.hook_resid_post |            5 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |

## [pythia-70m-deduped-mlp-sm](https://huggingface.co/ctigges/pythia-70m-deduped__mlp-sm_processed)

- **Huggingface Repo**: ctigges/pythia-70m-deduped__mlp-sm_processed
- **model**: pythia-70m-deduped
- **Additional Links**:
    - [Model](https://huggingface.co/EleutherAI/pythia-70m-deduped)
    - [Dashboards](https://www.neuronpedia.org/pythia-70m-deduped)

| id                    | architecture   | neuronpedia                 | hook_name             |   hook_layer |   d_sae |   context_size | dataset_path                     | normalize_activations   |
|:----------------------|:---------------|:----------------------------|:----------------------|-------------:|--------:|---------------:|:---------------------------------|:------------------------|
| blocks.0.hook_mlp_out | standard       | pythia-70m-deduped/0-mlp-sm | blocks.0.hook_mlp_out |            0 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.1.hook_mlp_out | standard       | pythia-70m-deduped/1-mlp-sm | blocks.1.hook_mlp_out |            1 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.2.hook_mlp_out | standard       | pythia-70m-deduped/2-mlp-sm | blocks.2.hook_mlp_out |            2 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.3.hook_mlp_out | standard       | pythia-70m-deduped/3-mlp-sm | blocks.3.hook_mlp_out |            3 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.4.hook_mlp_out | standard       | pythia-70m-deduped/4-mlp-sm | blocks.4.hook_mlp_out |            4 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.5.hook_mlp_out | standard       | pythia-70m-deduped/5-mlp-sm | blocks.5.hook_mlp_out |            5 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |

## [pythia-70m-deduped-att-sm](https://huggingface.co/ctigges/pythia-70m-deduped__att-sm_processed)

- **Huggingface Repo**: ctigges/pythia-70m-deduped__att-sm_processed
- **model**: pythia-70m-deduped
- **Additional Links**:
    - [Model](https://huggingface.co/EleutherAI/pythia-70m-deduped)
    - [Dashboards](https://www.neuronpedia.org/pythia-70m-deduped)

| id                     | architecture   | neuronpedia                 | hook_name              |   hook_layer |   d_sae |   context_size | dataset_path                     | normalize_activations   |
|:-----------------------|:---------------|:----------------------------|:-----------------------|-------------:|--------:|---------------:|:---------------------------------|:------------------------|
| blocks.0.hook_attn_out | standard       | pythia-70m-deduped/0-att-sm | blocks.0.hook_attn_out |            0 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.1.hook_attn_out | standard       | pythia-70m-deduped/1-att-sm | blocks.1.hook_attn_out |            1 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.2.hook_attn_out | standard       | pythia-70m-deduped/2-att-sm | blocks.2.hook_attn_out |            2 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.3.hook_attn_out | standard       | pythia-70m-deduped/3-att-sm | blocks.3.hook_attn_out |            3 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.4.hook_attn_out | standard       | pythia-70m-deduped/4-att-sm | blocks.4.hook_attn_out |            4 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |
| blocks.5.hook_attn_out | standard       | pythia-70m-deduped/5-att-sm | blocks.5.hook_attn_out |            5 |   32768 |            128 | EleutherAI/the_pile_deduplicated | none                    |

## [gpt2-small-res_sll-ajt](https://huggingface.co/neuronpedia/gpt2-small__res_sll-ajt)

- **Huggingface Repo**: neuronpedia/gpt2-small__res_sll-ajt
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)
    - [Dashboards](https://www.neuronpedia.org/gpt2-small/res_sll-ajt)

| id                       | architecture   | neuronpedia               | hook_name                |   hook_layer |   d_sae |   context_size | dataset_path                                          | normalize_activations   |
|:-------------------------|:---------------|:--------------------------|:-------------------------|-------------:|--------:|---------------:|:------------------------------------------------------|:------------------------|
| blocks.2.hook_resid_pre  | standard       | gpt2-small/2-res_sll-ajt  | blocks.2.hook_resid_pre  |            2 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |
| blocks.6.hook_resid_pre  | standard       | gpt2-small/6-res_sll-ajt  | blocks.6.hook_resid_pre  |            6 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |
| blocks.10.hook_resid_pre | standard       | gpt2-small/10-res_sll-ajt | blocks.10.hook_resid_pre |           10 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |

## [gpt2-small-res_slefr-ajt](https://huggingface.co/neuronpedia/gpt2-small__res_slefr-ajt)

- **Huggingface Repo**: neuronpedia/gpt2-small__res_slefr-ajt
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)
    - [Dashboards](https://www.neuronpedia.org/gpt2-small/res_slefr-ajt)

| id                       | architecture   | neuronpedia                 | hook_name                |   hook_layer |   d_sae |   context_size | dataset_path                                          | normalize_activations   |
|:-------------------------|:---------------|:----------------------------|:-------------------------|-------------:|--------:|---------------:|:------------------------------------------------------|:------------------------|
| blocks.2.hook_resid_pre  | standard       | gpt2-small/2-res_slefr-ajt  | blocks.2.hook_resid_pre  |            2 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |
| blocks.6.hook_resid_pre  | standard       | gpt2-small/6-res_slefr-ajt  | blocks.6.hook_resid_pre  |            6 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |
| blocks.10.hook_resid_pre | standard       | gpt2-small/10-res_slefr-ajt | blocks.10.hook_resid_pre |           10 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |

## [gpt2-small-res_scl-ajt](https://huggingface.co/neuronpedia/gpt2-small__res_scl-ajt)

- **Huggingface Repo**: neuronpedia/gpt2-small__res_scl-ajt
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)
    - [Dashboards](https://www.neuronpedia.org/gpt2-small/res_scl-ajt)

| id                       | architecture   | neuronpedia               | hook_name                |   hook_layer |   d_sae |   context_size | dataset_path                                          | normalize_activations   |
|:-------------------------|:---------------|:--------------------------|:-------------------------|-------------:|--------:|---------------:|:------------------------------------------------------|:------------------------|
| blocks.2.hook_resid_pre  | standard       | gpt2-small/2-res_scl-ajt  | blocks.2.hook_resid_pre  |            2 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |
| blocks.6.hook_resid_pre  | standard       | gpt2-small/6-res_scl-ajt  | blocks.6.hook_resid_pre  |            6 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |
| blocks.10.hook_resid_pre | standard       | gpt2-small/10-res_scl-ajt | blocks.10.hook_resid_pre |           10 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |

## [gpt2-small-res_sle-ajt](https://huggingface.co/neuronpedia/gpt2-small__res_sle-ajt)

- **Huggingface Repo**: neuronpedia/gpt2-small__res_sle-ajt
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)
    - [Dashboards](https://www.neuronpedia.org/gpt2-small/res_sle-ajt)

| id                       | architecture   | neuronpedia               | hook_name                |   hook_layer |   d_sae |   context_size | dataset_path                                          | normalize_activations   |
|:-------------------------|:---------------|:--------------------------|:-------------------------|-------------:|--------:|---------------:|:------------------------------------------------------|:------------------------|
| blocks.2.hook_resid_pre  | standard       | gpt2-small/2-res_sle-ajt  | blocks.2.hook_resid_pre  |            2 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |
| blocks.6.hook_resid_pre  | standard       | gpt2-small/6-res_sle-ajt  | blocks.6.hook_resid_pre  |            6 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |
| blocks.10.hook_resid_pre | standard       | gpt2-small/10-res_sle-ajt | blocks.10.hook_resid_pre |           10 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |

## [gpt2-small-res_sce-ajt](https://huggingface.co/neuronpedia/gpt2-small__res_sce-ajt)

- **Huggingface Repo**: neuronpedia/gpt2-small__res_sce-ajt
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)
    - [Dashboards](https://www.neuronpedia.org/gpt2-small/res_sce-ajt)

| id                       | architecture   | neuronpedia               | hook_name                |   hook_layer |   d_sae |   context_size | dataset_path                                          | normalize_activations   |
|:-------------------------|:---------------|:--------------------------|:-------------------------|-------------:|--------:|---------------:|:------------------------------------------------------|:------------------------|
| blocks.2.hook_resid_pre  | standard       | gpt2-small/2-res_sce-ajt  | blocks.2.hook_resid_pre  |            2 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |
| blocks.6.hook_resid_pre  | standard       | gpt2-small/6-res_sce-ajt  | blocks.6.hook_resid_pre  |            6 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |
| blocks.10.hook_resid_pre | standard       | gpt2-small/10-res_sce-ajt | blocks.10.hook_resid_pre |           10 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |

## [gpt2-small-res_scefr-ajt](https://huggingface.co/neuronpedia/gpt2-small__res_scefr-ajt)

- **Huggingface Repo**: neuronpedia/gpt2-small__res_scefr-ajt
- **model**: gpt2-small
- **Additional Links**:
    - [Model](https://huggingface.co/gpt2)
    - [Dashboards](https://www.neuronpedia.org/gpt2-small/res_scefr-ajt)

| id                       | architecture   | neuronpedia                 | hook_name                |   hook_layer |   d_sae |   context_size | dataset_path                                          | normalize_activations   |
|:-------------------------|:---------------|:----------------------------|:-------------------------|-------------:|--------:|---------------:|:------------------------------------------------------|:------------------------|
| blocks.2.hook_resid_pre  | standard       | gpt2-small/2-res_scefr-ajt  | blocks.2.hook_resid_pre  |            2 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |
| blocks.6.hook_resid_pre  | standard       | gpt2-small/6-res_scefr-ajt  | blocks.6.hook_resid_pre  |            6 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |
| blocks.10.hook_resid_pre | standard       | gpt2-small/10-res_scefr-ajt | blocks.10.hook_resid_pre |           10 |   46080 |            128 | apollo-research/Skylion007-openwebtext-tokenizer-gpt2 | none                    |

