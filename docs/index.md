<img width="1308" alt="Screenshot 2024-03-21 at 3 08 28 pm" src="https://github.com/jbloomAus/mats_sae_training/assets/69127271/209012ec-a779-4036-b4be-7b7739ea87f6">

# MATS SAE Training
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![build](https://github.com/jbloomAus/mats_sae_training/actions/workflows/tests.yml/badge.svg)](https://github.com/jbloomAus/mats_sae_training/actions/workflows/tests.yml)
[![Deploy Docs](https://github.com/jbloomAus/mats_sae_training/actions/workflows/deploy_docs.yml/badge.svg)](https://github.com/jbloomAus/mats_sae_training/actions/workflows/deploy_docs.yml)
[![codecov](https://codecov.io/gh/jbloomAus/mats_sae_training/graph/badge.svg?token=N83NGH8CGE)](https://codecov.io/gh/jbloomAus/mats_sae_training)

The MATS SAE training codebase (we'll rename it soon) exists to help researchers:

- Train sparse autoencoders.
- Analyse sparse autoencoders and neural network internals.
- Generate insights which make it easier to create safe and aligned AI systems.

**Please note these docs are in beta. We intend to make them cleaner and more comprehensive over time.**

## Quick Start

### Set Up

This project uses [Poetry](https://python-poetry.org/) for dependency management. Ensure Poetry is installed, then to install the dependencies, run:

```
poetry install
```

### Loading Sparse Autoencoders from Huggingface

[Previously trained sparse autoencoders](https://huggingface.co/jbloom/GPT2-Small-SAEs) can be loaded from huggingface with close to single line of code. For more details and performance metrics for these sparse autoencoder, read my [blog post](https://www.alignmentforum.org/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream). 

```python
import torch 
from sae_training.utils import LMSparseAutoencoderSessionloader
from huggingface_hub import hf_hub_download

layer = 8 # pick a layer you want.
REPO_ID = "jbloom/GPT2-Small-SAEs"
FILENAME = f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"
path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
model, sparse_autoencoder, activation_store = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
    path = path
)
sparse_autoencoder.eval()
```

You can also load the feature sparsity from huggingface. 

```python
FILENAME = f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576_log_feature_sparsity.pt"
path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
log_feature_sparsity = torch.load(path, map_location=sparse_autoencoder.cfg.device)

```
### Background

We highly recommend this [tutorial](https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab).



## Code Overview

The codebase contains 2 folders worth caring about:

- sae_training: The main body of the code is here. Everything required for training SAEs. 
- sae_analysis: This code is mainly house the feature visualizer code we use to generate dashboards. It was written by Callum McDougal but I've ported it here with permission and edited it to work with a few different activation types. 

Some other folders:

- tutorials: These aren't well maintained but I'll aim to clean them up soon. 
- tests: When first developing the codebase, I was writing more tests. I have no idea whether they are currently working!


## Loading a Pretrained Language Model 

Once your SAE is trained, the final SAE weights will be saved to wandb and are loadable via the session loader. The session loader will return:
- The model your SAE was trained on (presumably you're interested in studying this. It's always a HookedTransformer)
- Your SAE.
- An activations loader: from which you can get randomly sampled activations or batches of tokens from the dataset you used to train the SAE. (more on this in the tutorial)

```python
from sae_training.utils import LMSparseAutoencoderSessionloader

path ="path/to/sparse_autoencoder.pt"
model, sparse_autoencoder, activations_loader = LMSparseAutoencoderSessionloader.load_session_from_pretrained(
    path
)

```
## Tutorials

I wrote a tutorial to show users how to do some basic exploration of their SAE:

- `evaluating_your_sae.ipynb`: A quick/dirty notebook showing how to check L0 and Prediction loss with your SAE, as well as showing how to generate interactive dashboards using Callum's reporduction of [Anthropics interface](https://transformer-circuits.pub/2023/monosemantic-features#setup-interface).
- `logits_lens_with_features.ipynb`: A notebook showing how to reproduce the analysis from this [LessWrong post](https://www.lesswrong.com/posts/qykrYY6rXXM7EEs8Q/understanding-sae-features-with-the-logit-lens).

## Example Dashboard

WandB Dashboards provide lots of useful insights while training SAE's. Here's a screenshot from one training run. 

![screenshot](dashboard_screenshot.png)




## Citations and References:

Research:
- [Towards Monosemanticy](https://transformer-circuits.pub/2023/monosemantic-features)
- [Sparse Autoencoders Find Highly Interpretable Features in Language Model](https://arxiv.org/abs/2309.08600)



Reference Implementations:
- [Neel Nanda](https://github.com/neelnanda-io/1L-Sparse-Autoencoder)
- [AI-Safety-Foundation](https://github.com/ai-safety-foundation/sparse_autoencoder).
- [Arthur Conmy](https://github.com/ArthurConmy/sae).
- [Callum McDougall](https://github.com/callummcdougall/sae-exercises-mats/tree/main)
