<img width="1308" alt="Screenshot 2024-03-21 at 3 08 28 pm" src="https://github.com/jbloomAus/mats_sae_training/assets/69127271/209012ec-a779-4036-b4be-7b7739ea87f6">

# SAELens
[![PyPI](https://img.shields.io/pypi/v/sae-lens?color=blue)](https://pypi.org/project/sae-lens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![build](https://github.com/jbloomAus/SAELens/actions/workflows/build.yml/badge.svg)](https://github.com/jbloomAus/SAELens/actions/workflows/build.yml)
[![Deploy Docs](https://github.com/jbloomAus/SAELens/actions/workflows/deploy_docs.yml/badge.svg)](https://github.com/jbloomAus/SAELens/actions/workflows/deploy_docs.yml)
[![codecov](https://codecov.io/gh/jbloomAus/SAELens/graph/badge.svg?token=N83NGH8CGE)](https://codecov.io/gh/jbloomAus/SAELens)

The SAELens training codebase exists to help researchers:

- Train sparse autoencoders.
- Analyse sparse autoencoders and neural network internals.
- Generate insights which make it easier to create safe and aligned AI systems.

**Please note these docs are in beta. We intend to make them cleaner and more comprehensive over time.**

## Quick Start

### Installation

```
pip install sae-lens
```

### Loading Sparse Autoencoders from Huggingface


#### Loading officially supported SAEs

To load an officially supported sparse autoencoder, you can use `SparseAutoencoder.from_pretrained()` as below:

```python
from sae_lens import SparseAutoencoder

layer = 8 # pick a layer you want.
sparse_autoencoder = SparseAutoencoder.from_pretrained(
    "gpt2-small-res-jb", f"blocks.{layer}.hook_resid_pre"
)
```

You can see other importable SAEs in   `sae_lens/pretrained_saes.yaml`.

(We'd accept a PR that converts this yaml to a nice table in the docs!)

#### Loading SAEs, ActivationsStore and Models from HuggingFace.

For more advanced use-cases like fine-tuning a pre-trained SAE, [previously trained sparse autoencoders](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted) can be loaded from huggingface with close to single line of code. For more details and performance metrics for these sparse autoencoder, read my [blog post](https://www.alignmentforum.org/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream). 

```python

# picking up from the avoce chunk.
from sae_lens import LMSparseAutoencoderSessionloader

model, _, activation_store = LMSparseAutoencoderSessionloader(sparse_autoencoder.cfg).load_sae_training_group_session(
    path = path
)
sparse_autoencoder.eval()
```

### Background

We highly recommend this [tutorial](https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab).

## Tutorials

I wrote a tutorial to show users how to do some basic exploration of their SAE:

- [Loading and Analysing Pre-Trained Sparse Autoencoders](tutorials/basic_loading_and_analysing.ipynb)
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/jbloomAus/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb)
 - [Understanding SAE Features with the Logit Lens](tutorials/logits_lens_with_features.ipynb)
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/jbloomAus/SAELens/blob/main/tutorials/logits_lens_with_features.ipynb)
  - [Training a Sparse Autoencoder](tutorials/training_a_sparse_autoencoder.ipynb)
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/jbloomAus/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb)


## Example WandB Training Dashboard

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
