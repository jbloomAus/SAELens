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

To load a pretrained sparse autoencoder, you can use `SAE.from_pretrained()` as below. Note that we return the *original cfg dict* from the huggingface repo so that it's easy to debug older configs that are being handled when we import an SAe. We also return a sparsity tensor if it is present in the repo. For an example repo structure, see [here](https://huggingface.co/jbloom/Gemma-2b-Residual-Stream-SAEs). 

```python
from sae_lens import SAE

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "gpt2-small-res-jb", # see other options in sae_lens/pretrained_saes.yaml
    sae_id = "blocks.8.hook_resid_pre", # won't always be a hook point
    device = device
)
```

You can see other importable SAEs on [this page](https://jbloomaus.github.io/SAELens/sae_table/).

### Background and further Readings

We highly recommend this [tutorial](https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab).

For recent progress in SAEs, we recommend the LessWrong forum's [Sparse Autoencoder tag](https://www.lesswrong.com/tag/sparse-autoencoders-saes)

## Tutorials

I wrote a tutorial to show users how to do some basic exploration of their SAE:

- Loading and Analysing Pre-Trained Sparse Autoencoders [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/jbloomAus/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb)
 - Understanding SAE Features with the Logit Lens [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/jbloomAus/SAELens/blob/main/tutorials/logits_lens_with_features.ipynb)
  - Training a Sparse Autoencoder [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/jbloomAus/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb)


## Example WandB Dashboard

WandB Dashboards provide lots of useful insights while training SAE's. Here's a screenshot from one training run. 

![screenshot](dashboard_screenshot.png)

## Citation

```
@misc{bloom2024saetrainingcodebase,
   title = {SAELens Training
   author = {Joseph Bloom, David Chanin},
   year = {2024},
   howpublished = {\url{https://github.com/jbloomAus/SAELens}},
}}
```
