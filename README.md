# MATS SAE Training

This codebase contains training scripts and analysis code for Sparse AutoEncoders. I wasn't planning to share this codebase initially but I've recieved feedback that others have found it useful so I'm going to slowly transition it to be a more serious repo (formating/linting/testing etc.). In the mean time, please feel free to add Pull Requests or make issues if you have any trouble with it. 


## Set Up

This project uses [Poetry](https://python-poetry.org/) for dependency management. Ensure Poetry is installed, then to install the dependencies, run:

```
poetry install
```

## Background

We highly recommend this [tutorial](https://www.lesswrong.com/posts/LnHowHgmrMbWtpkxx/intro-to-superposition-and-sparse-autoencoders-colab).

## Code Overview

The codebase contains 2 folders worth caring about:

- sae_training: The main body of the code is here. Everything required for training SAEs. 
- sae_analysis: This code is mainly house the feature visualizer code we use to generate dashboards. It was written by Callum McDougal but I've ported it here with permission and edited it to work with a few different activation types. 

Some other folders:

- tutorials: These aren't well maintained but I'll aim to clean them up soon. 
- tests: When first developing the codebase, I was writing more tests. I have no idea whether they are currently working!

I've been commiting my research code to the `Research` folder but am not expecting other people use or look at that. 

## Loading Sparse Autoencoders from Huggingface

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


## Training a Sparse Autoencoder on a Language Model

Sparse Autoencoders can be intimidating at first but it's fairly simple to train one once you know what each part of the config does. I've created a config class which you instantiate and pass to the runner which will complete your training run and log it's progress to wandb. 

Let's go through the major components of the config:
- Data: SAE's autoencode model activations. We need to specify the model, the part of the models activations we want to autoencode and the dataset the model is operating on when generating those activations. We now automatically detect if that dataset is tokenized and most huggingface datasets should be fine. One slightly annoying detail is that you need to know the dimensionality of those activations when contructing your SAE but you can get that in the transformerlens [docs](https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html). Any language model in the table from those docs should work. 
- SAE Parameters: Your expansion factor will determine the size of your SAE and the decoder bias initialization method should always be geometric_median or mean. Mean is faster but theoretically sub-optimal. I use another package to get the geometric median and it can be quite slow. 
- Training Parameters: These are most critical. The right L1 coefficient (coefficient in the activation sparsity inducing term in the loss) changes with your learning rate but a good bet would be to use LR 4e-4 and L1 8e-5 for GPT2 small. These will vary for other models and playing around with them / short runs can be helpful. Training batch size of 4096 is standard and I'm not really sure whether there's benefit to playing with it. In theory a larger context size (one accurate to whatever the model was trained with) seems good but it's computationally cheaper to use 128. Learning rate warm up is important to avoid dead neurons. 
- Activation Store Parameters: The activation store shuffles activations from forward passes over samples from your data. The larger it is, the better shuffling you'll get. In theory more shuffling is good. The total training tokens is a very important parameter. The more the better, but you'll often see good results having trained on a few hundred million tokens. Store batch batch size is a function of your gpu and how many forward passes of your model you want to do simultaneously when collecting activations.
- Dead Neurons / Sparsity Metrics: The config around resampling was more important when we were using resampling to avoid dead neurons (see Anthropic's post on this), but using ghost gradients, the resampling protcol is much simpler. I'd always set ghost grad to True and feature sampling method to None. The feature sampling window effects the dashboard statistics tracking feature occurence and the dead feature window tracks how many forward passes a neuron must not activate before we apply ghost grads to it. 
- WANDB: Fairly straightfoward. Don't set log frequency too high or your dashboard will be slow!
- Device: I can run this code on my macbook with "mps" but mostly do runs with cuda. 
- Dtype: Float16 maybe could work but I had some funky results and have left it at float32 for the time being. 
- Checkpoints: I'd collected checkpoints on runs you care about but turn them off when tuning since it can be slow. 


```python
import torch
import os 
import sys 

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB__SERVICE_WAIT"] = "300"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.lm_runner import language_model_sae_runner

cfg = LanguageModelSAERunnerConfig(

    # Data Generating Function (Model + Training Distibuion)
    model_name = "gpt2-small",
    hook_point = "blocks.2.hook_resid_pre",
    hook_point_layer = 2,
    d_in = 768,
    dataset_path = "Skylion007/openwebtext",
    is_dataset_tokenized=False,
    
    # SAE Parameters
    expansion_factor = 64,
    b_dec_init_method = "geometric_median",
    
    # Training Parameters
    lr = 0.0004,
    l1_coefficient = 0.00008,
    lr_scheduler_name="constantwithwarmup",
    train_batch_size = 4096,
    context_size = 128,
    lr_warm_up_steps=5000,
    
    # Activation Store Parameters
    n_batches_in_buffer = 128,
    total_training_tokens = 1_000_000 * 300,
    store_batch_size = 32,
    
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_window = 1000,
    dead_feature_window=5000,
    dead_feature_threshold = 1e-6,
    
    # WANDB
    log_to_wandb = True,
    wandb_project= "mats_sae_training_gpt2",
    wandb_entity = None,
    wandb_log_frequency=100,
    
    # Misc
    device = "cuda",
    seed = 42,
    n_checkpoints = 10,
    checkpoint_path = "checkpoints",
    dtype = torch.float32,
    )

sparse_autoencoder = language_model_sae_runner(cfg)

```


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

I wrote a tutorial to show users how to do some basic exploration of their SAE. 
- `evaluating_your_sae.ipynb`: A quick/dirty notebook showing how to check L0 and Prediction loss with your SAE, as well as showing how to generate interactive dashboards using Callum's reporduction of [Anthropics interface](https://transformer-circuits.pub/2023/monosemantic-features#setup-interface). 

## Example Dashboard

WandB Dashboards provide lots of useful insights while training SAE's. Here's a screenshot from one training run. 

![screenshot](content/dashboard_screenshot.pngdashboard_screenshot.png)


## Example Output

Here's one feature we found in the residual stream of Layer 10 of GPT-2 Small:

![alt text](content/readme_screenshot_predict_pronoun_feature.png). Open `gpt2_resid_pre10_predict_pronoun_feature.html` in your browser to interact with the dashboard (WIP).

Note, probably this feature could split into more mono-semantic features in a larger SAE that had been trained for longer. (this was was only about 49152 features trained on 10M tokens from OpenWebText).


## Citations and References:

Research:
- [Towards Monosemanticy](https://transformer-circuits.pub/2023/monosemantic-features)
- [Sparse Autoencoders Find Highly Interpretable Features in Language Model](https://arxiv.org/abs/2309.08600)



Reference Implementations:
- [Neel Nanda](https://github.com/neelnanda-io/1L-Sparse-Autoencoder)
- [AI-Safety-Foundation](https://github.com/ai-safety-foundation/sparse_autoencoder).
- [Arthur Conmy](https://github.com/ArthurConmy/sae).
- [Callum McDougall](https://github.com/callummcdougall/sae-exercises-mats/tree/main)
