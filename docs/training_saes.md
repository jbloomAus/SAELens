# Training Sparse Autoencoders

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

from sae_lens import LanguageModelSAERunnerConfig, language_model_sae_runner

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
    training_tokens = 1_000_000 * 300,
    store_batch_size = 32,
    
    # Dead Neurons and Sparsity
    use_ghost_grads=True,
    feature_sampling_window = 1000,
    dead_feature_window=5000,
    dead_feature_threshold = 1e-6,
    
    # WANDB
    log_to_wandb = True,
    wandb_project= "gpt2",
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
