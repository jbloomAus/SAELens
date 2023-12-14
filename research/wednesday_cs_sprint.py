# %% Set Up
import json
import os
import sys

sys.path.append("..")
from functools import partial
from pathlib import Path
from typing import Dict

import plotly.express as px
import torch
from datasets import load_dataset
from transformer_lens import utils

import wandb
from sae_analysis.visualizer import data_fns, html_fns
from sae_analysis.visualizer.data_fns import FeatureData, get_feature_data
from sae_training.utils import LMSparseAutoencoderSessionloader

if torch.backends.mps.is_available():
    device = "mps" 
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_grad_enabled(False)

def imshow(x, **kwargs):
    x_numpy = utils.to_numpy(x)
    px.imshow(x_numpy, **kwargs).show()
    
    
from typing import List, Tuple, Union

from jaxtyping import Float
from torch import Tensor

# %% Load Model
from transformer_lens import HookedTransformer

# %% [markdown]

# # Wednesday CS Sprint -> Max Activating Examples and Exploring SAE

# Stealing a bunch from Callum's notebook here (thanks Callum!)
# https://github.com/callummcdougall/SERI-MATS-2023-Streamlit-pages/blob/6a116ab47d903ea37b14edcfab96ffd097892cdb/transformer_lens/rs/callum/max_activating_exploration.ipynb


model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    # refactor_factored_attn_matrices=True,
)
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)

# %% Load Dataset / Model
import numpy as np
import torch
from datasets import load_dataset

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.sparse_autoencoder import SparseAutoencoder
from sae_training.utils import LMSparseAutoencoderSessionloader

# dataset = load_dataset("Skylion007/openwebtext", streaming=True)

def get_webtext(seed: int = 420, dataset="stas/openwebtext-10k") -> List[str]:
    """Get 10,000 sentences from the OpenWebText dataset"""

    # Let's see some WEBTEXT
    raw_dataset = load_dataset(dataset)
    train_dataset = raw_dataset["train"]
    dataset = [train_dataset[i]["text"] for i in range(len(train_dataset))]

    # Shuffle the dataset (I don't want the Hitler thing being first so use a seeded shuffle)
    np.random.seed(seed)
    np.random.shuffle(dataset)

    return dataset

data = get_webtext()
# %%

from tqdm import tqdm
from transformer_lens.components import HookPoint

LAYER_IDX, HEAD_IDX = (10, 7)
W_U = model.W_U.clone()
HEAD_HOOK_NAME = utils.get_act_name("result", LAYER_IDX)

NUM_PROMPTS = 100
BATCH_SIZE = 10

def hook_to_ablate_head(head_output: Float[Tensor, "batch seq_len head_idx d_head"], hook: HookPoint, head = (LAYER_IDX, HEAD_IDX)):
    assert head[0] == hook.layer()
    assert "result" in hook.name
    head_output[:, :, head[1], :] = 0
    return head_output


# %% [markdown]

# # Max activating examples for 10.7 (by ablation)
# Want to see where head 10.7 is most useful!

# We can see cross-entropy loss increases by 0.01 on average when this head is ablated. That might seem like not a lot, but it's actually not far off distribution to other late-stage heads.

# %%
str_token_list = []
loss_list = []
ablated_loss_list = []

for i in tqdm(range(NUM_PROMPTS)):
    # new_str = data[BATCH_SIZE * i: BATCH_SIZE * (i + 1)]
    new_str = data[i]
    new_str_tokens = model.to_str_tokens(new_str)
    tokens = model.to_tokens(new_str)
    # tokens = t.stack(tokens).to(device)
    loss = model(tokens, return_type="loss", loss_per_token=True)
    ablated_loss = model.run_with_hooks(tokens, return_type="loss", loss_per_token=True, fwd_hooks=[(HEAD_HOOK_NAME, hook_to_ablate_head)])
    loss_list.append(loss)
    ablated_loss_list.append(ablated_loss)
    str_token_list.append(new_str_tokens)


all_loss = torch.cat(loss_list, dim=-1).squeeze()
all_ablated_loss = torch.cat(ablated_loss_list, dim=-1).squeeze()


# %%

px.histogram(
    (all_ablated_loss - all_loss).detach().cpu().numpy(),
    title="Difference in loss after ablating (positive ⇒ loss increases)",
    labels={"x": "Difference in cross-entropy loss"},
    template="simple_white",
    # add_mean_line=True,
    width=1000,
    nbins=200,
    # static=True,
)


# %%

from importlib import reload

from callum import max_activating_exploration

reload(max_activating_exploration)

find_best_improvements = max_activating_exploration.find_best_improvements
total_num_tokens = sum(len(i) for i in str_token_list)
top_pct = int(total_num_tokens * 0.01)

best_k_indices, best_k_loss_decrease = find_best_improvements(str_token_list, loss_list, ablated_loss_list, k=top_pct)
worst_k_indices, worst_k_loss_decrease = find_best_improvements(str_token_list, loss_list, ablated_loss_list, k=top_pct, worst=True)

# %%
n=10
caches_and_tokens = max_activating_exploration.print_best_outputs(
    best_k_indices[:n],
    best_k_loss_decrease[:n],
    hook = (HEAD_HOOK_NAME, hook_to_ablate_head),
    model = model,
    data = data,
    n = n,
    random = False,
    return_caches = True,
    names_filter = lambda name: name == utils.get_act_name("pattern", LAYER_IDX),
)

# %%

n=10
caches_and_tokens = max_activating_exploration.print_best_outputs(
    worst_k_indices[:n],
    worst_k_loss_decrease[:n],
    hook = (HEAD_HOOK_NAME, hook_to_ablate_head),
    model = model,
    data = data,
    n = n,
    random = False,
    return_caches = True,
    names_filter = lambda name: name == utils.get_act_name("pattern", LAYER_IDX),
)

# %% 
import pandas as pd

# flatten list of lists and lost lists and put in dataframe
flat_list = [item for sublist in str_token_list for item in sublist[1:]]
flat_loss_list = [item.item() for sublist in loss_list for item in sublist.flatten()]
flat_ablated_loss_list = [item.item() for sublist in ablated_loss_list for item in sublist.flatten()]
loss_list_df = pd.DataFrame(list(zip(flat_list, flat_loss_list, flat_ablated_loss_list)), columns =['str_token', 'loss', 'ablated_loss'])
loss_list_df['loss_diff'] = loss_list_df['ablated_loss'] - loss_list_df['loss']
loss_list_df.head()


# %%
loss_list_df.sort_values(by=['loss_diff'], ascending=False).head(10)
# Ok so we have our examples of where the head reduces loss by firing, is that actually the thing we care about?
# Not really, in particular, I want to see where the head is most useful, via suppressing the same token?
# Hmmm, that seems like it will bias our investigation toward the CS theory (which is what we want toi test)

# I want a dataframe of correct token, predicted token, and the loss decrease attributable to the head firing. 
# 


# %%
str_input = "All's fair in love and"
answer = " war"
incorrect = " love"
model.reset_hooks()
utils.test_prompt(str_input, answer, model)

# %%

toks = model.to_tokens(str_input)

model.reset_hooks()
logits, cache = model.run_with_cache(toks, return_type="logits")
logits = logits[0, -1]

neg_head_output = cache["result", 10][0, -1, 7]
neg_head_logits = neg_head_output @ model.W_U
assert neg_head_logits.shape == (model.cfg.d_vocab,)
neg_head_logprobs = neg_head_logits.log_softmax(dim=-1)

top5 = neg_head_logprobs.topk(5, largest=False)

for index, value in zip(top5.indices, top5.values):
    token = model.to_single_str_token(index.item())
    print(f"|{token}| = {value:.2f}")
    
    
# %%
import circuitsvis as cv

all_attn = torch.concat([
    cache["pattern", layer][0] for layer in range(12)
])
html = cv.attention.attention_heads(
    attention=all_attn,
    tokens=model.to_str_tokens(toks),
    attention_head_names=[f"{layer}.{head_idx}" for layer in range(12) for head_idx in range(12)]
)
f = Path(r".")
with open(f / "temp_file_3.html", "w") as f2:
    f2.write(str(html))