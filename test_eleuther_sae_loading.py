#%%
from datasets import load_dataset
from transformer_lens import HookedTransformer

from sae_lens import SAE
from sae_lens.toolkit.pretrained_sae_loaders import eleuther_llama3_loader

#%%
cfg, state_dict, _ = eleuther_llama3_loader(
    repo_id="EleutherAI/sae-llama-3-8b-32x",
    folder_name="layers.11",
    device="cuda"
)
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release = "sae-llama-3-8b-eai",
    sae_id = "blocks.11.hook_resid_pre",
    device = "cuda"
)
# %%
cfg
# %%
state_dict.keys()
# %%
state_dict["b_enc"].shape
# %%
