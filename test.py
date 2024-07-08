# %%
from sae_lens.sae import SAE

sae = SAE.load_from_pretrained(
    path="/media/curttigges/project-files/projects/Mistral-7B-Residual-Stream-SAEs/mistral_7b_layer_8",
    device="cuda:0",
)
print(sae.device)
# %%
