# import pandas as pd
# import plotly.express as px
# import pytest
import torch
from tqdm import tqdm

# from sae_lens.training.activations_store import ActivationsStore
# from sae_lens.training.evals import run_evals
from sae_lens.sae import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

# from transformer_lens import HookedTransformer

# from tests.unit.helpers import load_model_cached


def test_eval_all_loadable_saes():

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"

    saes_directory = get_pretrained_saes_directory()
    for release, lookup in tqdm(saes_directory.items()):
        for sae_name, sae_info in lookup.saes_map.items():
            print(f"Loading {sae_name} from {release}")
            print(sae_info)
            sae, cfg_dict = SAE.from_pretrained(release, sae_name, device=device)
            print(sae)
            print(cfg_dict)
