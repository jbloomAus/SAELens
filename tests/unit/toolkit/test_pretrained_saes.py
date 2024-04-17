import pathlib
import shutil

import pytest
import torch

from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from sae_lens.toolkit import pretrained_saes
from sae_lens.training.config import LanguageModelSAERunnerConfig

def test_convert_old_to_modern_saelens_format():
    out_dir = pathlib.Path('unit_test_tmp')
    out_dir.mkdir(exist_ok=True)
    legacy_out_file = str(out_dir/'test.pt')
    new_out_folder = str(out_dir/'test')

    #Make an SAE, save old version
    cfg = LanguageModelSAERunnerConfig(
        dtype=torch.float32,
        hook_point = 'blocks.0.hook_mlp_out',
    )
    old_sae = SparseAutoencoder(cfg)
    old_sae.save_model_legacy(legacy_out_file)

    #convert file format
    pretrained_saes.convert_old_to_modern_saelens_format(
        legacy_out_file,
        new_out_folder
    )

    #Load from new converted file
    new_sae = SparseAutoencoder.load_from_pretrained(
        new_out_folder
    )
    shutil.rmtree(out_dir) #cleanup

    #Test similarity
    assert torch.allclose(new_sae.W_enc, old_sae.W_enc)
    assert torch.allclose(new_sae.W_dec, old_sae.W_dec)