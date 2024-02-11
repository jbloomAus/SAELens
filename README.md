# Training SAEs on OthelloGPT

We forked Joseph Bloom's [SAE training codebase](https://github.com/jbloomAus/mats_sae_training), and are using it to train SAEs on OthelloGPT.

## Set Up

```

conda create --name mats_sae_training python=3.11 -y
conda activate mats_sae_training
pip install -r requirements.txt

```

If `conda activate mats_sae_training` doesn't work, try `source activate mats_sae_training`.

## Files

- `othellogpt_train_sae.ipynb` - notebook to train SAEs on OthelloGPT
- `othellogpt_probe_analysis.ipynb` - compare SAE enc/dec directions with probe directions
- `othellogpt_interp.ipynb`
- `othellogpt_board_analysis.ipynb`
