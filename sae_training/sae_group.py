from functools import partial
import torch.nn.functional as F
import dataclasses
from sae_training.sparse_autoencoder import SparseAutoencoder


class SAEGroup:
    def __init__(self, cfg):
        self.cfg = cfg
        self.autoencoders = []  # This will store tuples of (instance, hyperparameters)
        self._init_autoencoders(cfg)

    def _init_autoencoders(self, cfg):
        # Dynamically get all combinations of hyperparameters from cfg
        from itertools import product

        # Extract all hyperparameter lists from cfg
        hyperparameters = {k: v for k, v in vars(cfg).items() if isinstance(v, list)}
        keys, values = zip(*hyperparameters.items())

        # Create all combinations of hyperparameters
        for combination in product(*values):
            cfg_copy = dataclasses.replace(cfg)
            params = dict(zip(keys, combination))
            cfg_copy.update(params)
            # Insert the layer into the hookpoint
            cfg_copy.hook_point = cfg_copy.hook_point.format(layer=cfg.copy.hook_point_layer)
            # Create and store both the SparseAutoencoder instance and its parameters
            self.autoencoders.append((SparseAutoencoder(cfg_copy), cfg_copy))

    def __iter__(self):
        # Make SAEGroup iterable over its SparseAutoencoder instances and their parameters
        for ae, params in self.autoencoders:
            yield ae, params  # Yielding as a tuple

    def __len__(self):
        # Return the number of SparseAutoencoder instances
        return len(self.autoencoders)