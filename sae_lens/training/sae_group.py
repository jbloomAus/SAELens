from __future__ import annotations

import dataclasses
import gzip
import os
import pickle
from itertools import product
from types import SimpleNamespace
from typing import Iterator

import torch

from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from sae_lens.training.utils import BackwardsCompatibleUnpickler


class SAEGroup:

    autoencoders: list[SparseAutoencoder]

    def __init__(self, cfg: LanguageModelSAERunnerConfig):
        self.cfg = cfg
        self.autoencoders = []  # This will store tuples of (instance, hyperparameters)
        self._init_autoencoders(cfg)

    def _init_autoencoders(self, cfg: LanguageModelSAERunnerConfig):
        # Dynamically get all combinations of hyperparameters from cfg
        # Extract all hyperparameter lists from cfg
        hyperparameters = {k: v for k, v in vars(cfg).items() if isinstance(v, list)}
        if len(hyperparameters) > 0:
            keys, values = zip(*hyperparameters.items())
        else:
            keys, values = (), ([()],)  # Ensure product(*values) yields one combination

        # Create all combinations of hyperparameters
        for combination in product(*values):
            params = dict(zip(keys, combination))
            cfg_copy = dataclasses.replace(cfg, **params)
            cfg_copy.__post_init__()
            # Insert the layer into the hookpoint
            cfg_copy.hook_point = cfg_copy.hook_point.format(
                layer=cfg_copy.hook_point_layer
            )
            # Create and store both the SparseAutoencoder instance and its parameters
            self.autoencoders.append(SparseAutoencoder(cfg_copy))

    def __iter__(self) -> Iterator[SparseAutoencoder]:
        # Make SAEGroup iterable over its SparseAutoencoder instances and their parameters
        for ae in self.autoencoders:
            yield ae  # Yielding as a tuple

    def __len__(self):
        # Return the number of SparseAutoencoder instances
        return len(self.autoencoders)

    def to(self, device: torch.device | str):
        for ae in self.autoencoders:
            ae.to(device)

    @classmethod
    def load_from_pretrained(cls, path: str) -> "SAEGroup":
        """
        Load function for the model. Loads the model's state_dict and the config used to train it.
        This method can be called directly on the class, without needing an instance.
        """

        # Ensure the file exists
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No file found at specified path: {path}")

        # Load the state dictionary
        if path.endswith(".pt"):
            try:
                # this is hacky, but can't figure out how else to get torch to use our custom unpickler
                fake_pickle = SimpleNamespace()
                fake_pickle.Unpickler = BackwardsCompatibleUnpickler
                fake_pickle.__name__ = pickle.__name__

                if torch.cuda.is_available():
                    group = torch.load(
                        path,
                        pickle_module=fake_pickle,
                    )
                else:
                    map_loc = "mps" if torch.backends.mps.is_available() else "cpu"
                    group = torch.load(
                        path, pickle_module=fake_pickle, map_location=map_loc
                    )
                    if isinstance(group, dict):
                        group["cfg"].device = map_loc
                    else:
                        group.cfg.device = map_loc
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pt file: {e}")

        elif path.endswith(".pkl.gz"):
            try:
                with gzip.open(path, "rb") as f:
                    group = pickle.load(f)
            except Exception as e:
                raise IOError(
                    f"Error loading the state dictionary from .pkl.gz file: {e}"
                )
        elif path.endswith(".pkl"):
            try:
                with open(path, "rb") as f:
                    group = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl file: {e}")
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt, .pkl, and .pkl.gz"
            )

        # handle loading old autoencoders where before SAEGroup existed, where we just save a dict
        if isinstance(group, dict):
            sparse_autoencoder = SparseAutoencoder(cfg=group["cfg"])
            sparse_autoencoder.load_state_dict(group["state_dict"])
            group = cls(group["cfg"])
            group.autoencoders[0] = sparse_autoencoder

        if not isinstance(group, cls):
            raise ValueError("The loaded object is not a valid SAEGroup")

        return group

    def save_model(self, path: str):
        """
        Basic save function for the model. Saves the model's state_dict and the config used to train it.
        """

        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        if path.endswith(".pt"):
            torch.save(self, path)
        elif path.endswith("pkl.gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(self, f)
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt and .pkl.gz"
            )

        print(f"Saved model to {path}")

    def get_name(self):
        layers = self.cfg.hook_point_layer
        if not isinstance(layers, list):
            layers = [layers]
        if len(layers) > 1:
            layer_string = f"{min(layers)-max(layers)}"
        else:
            layer_string = f"{layers[0]}"
        sae_name = f"sae_group_{self.cfg.model_name}_{self.cfg.hook_point.format(layer=layer_string)}_{self.cfg.d_sae}"
        return sae_name

    def eval(self):
        for ae in self.autoencoders:
            ae.eval()

    def train(self):
        for ae in self.autoencoders:
            ae.train()
