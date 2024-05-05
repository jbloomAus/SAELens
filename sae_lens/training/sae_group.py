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


class SparseAutoencoderDictionary:

    autoencoders: dict[str, SparseAutoencoder]

    def __init__(self, cfg: LanguageModelSAERunnerConfig):
        self.cfg = cfg
        self.autoencoders = {}  # This will store tuples of (instance, hyperparameters)
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

            sae = SparseAutoencoder(cfg_copy)

            sae_name = (
                f"{sae.cfg.model_name.replace('/', '_')}_{sae.cfg.hook_point}_{sae.cfg.d_sae}_"
                + "_".join([f"{k}_{v}" for k, v in zip(keys, combination)])
            )

            # Create and store both the SparseAutoencoder instance and its parameters
            self.autoencoders[sae_name] = sae

    def __getitem__(self, key: str) -> SparseAutoencoder:
        return self.autoencoders[key]

    def __iter__(self) -> Iterator[tuple[str, SparseAutoencoder]]:
        # Make SAEGroup iterable over its SparseAutoencoder instances and their parameters
        for name, sae in self.autoencoders.items():
            yield name, sae  # Yielding as a tuple

    def __len__(self):
        # Return the number of SparseAutoencoder instances
        return len(self.autoencoders)

    def to(self, device: torch.device | str):
        for ae in self.autoencoders.values():
            ae.to(device)

    @classmethod
    def load_from_pretrained_legacy(cls, path: str) -> "SparseAutoencoderDictionary":
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
            cfg = group["cfg"]
            # need to add fields to old configs
            if not hasattr(cfg, "model_kwargs"):
                cfg.model_kwargs = {}
            if not hasattr(cfg, "sae_lens_version"):
                cfg.sae_lens_version = "0.0.0"
            if not hasattr(cfg, "sae_lens_training_version"):
                cfg.sae_lens_training_version = "0.0.0"
            sparse_autoencoder = SparseAutoencoder(cfg=cfg)
            # add dummy scaling factor to the state dict
            group["state_dict"]["scaling_factor"] = torch.ones(
                cfg.d_sae, dtype=cfg.dtype, device=cfg.device
            )
            sparse_autoencoder.load_state_dict(group["state_dict"])
            group = cls(cfg)
            for key in group.autoencoders:
                group.autoencoders[key] = sparse_autoencoder

        if not isinstance(group, cls):
            raise ValueError("The loaded object is not a valid SAEGroup")

        return group

    @classmethod
    def load_from_pretrained(
        cls, path: str, device: str = "cpu"
    ) -> "SparseAutoencoderDictionary":

        autoencoders = {}

        # check if there any folders inside the current path
        folders_in_current_path = [
            f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))
        ]
        if len(folders_in_current_path) == 0:
            # sae_name is the folder name
            sae_name = os.path.basename(path)
            autoencoders[sae_name] = SparseAutoencoder.load_from_pretrained(
                path,
                device,
            )
        else:  # we have a list of SAE folders
            for sae_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, sae_name)):
                    autoencoders[sae_name] = SparseAutoencoder.load_from_pretrained(
                        os.path.join(path, sae_name),
                        device,
                    )

        # Create the SAEGroup object

        # loaded SAE groups will not contain a cfg that matches the original.
        # TODO: figure out how to handle this
        sae_name = next(iter(autoencoders.keys()))
        sae_group = cls(autoencoders[sae_name].cfg)
        sae_group.autoencoders = autoencoders
        return sae_group

    def save_saes(self, path: str):
        """
        Basic save function for the model. Saves the model's state_dict and the config used to train it.
        """

        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        if len(self.autoencoders) == 1:
            sae: SparseAutoencoder = next(iter(self))[1]
            sae.save_model(path)
        else:
            for i, autoencoder in self.autoencoders.items():
                subfolder_name = f"{path}/{i}"
                os.makedirs(subfolder_name, exist_ok=True)
                autoencoder.save_model(f"{path}/{i}")

    def get_name(self):
        sae_name = f"sae_group_{self.cfg.model_name.replace('/', '_')}_{self.cfg.hook_point}_{self.cfg.d_sae}"
        return sae_name

    def eval(self):
        for ae in self.autoencoders.values():
            ae.eval()

    def train(self):
        for ae in self.autoencoders.values():
            ae.train()
