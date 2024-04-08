from typing import Tuple

from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.sae_group import SAEGroup


class LMSparseAutoencoderSessionloader:
    """
    Responsible for loading all required
    artifacts and files for training
    a sparse autoencoder on a language model
    or analysing a pretraining autoencoder
    """

    def __init__(self, cfg: LanguageModelSAERunnerConfig):
        self.cfg = cfg

    def load_session(
        self,
    ) -> Tuple[HookedTransformer, SAEGroup, ActivationsStore]:
        """
        Loads a session for training a sparse autoencoder on a language model.
        """

        model = self.get_model(self.cfg.model_name)
        model.to(self.cfg.device)
        activations_loader = self.get_activations_loader(self.cfg, model)
        sparse_autoencoder = self.initialize_sparse_autoencoder(self.cfg)

        return model, sparse_autoencoder, activations_loader

    @classmethod
    def load_session_from_pretrained(
        cls, path: str
    ) -> Tuple[HookedTransformer, SAEGroup, ActivationsStore]:
        """
        Loads a session for analysing a pretrained sparse autoencoder group.
        """
        sparse_autoencoders = SAEGroup.load_from_pretrained(path)
        model, _, activations_loader = cls(sparse_autoencoders.cfg).load_session()
        return model, sparse_autoencoders, activations_loader

    def get_model(self, model_name: str):
        """
        Loads a model from transformer lens
        """

        # Todo: add check that model_name is valid

        model = HookedTransformer.from_pretrained(model_name)

        return model

    def initialize_sparse_autoencoder(self, cfg: LanguageModelSAERunnerConfig):
        """
        Initializes a sparse autoencoder group, which contains multiple sparse autoencoders
        """

        sparse_autoencoder = SAEGroup(cfg)

        return sparse_autoencoder

    def get_activations_loader(
        self, cfg: LanguageModelSAERunnerConfig, model: HookedTransformer
    ):
        """
        Loads a DataLoaderBuffer for the activations of a language model.
        """

        activations_loader = ActivationsStore.from_config(
            model,
            cfg,
        )

        return activations_loader
