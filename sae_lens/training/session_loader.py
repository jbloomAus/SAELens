from typing import Tuple

from transformer_lens import HookedTransformer

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.sae_group import SparseAutoencoderDictionary
from sae_lens.training.sparse_autoencoder import SparseAutoencoder


class LMSparseAutoencoderSessionloader:
    """
    Responsible for loading all required
    artifacts and files for training
    a sparse autoencoder on a language model
    or analysing a pretraining autoencoder
    """

    def __init__(self, cfg: LanguageModelSAERunnerConfig):
        self.cfg = cfg

    def load_sae_training_group_session(
        self,
    ) -> Tuple[HookedTransformer, SparseAutoencoderDictionary, ActivationsStore]:
        """
        Loads a session for training a sparse autoencoder on a language model.
        """

        model = self.get_model(self.cfg.model_name)
        model.to(self.cfg.device)
        activations_loader = ActivationsStore.from_config(
            model,
            self.cfg,
        )

        sae_group = SparseAutoencoderDictionary(self.cfg)

        return model, sae_group, activations_loader

    @classmethod
    def load_pretrained_sae(
        cls, path: str, device: str = "cpu"
    ) -> Tuple[HookedTransformer, SparseAutoencoderDictionary, ActivationsStore]:
        """
        Loads a session for analysing a pretrained sparse autoencoder.
        """

        # load the SAE
        sparse_autoencoder = SparseAutoencoder.load_from_pretrained(path)
        sparse_autoencoder.to(device)
        sparse_autoencoder.cfg.device = device

        # load the model, SAE and activations loader with it.
        session_loader = cls(sparse_autoencoder.cfg)
        model, sae_group, activations_loader = (
            session_loader.load_sae_training_group_session()
        )

        return model, sae_group, activations_loader

    def get_model(self, model_name: str) -> HookedTransformer:
        """
        Loads a model from transformer lens.

        Abstracted to allow for easy modification.
        """

        # Todo: add check that model_name is valid

        model = HookedTransformer.from_pretrained(
            model_name,
            device=self.cfg.device,
        )

        return model
