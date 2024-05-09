"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import gzip
import json
import os
import pickle
from typing import Callable, NamedTuple, Optional

import einops
import torch
from jaxtyping import Float
from safetensors.torch import save_file
from torch import nn
from transformer_lens.hook_points import HookedRootModule, HookPoint

from sae_lens.toolkit.pretrained_sae_loaders import (
    NAMED_PRETRAINED_SAE_LOADERS,
    load_pretrained_sae_lens_sae_components,
)
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.training.activation_functions import get_activation_fn
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.utils import BackwardsCompatiblePickleClass

SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"


class ForwardOutput(NamedTuple):
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    loss: torch.Tensor
    mse_loss: torch.Tensor
    l1_loss: torch.Tensor
    ghost_grad_loss: torch.Tensor | float


class SparseAutoencoder(HookedRootModule):
    """ """

    l1_coefficient: float
    lp_norm: float
    d_sae: int
    use_ghost_grads: bool
    normalize_sae_decoder: bool
    hook_point_layer: int
    dtype: torch.dtype
    device: str | torch.device
    noise_scale: float
    activation_fn: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}"
            )
        assert cfg.d_sae is not None  # keep pyright happy
        # lists are valid only for SAEGroup cfg, not SAE cfg vals
        assert not isinstance(cfg.l1_coefficient, list)
        assert not isinstance(cfg.lp_norm, list)
        assert not isinstance(cfg.lr, list)
        assert not isinstance(cfg.lr_scheduler_name, list)
        assert not isinstance(cfg.lr_warm_up_steps, list)
        assert not isinstance(cfg.use_ghost_grads, list)
        assert not isinstance(cfg.hook_point_layer, list)
        assert (
            "{layer}" not in cfg.hook_point
        ), "{layer} must be replaced with the actual layer number in SAE cfg"

        self.d_sae = cfg.d_sae
        self.l1_coefficient = cfg.l1_coefficient
        self.lp_norm = cfg.lp_norm
        self.dtype = cfg.dtype
        self.device = cfg.device
        self.use_ghost_grads = cfg.use_ghost_grads
        self.normalize_sae_decoder = cfg.normalize_sae_decoder
        self.hook_point_layer = cfg.hook_point_layer
        self.noise_scale = cfg.noise_scale
        self.activation_fn = get_activation_fn(cfg.activation_fn)

        if self.cfg.scale_sparsity_penalty_by_decoder_norm:
            self.get_sparsity_loss_term = self.get_sparsity_loss_term_decoder_norm
        else:
            self.get_sparsity_loss_term = self.get_sparsity_loss_term_standard

        self.initialize_weights()

        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()

        self.setup()  # Required for `HookedRootModule`s

    def initialize_weights(self):
        """
        Wrapped around weight initialization code to make init cleaner.

        """

        # no config changes encoder bias init for now.
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )

        # Start with the default init strategy:
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_sae, self.d_in, dtype=self.dtype, device=self.device)
            )
        )
        if self.cfg.decoder_orthogonal_init:
            self.W_dec.data = nn.init.orthogonal_(self.W_dec.data.T).T

        elif self.cfg.decoder_heuristic_init:
            self.W_dec = nn.Parameter(
                torch.rand(self.d_sae, self.d_in, dtype=self.dtype, device=self.device)
            )
            self.initialize_decoder_norm_constant_norm()

        elif self.cfg.normalize_sae_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device)
            )
        )

        # Then we intialize the encoder weights (either as the transpose of decoder or not)
        if self.cfg.init_encoder_as_decoder_transpose:
            self.W_enc.data = self.W_dec.data.T.clone().contiguous()
        else:
            self.W_enc = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(
                        self.d_in, self.d_sae, dtype=self.dtype, device=self.device
                    )
                )
            )

        if self.normalize_sae_decoder:
            with torch.no_grad():
                # Anthropic normalize this to have unit columns
                self.set_decoder_norm_to_unit_norm()

        # methdods which change b_dec as a function of the dataset are implemented after init.
        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
        )

        # scaling factor for fine-tuning (not to be used in initial training)
        self.scaling_factor = nn.Parameter(
            torch.ones(self.d_sae, dtype=self.dtype, device=self.device)
        )

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        feature_acts, _ = self._encode_with_hidden_pre(x)
        return feature_acts

    def _encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        """Encodes input activation tensor x into an SAE feature activation tensor."""
        # move x to correct dtype
        x = x.to(self.dtype)
        sae_in = self.hook_sae_in(
            x - (self.b_dec * self.cfg.apply_b_dec_to_input)
        )  # Remove decoder bias as per Anthropic

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_enc
        )
        noisy_hidden_pre = hidden_pre
        if self.noise_scale > 0:
            noise = torch.randn_like(hidden_pre) * self.noise_scale
            noisy_hidden_pre = hidden_pre + noise
        feature_acts = self.hook_hidden_post(self.activation_fn(noisy_hidden_pre))

        return feature_acts, hidden_pre

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """Decodes SAE feature activation tensor into a reconstructed input activation tensor."""
        sae_out = self.hook_sae_out(
            einops.einsum(
                feature_acts
                * self.scaling_factor,  # need to make sure this handled when loading old models.
                self.W_dec,
                "... d_sae, d_sae d_in -> ... d_in",
            )
            + self.b_dec
        )
        return sae_out

    def get_sparsity_loss_term_standard(
        self, feature_acts: torch.Tensor
    ) -> torch.Tensor:
        """
        Sparsity loss term calculated as the L1 norm of the feature activations.
        """
        sparsity = feature_acts.norm(p=self.lp_norm, dim=-1)
        return sparsity

    def get_sparsity_loss_term_decoder_norm(
        self, feature_acts: torch.Tensor
    ) -> torch.Tensor:
        """
        Sparsity loss term for decoder norm regularization.
        """
        weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
        sparsity = weighted_feature_acts.norm(
            p=self.lp_norm, dim=-1
        )  # sum over the feature dimension
        return sparsity

    def forward(
        self, x: torch.Tensor, dead_neuron_mask: torch.Tensor | None = None
    ) -> ForwardOutput:

        feature_acts, hidden_pre = self._encode_with_hidden_pre(x)
        sae_out = self.decode(feature_acts)

        # add config for whether l2 is normalized:
        per_item_mse_loss = _per_item_mse_loss_with_target_norm(
            sae_out, x, self.cfg.mse_loss_normalization
        )

        # gate on config and training so evals is not slowed down.
        if (
            self.use_ghost_grads
            and self.training
            and dead_neuron_mask is not None
            and dead_neuron_mask.sum() > 0
        ):
            ghost_grad_loss = self.calculate_ghost_grad_loss(
                x=x,
                sae_out=sae_out,
                per_item_mse_loss=per_item_mse_loss,
                hidden_pre=hidden_pre,
                dead_neuron_mask=dead_neuron_mask,
            )
        else:
            ghost_grad_loss = 0

        mse_loss = per_item_mse_loss.sum(dim=-1).mean()
        sparsity = self.get_sparsity_loss_term(feature_acts)
        l1_loss = (self.l1_coefficient * sparsity).mean()
        loss = mse_loss + l1_loss + ghost_grad_loss

        return ForwardOutput(
            sae_out=sae_out,
            feature_acts=feature_acts,
            loss=loss,
            mse_loss=mse_loss,
            l1_loss=l1_loss,
            ghost_grad_loss=ghost_grad_loss,
        )

    @torch.no_grad()
    def initialize_b_dec_with_precalculated(self, origin: torch.Tensor):
        out = torch.tensor(origin, dtype=self.dtype, device=self.device)
        self.b_dec.data = out

    @torch.no_grad()
    def initialize_b_dec_with_mean(self, all_activations: torch.Tensor):
        previous_b_dec = self.b_dec.clone().cpu()
        out = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec with mean of activations")
        print(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        print(f"New distances: {distances.median(0).values.mean().item()}")

        self.b_dec.data = out.to(self.dtype).to(self.device)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def initialize_decoder_norm_constant_norm(self, norm: float = 0.1):
        """
        A heuristic proceedure inspired by:
        https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        """
        # TODO: Parameterise this as a function of m and n

        # ensure W_dec norms at unit norm
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data *= 0.1  # will break tests but do this for now.

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )

    def save_model_legacy(self, path: str):
        """
        Basic save function for the model. Saves the model's state_dict and the config used to train it.
        """

        # check if path exists
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        state_dict = {"cfg": self.cfg, "state_dict": self.state_dict()}

        if path.endswith(".pt"):
            torch.save(state_dict, path)
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt and .pkl.gz"
            )

        print(f"Saved model to {path}")

    def save_model(self, path: str, sparsity: Optional[torch.Tensor] = None):

        if not os.path.exists(path):
            os.mkdir(path)

        # generate the weights
        save_file(self.state_dict(), f"{path}/{SAE_WEIGHTS_PATH}")

        # save the config
        config = {
            **self.cfg.__dict__,
            # some args may not be serializable by default
            "dtype": str(self.cfg.dtype),
            "device": str(self.cfg.device),
        }

        with open(f"{path}/{SAE_CFG_PATH}", "w") as f:
            json.dump(config, f)

        if sparsity is not None:
            sparsity_in_dict = {"sparsity": sparsity}
            save_file(sparsity_in_dict, f"{path}/{SPARSITY_PATH}")  # type: ignore

    @classmethod
    def load_from_pretrained_legacy(cls, path: str):
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
                if torch.backends.mps.is_available():
                    state_dict = torch.load(
                        path,
                        map_location="mps",
                        pickle_module=BackwardsCompatiblePickleClass,
                    )
                    state_dict["cfg"].device = "mps"
                else:
                    state_dict = torch.load(
                        path, pickle_module=BackwardsCompatiblePickleClass
                    )
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pt file: {e}")
        elif path.endswith(".pkl.gz"):
            try:
                with gzip.open(path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(
                    f"Error loading the state dictionary from .pkl.gz file: {e}"
                )
        elif path.endswith(".pkl"):
            try:
                with open(path, "rb") as f:
                    state_dict = pickle.load(f)
            except Exception as e:
                raise IOError(f"Error loading the state dictionary from .pkl file: {e}")
        else:
            raise ValueError(
                f"Unexpected file extension: {path}, supported extensions are .pt, .pkl, and .pkl.gz"
            )

        # Ensure the loaded state contains both 'cfg' and 'state_dict'
        if "cfg" not in state_dict or "state_dict" not in state_dict:
            raise ValueError(
                "The loaded state dictionary must contain 'cfg' and 'state_dict' keys"
            )

        # Create an instance of the class using the loaded configuration
        instance = cls(cfg=state_dict["cfg"])
        if "scaling_factor" not in state_dict["state_dict"]:
            assert isinstance(instance.cfg.d_sae, int)
            state_dict["state_dict"]["scaling_factor"] = torch.ones(
                instance.cfg.d_sae, dtype=instance.cfg.dtype, device=instance.cfg.device
            )
        instance.load_state_dict(state_dict["state_dict"], strict=True)

        return instance

    @classmethod
    def load_from_pretrained(
        cls, path: str, device: str = "cpu"
    ) -> "SparseAutoencoder":

        config_path = os.path.join(path, "cfg.json")
        weight_path = os.path.join(path, "sae_weights.safetensors")

        cfg, state_dict = load_pretrained_sae_lens_sae_components(
            config_path, weight_path, device
        )

        sae = cls(cfg)
        sae.load_state_dict(state_dict)

        return sae

    @classmethod
    def from_pretrained(
        cls, release: str, sae_id: str, device: str = "cpu"
    ) -> "SparseAutoencoder":
        """

        Load a pretrained SAE from the Hugging Face model hub.

        Args:
            release: The release name. This will be mapped to a huggingface repo id based on the pretrained_saes.yaml file.
            id: The id of the SAE to load. This will be mapped to a path in the huggingface repo.
            device: The device to load the SAE on.

        """

        # get sae directory
        sae_directory = get_pretrained_saes_directory()

        # get the repo id and path to the SAE
        if release not in sae_directory:
            raise ValueError(
                f"Release {release} not found in pretrained SAEs directory."
            )
        if sae_id not in sae_directory[release].saes_map:
            raise ValueError(f"ID {sae_id} not found in release {release}.")
        sae_info = sae_directory[release]
        hf_repo_id = sae_info.repo_id
        hf_path = sae_info.saes_map[sae_id]

        conversion_loader_name = sae_info.conversion_func or "sae_lens"
        if conversion_loader_name not in NAMED_PRETRAINED_SAE_LOADERS:
            raise ValueError(
                f"Conversion func {conversion_loader_name} not found in NAMED_PRETRAINED_SAE_LOADERS."
            )
        conversion_loader = NAMED_PRETRAINED_SAE_LOADERS[conversion_loader_name]

        cfg, state_dict = conversion_loader(
            repo_id=hf_repo_id,
            folder_name=hf_path,
            device=device,
            force_download=False,
        )

        sae = cls(cfg)
        sae.load_state_dict(state_dict)

        return sae

    def get_name(self):
        sae_name = f"sparse_autoencoder_{self.cfg.model_name}_{self.cfg.hook_point}_{self.cfg.d_sae}"
        return sae_name

    def calculate_ghost_grad_loss(
        self,
        x: torch.Tensor,
        sae_out: torch.Tensor,
        per_item_mse_loss: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 1.
        residual = x - sae_out
        l2_norm_residual = torch.norm(residual, dim=-1)

        # 2.
        feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
        ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
        l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
        norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
        ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

        # 3.
        per_item_mse_loss_ghost_resid = _per_item_mse_loss_with_target_norm(
            ghost_out, residual.detach(), self.cfg.mse_loss_normalization
        )
        mse_rescaling_factor = (
            per_item_mse_loss / (per_item_mse_loss_ghost_resid + 1e-6)
        ).detach()
        per_item_mse_loss_ghost_resid = (
            mse_rescaling_factor * per_item_mse_loss_ghost_resid
        )

        return per_item_mse_loss_ghost_resid.mean()


def _per_item_mse_loss_with_target_norm(
    preds: torch.Tensor,
    target: torch.Tensor,
    mse_loss_normalization: Optional[str] = None,
) -> torch.Tensor:
    """
    Calculate MSE loss per item in the batch, without taking a mean.
    Then, optionally, normalizes by the L2 norm of the centered target.
    This normalization seems to improve performance.
    """
    if mse_loss_normalization == "dense_batch":
        target_centered = target - target.mean(dim=0, keepdim=True)
        normalization = target_centered.norm(dim=-1, keepdim=True)
        return torch.nn.functional.mse_loss(preds, target, reduction="none") / (
            normalization + 1e-6
        )
    else:
        return torch.nn.functional.mse_loss(preds, target, reduction="none")
