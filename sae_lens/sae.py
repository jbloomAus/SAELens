"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import einops
import torch
from jaxtyping import Float
from safetensors.torch import save_file
from torch import nn
from transformer_lens.hook_points import HookedRootModule, HookPoint

from sae_lens.config import DTYPE_MAP
from sae_lens.toolkit.pretrained_sae_loaders import (
    NAMED_PRETRAINED_SAE_LOADERS,
    load_pretrained_sae_lens_sae_components,
)
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"


@dataclass
class SAEConfig:

    # forward pass details.
    d_in: int
    d_sae: int
    activation_fn_str: str
    apply_b_dec_to_input: bool
    finetuning_scaling_factor: bool

    # dataset it was trained on details.
    context_size: int
    model_name: str
    hook_name: str
    hook_layer: int
    hook_head_index: Optional[int]
    prepend_bos: bool
    dataset_path: str
    normalize_activations: bool

    # misc
    dtype: str
    device: str
    sae_lens_training_version: Optional[str]

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAEConfig":

        # rename dict:
        rename_dict = {  # old : new
            "hook_point": "hook_name",
            "hook_point_head_index": "hook_head_index",
            "hook_point_layer": "hook_layer",
            "activation_fn": "activation_fn_str",
        }
        config_dict = {rename_dict.get(k, k): v for k, v in config_dict.items()}

        # use only config terms that are in the dataclass
        config_dict = {
            k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__  # type: ignore
        }
        return cls(**config_dict)

    # def __post_init__(self):

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "dtype": self.dtype,
            "device": self.device,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "hook_head_index": self.hook_head_index,
            "activation_fn_str": self.activation_fn_str,  # use string for serialization
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "finetuning_scaling_factor": self.finetuning_scaling_factor,
            "sae_lens_training_version": self.sae_lens_training_version,
            "prepend_bos": self.prepend_bos,
            "dataset_path": self.dataset_path,
            "context_size": self.context_size,
            "normalize_activations": self.normalize_activations,
        }


class SAE(HookedRootModule):
    """ """

    cfg: SAEConfig
    dtype: torch.dtype
    device: torch.device

    # analysis
    use_error_term: bool

    def __init__(
        self,
        cfg: SAEConfig,
        use_error_term: bool = False,
    ):
        super().__init__()

        self.cfg = cfg
        self.activation_fn = get_activation_fn(cfg.activation_fn_str)
        self.dtype = DTYPE_MAP[cfg.dtype]
        self.device = torch.device(cfg.device)
        self.use_error_term = use_error_term

        self.initialize_weights_basic()

        # handle presence / absence of scaling factor.
        if self.cfg.finetuning_scaling_factor:
            self.apply_finetuning_scaling_factor = (
                lambda x: x * self.finetuning_scaling_factor
            )
        else:
            self.apply_finetuning_scaling_factor = lambda x: x

        # set up hooks
        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint()
        self.hook_sae_output = HookPoint()
        self.hook_sae_recons = HookPoint()
        self.hook_sae_error = HookPoint()

        # handle hook_z reshaping if needed.
        # this is very cursed and should be refactored. it exists so that we can reshape out
        # the z activations for hook_z SAEs. but don't know d_head if we split up the forward pass
        # into a separate encode and decode function.
        # this will cause errors if we call decode before encode.
        if self.cfg.hook_name.endswith("_z"):
            self.turn_on_forward_pass_hook_z_reshaping()
        else:
            # need to default the reshape fns
            self.turn_off_forward_pass_hook_z_reshaping()

        self.setup()  # Required for `HookedRootModule`s

    def initialize_weights_basic(self):

        # no config changes encoder bias init for now.
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )

        # Start with the default init strategy:
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
        )

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
                )
            )
        )

        # methdods which change b_dec as a function of the dataset are implemented after init.
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

        # scaling factor for fine-tuning (not to be used in initial training)
        # TODO: Make this optional and not included with all SAEs by default (but maintain backwards compatibility)
        if self.cfg.finetuning_scaling_factor:
            self.finetuning_scaling_factor = nn.Parameter(
                torch.ones(self.cfg.d_sae, dtype=self.dtype, device=self.device)
            )

    # Basic Forward Pass Functionality.
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        if self.use_error_term:
            with torch.no_grad():
                # Recompute everything without hooks to get true error term
                # Otherwise, the output with error term will always equal input, even for causal interventions that affect x_reconstruct
                # This is in a no_grad context to detach the error, so we can compute SAE feature gradients (eg for attribution patching). See A.3 in https://arxiv.org/pdf/2403.19647.pdf for more detail
                # NOTE: we can't just use `sae_error = input - x_reconstruct.detach()` or something simpler, since this would mean intervening on features would mean ablating features still results in perfect reconstruction.

                # move x to correct dtype
                x = x.to(self.dtype)

                # handle hook z reshaping if needed.
                sae_in = self.reshape_fn_in(x)  # type: ignore

                # apply b_dec_to_input if using that method.
                sae_in_cent = sae_in - (self.b_dec * self.cfg.apply_b_dec_to_input)

                # "... d_in, d_in d_sae -> ... d_sae",
                hidden_pre = sae_in_cent @ self.W_enc + self.b_enc
                feature_acts = self.activation_fn(hidden_pre)
                x_reconstruct_clean = self.reshape_fn_out(
                    self.apply_finetuning_scaling_factor(feature_acts) @ self.W_dec
                    + self.b_dec,
                    d_head=self.d_head,
                )

                sae_error = self.hook_sae_error(x - x_reconstruct_clean)

            return self.hook_sae_output(sae_out + sae_error)

        return self.hook_sae_output(sae_out)

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:

        # move x to correct dtype
        x = x.to(self.dtype)

        # handle hook z reshaping if needed.
        x = self.reshape_fn_in(x)  # type: ignore

        # apply b_dec_to_input if using that method.
        sae_in = self.hook_sae_input(x - (self.b_dec * self.cfg.apply_b_dec_to_input))

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))

        return feature_acts

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """Decodes SAE feature activation tensor into a reconstructed input activation tensor."""
        # "... d_sae, d_sae d_in -> ... d_in",
        sae_out = self.hook_sae_recons(
            self.apply_finetuning_scaling_factor(feature_acts) @ self.W_dec + self.b_dec
        )

        # handle hook z reshaping if needed.
        sae_out = self.reshape_fn_out(sae_out, self.d_head)  # type: ignore

        return sae_out

    @torch.no_grad()
    def fold_W_dec_norm(self):
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T
        self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()

    @torch.no_grad()
    def fold_activation_norm_scaling_factor(
        self, activation_norm_scaling_factor: float
    ):
        self.W_enc.data = self.W_enc.data * activation_norm_scaling_factor

    def save_model(self, path: str, sparsity: Optional[torch.Tensor] = None):

        if not os.path.exists(path):
            os.mkdir(path)

        # generate the weights
        save_file(self.state_dict(), f"{path}/{SAE_WEIGHTS_PATH}")

        # save the config
        config = self.cfg.to_dict()

        with open(f"{path}/{SAE_CFG_PATH}", "w") as f:
            json.dump(config, f)

        if sparsity is not None:
            sparsity_in_dict = {"sparsity": sparsity}
            save_file(sparsity_in_dict, f"{path}/{SPARSITY_PATH}")  # type: ignore

    @classmethod
    def load_from_pretrained(
        cls, path: str, device: str = "cpu", dtype: str = "float32"
    ) -> "SAE":

        config_path = os.path.join(path, "cfg.json")
        weight_path = os.path.join(path, "sae_weights.safetensors")

        cfg_dict, state_dict, _ = load_pretrained_sae_lens_sae_components(
            config_path, weight_path, device, dtype
        )

        sae_cfg = SAEConfig.from_dict(cfg_dict)

        sae = cls(sae_cfg)
        sae.load_state_dict(state_dict)

        return sae

    @classmethod
    def from_pretrained(
        cls,
        release: str,
        sae_id: str,
        device: str = "cpu",
    ) -> Tuple["SAE", dict[str, Any], Optional[torch.Tensor]]:
        """

        Load a pretrained SAE from the Hugging Face model hub.

        Args:
            release: The release name. This will be mapped to a huggingface repo id based on the pretrained_saes.yaml file.
            id: The id of the SAE to load. This will be mapped to a path in the huggingface repo.
            device: The device to load the SAE on.
            return_sparsity_if_present: If True, will return the log sparsity tensor if it is present in the model directory in the Hugging Face model hub.
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

        cfg_dict, state_dict, log_sparsities = conversion_loader(
            repo_id=hf_repo_id,
            folder_name=hf_path,
            device=device,
            force_download=False,
        )

        sae = cls(SAEConfig.from_dict(cfg_dict))
        sae.load_state_dict(state_dict)

        return sae, cfg_dict, log_sparsities

    def get_name(self):
        sae_name = f"sae_{self.cfg.model_name}_{self.cfg.hook_name}_{self.cfg.d_sae}"
        return sae_name

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAE":
        return cls(SAEConfig.from_dict(config_dict))

    def turn_on_forward_pass_hook_z_reshaping(self):

        assert self.cfg.hook_name.endswith(
            "_z"
        ), "This method should only be called for hook_z SAEs."

        def reshape_fn_in(x: torch.Tensor):
            self.d_head = x.shape[-1]  # type: ignore
            self.reshape_fn_in = lambda x: einops.rearrange(
                x, "... n_heads d_head -> ... (n_heads d_head)"
            )
            return einops.rearrange(x, "... n_heads d_head -> ... (n_heads d_head)")

        self.reshape_fn_in = reshape_fn_in

        self.reshape_fn_out = lambda x, d_head: einops.rearrange(
            x, "... (n_heads d_head) -> ... n_heads d_head", d_head=d_head
        )
        self.hook_z_reshaping_mode = True

    def turn_off_forward_pass_hook_z_reshaping(self):
        self.reshape_fn_in = lambda x: x
        self.reshape_fn_out = lambda x, d_head: x
        self.d_head = None
        self.hook_z_reshaping_mode = False


def get_activation_fn(activation_fn: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation_fn == "relu":
        return torch.nn.ReLU()
    elif activation_fn == "tanh-relu":

        def tanh_relu(input: torch.Tensor) -> torch.Tensor:
            input = torch.relu(input)
            input = torch.tanh(input)
            return input

        return tanh_relu
    else:
        raise ValueError(f"Unknown activation function: {activation_fn}")
