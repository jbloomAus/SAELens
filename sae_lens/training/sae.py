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

from sae_lens.config import DTYPE_MAP, LanguageModelSAERunnerConfig
from sae_lens.toolkit.pretrained_sae_loaders import (
    NAMED_PRETRAINED_SAE_LOADERS,
    load_pretrained_sae_lens_sae_components,
)
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"


@dataclass
class TrainStepOutput:
    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    loss: torch.Tensor  # we need to call backwards on this
    mse_loss: float
    l1_loss: float
    ghost_grad_loss: float


@dataclass
class SAEConfig:

    # forward pass details.
    d_in: int
    d_sae: int
    activation_fn_str: str
    apply_b_dec_to_input: bool
    uses_scaling_factor: bool

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
        rename_dict = {
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
            "uses_scaling_factor": self.uses_scaling_factor,
            "sae_lens_training_version": self.sae_lens_training_version,
            "prepend_bos": self.prepend_bos,
            "dataset_path": self.dataset_path,
            "context_size": self.context_size,
            "normalize_activations": self.normalize_activations,
        }


@dataclass
class TrainingSAEConfig(SAEConfig):

    # Sparsity Loss Calculations
    l1_coefficient: float
    lp_norm: float
    use_ghost_grads: bool
    normalize_sae_decoder: bool
    noise_scale: float
    decoder_orthogonal_init: bool
    mse_loss_normalization: Optional[str]
    decoder_heuristic_init: bool = False
    init_encoder_as_decoder_transpose: bool = False
    scale_sparsity_penalty_by_decoder_norm: bool = False

    @classmethod
    def from_sae_runner_config(
        cls, cfg: LanguageModelSAERunnerConfig
    ) -> "TrainingSAEConfig":

        return cls(
            # base confg
            d_in=cfg.d_in,
            d_sae=cfg.d_sae,  # type: ignore
            dtype=cfg.dtype,
            device=cfg.device,
            model_name=cfg.model_name,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            hook_head_index=cfg.hook_head_index,
            activation_fn_str=cfg.activation_fn,
            apply_b_dec_to_input=cfg.apply_b_dec_to_input,
            uses_scaling_factor=cfg.finetuning_method is not None,
            sae_lens_training_version=cfg.sae_lens_training_version,
            context_size=cfg.context_size,
            dataset_path=cfg.dataset_path,
            prepend_bos=cfg.prepend_bos,
            # Training cfg
            l1_coefficient=cfg.l1_coefficient,
            lp_norm=cfg.lp_norm,
            use_ghost_grads=cfg.use_ghost_grads,
            normalize_sae_decoder=cfg.normalize_sae_decoder,
            noise_scale=cfg.noise_scale,
            decoder_orthogonal_init=cfg.decoder_orthogonal_init,
            mse_loss_normalization=cfg.mse_loss_normalization,
            decoder_heuristic_init=cfg.decoder_heuristic_init,
            init_encoder_as_decoder_transpose=cfg.init_encoder_as_decoder_transpose,
            scale_sparsity_penalty_by_decoder_norm=cfg.scale_sparsity_penalty_by_decoder_norm,
            normalize_activations=cfg.normalize_activations,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAEConfig":
        return TrainingSAEConfig(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "l1_coefficient": self.l1_coefficient,
            "lp_norm": self.lp_norm,
            "use_ghost_grads": self.use_ghost_grads,
            "normalize_sae_decoder": self.normalize_sae_decoder,
            "noise_scale": self.noise_scale,
            "decoder_orthogonal_init": self.decoder_orthogonal_init,
            "init_encoder_as_decoder_transpose": self.init_encoder_as_decoder_transpose,
            "mse_loss_normalization": self.mse_loss_normalization,
            "decoder_heuristic_init": self.decoder_heuristic_init,
            "scale_sparsity_penalty_by_decoder_norm": self.scale_sparsity_penalty_by_decoder_norm,
            "normalize_activations": self.normalize_activations,
        }

    # this needs to exist so we can initialize the parent sae cfg without the training specific
    # parameters. Maybe there's a cleaner way to do this
    def get_base_sae_cfg_dict(self) -> dict[str, Any]:
        return {
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "activation_fn_str": self.activation_fn_str,
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "dtype": self.dtype,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "hook_head_index": self.hook_head_index,
            "device": self.device,
            "context_size": self.context_size,
            "prepend_bos": self.prepend_bos,
            "uses_scaling_factor": self.uses_scaling_factor,
            "normalize_activations": self.normalize_activations,
            "dataset_path": self.dataset_path,
            "sae_lens_training_version": self.sae_lens_training_version,
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
        if self.cfg.uses_scaling_factor:
            self.apply_scaling_factor = lambda x: x * self.scaling_factor
        else:
            self.apply_scaling_factor = lambda x: x

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
        self.reshape_fn_in = lambda x: x
        self.reshape_fn_out = lambda x, d_head: x
        self.d_head = None
        if self.cfg.hook_name.endswith("_z"):

            def reshape_fn_in(x: torch.Tensor):
                self.d_head = x.shape[-1]  # type: ignore
                self.reshape_fn_in = lambda x: einops.rearrange(
                    x, "... n_heads d_head -> ... (n_heads d_head)"
                )
                return einops.rearrange(x, "... n_heads d_head -> ... (n_heads d_head)")

            self.reshape_fn_in = reshape_fn_in

        if self.cfg.hook_name.endswith("_z"):
            self.reshape_fn_out = lambda x, d_head: einops.rearrange(
                x, "... (n_heads d_head) -> ... n_heads d_head", d_head=d_head
            )

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
        if self.cfg.uses_scaling_factor:
            self.scaling_factor = nn.Parameter(
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
                    self.apply_scaling_factor(feature_acts) @ self.W_dec + self.b_dec,
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
            self.apply_scaling_factor(feature_acts) @ self.W_dec + self.b_dec
        )

        # handle hook z reshaping if needed.
        sae_out = self.reshape_fn_out(sae_out, self.d_head)  # type: ignore

        return sae_out

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

        cfg_dict, state_dict = load_pretrained_sae_lens_sae_components(
            config_path, weight_path, device, dtype
        )

        sae_cfg = SAEConfig.from_dict(cfg_dict)

        sae = cls(sae_cfg)
        sae.load_state_dict(state_dict)

        return sae

    @classmethod
    def from_pretrained(
        cls, release: str, sae_id: str, device: str = "cpu"
    ) -> Tuple["SAE", dict[str, Any]]:
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

        cfg_dict, state_dict = conversion_loader(
            repo_id=hf_repo_id,
            folder_name=hf_path,
            device=device,
            force_download=False,
        )

        if "prepend_bos" not in cfg_dict:
            # default to True for backwards compatibility
            cfg_dict["prepend_bos"] = True

        if "apply_b_dec_to_input" not in cfg_dict:
            # default to False for backwards compatibility
            cfg_dict["apply_b_dec_to_input"] = False

        if "uses_scaling_factor" not in cfg_dict:
            # default to False for backwards compatibility
            cfg_dict["uses_scaling_factor"] = False

        if "sae_lens_training_version" not in cfg_dict:
            cfg_dict["sae_lens_training_version"] = None

        if "activation_fn" not in cfg_dict:
            cfg_dict["activation_fn_str"] = "relu"

        if "normalize_activations" not in cfg_dict:
            cfg_dict["normalize_activations"] = False

        if "scaling_factor" in state_dict:
            # we were adding it anyway for a period of time but are no longer doing so.
            # so we should delete it if
            if torch.allclose(
                state_dict["scaling_factor"],
                torch.ones_like(state_dict["scaling_factor"]),
            ):
                del state_dict["scaling_factor"]
                cfg_dict["uses_scaling_factor"] = False
            else:
                assert cfg_dict[
                    "uses_scaling_factor"
                ], "Scaling factor is present but uses_scaling_factor is False."
        else:
            # it's there and it's not all 1's, we should use it.
            cfg_dict["uses_scaling_factor"] = False

        sae = cls(SAEConfig.from_dict(cfg_dict))
        sae.load_state_dict(state_dict)

        return sae, cfg_dict

    def get_name(self):
        sae_name = f"sae_{self.cfg.model_name}_{self.cfg.hook_name}_{self.cfg.d_sae}"
        return sae_name

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAE":
        return cls(SAEConfig.from_dict(config_dict))


class TrainingSAE(SAE):

    cfg: TrainingSAEConfig  # type: ignore
    use_error_term: bool
    dtype: torch.dtype
    device: torch.device

    def __init__(self, cfg: TrainingSAEConfig, use_error_term: bool = False):

        base_sae_cfg = SAEConfig.from_dict(cfg.get_base_sae_cfg_dict())
        super().__init__(base_sae_cfg)
        self.cfg = cfg  # type: ignore
        self.use_error_term = use_error_term

        self.initialize_weights_complex()

        # The training SAE will assume that the activation store handles
        # reshaping.
        self.reshape_fn_in = lambda x: x
        self.reshape_fn_out = lambda x, d_head: x

        self.mse_loss_fn = self._get_mse_loss_fn()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAE":
        return cls(TrainingSAEConfig.from_dict(config_dict))

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        feature_acts, _ = self.encode_with_hidden_pre(x)
        return feature_acts

    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:

        # move x to correct dtype
        x = x.to(self.dtype)

        # handle hook z reshaping if needed.
        x = self.reshape_fn_in(x)  # type: ignore

        # apply b_dec_to_input if using that method.
        sae_in = self.hook_sae_input(x - (self.b_dec * self.cfg.apply_b_dec_to_input))

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        hidden_pre_noised = hidden_pre + (
            torch.randn_like(hidden_pre) * self.cfg.noise_scale * self.training
        )
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre_noised))

        return feature_acts, hidden_pre_noised

    def forward(
        self,
        x: Float[torch.Tensor, "... d_in"],
    ) -> Float[torch.Tensor, "... d_in"]:

        feature_acts, _ = self.encode_with_hidden_pre(x)
        sae_out = self.decode(feature_acts)

        return sae_out

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: Optional[torch.Tensor] = None,
    ) -> TrainStepOutput:  # type: ignore

        # do a forward pass to get SAE out, but we also need the
        # hidden pre.
        feature_acts, _ = self.encode_with_hidden_pre(sae_in)
        sae_out = self.decode(feature_acts)

        # MSE LOSS
        per_item_mse_loss = self.mse_loss_fn(sae_out, sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        # GHOST GRADS
        if self.cfg.use_ghost_grads and self.training and dead_neuron_mask is not None:

            # first half of second forward pass
            _, hidden_pre = self.encode_with_hidden_pre(sae_in)
            ghost_grad_loss = self.calculate_ghost_grad_loss(
                x=sae_in,
                sae_out=sae_out,
                per_item_mse_loss=per_item_mse_loss,
                hidden_pre=hidden_pre,
                dead_neuron_mask=dead_neuron_mask,
            )
        else:
            ghost_grad_loss = 0.0

        # SPARSITY LOSS
        # either the W_dec norms are 1 and this won't do anything or they are not 1
        # and we're using their norm in the loss function.
        weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
        sparsity = weighted_feature_acts.norm(
            p=self.cfg.lp_norm, dim=-1
        )  # sum over the feature dimension

        l1_loss = (current_l1_coefficient * sparsity).mean()

        loss = mse_loss + l1_loss + ghost_grad_loss

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            loss=loss,
            mse_loss=mse_loss.item(),
            l1_loss=l1_loss.item(),
            ghost_grad_loss=(
                ghost_grad_loss.item()
                if isinstance(ghost_grad_loss, torch.Tensor)
                else ghost_grad_loss
            ),
        )

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
        # ghost grads use an exponentional activation function, ignoring whatever
        # the activation function is in the SAE. The forward pass uses the dead neurons only.
        feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
        ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
        l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
        norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
        ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

        # 3. There is some fairly complex rescaling here to make sure that the loss
        # is comparable to the original loss. This is because the ghost grads are
        # only calculated for the dead neurons, so we need to rescale the loss to
        # make sure that the loss is comparable to the original loss.
        # There have been methodological improvements that are not implemented here yet
        # see here: https://www.lesswrong.com/posts/C5KAZQib3bzzpeyrg/full-post-progress-update-1-from-the-gdm-mech-interp-team#Improving_ghost_grads
        per_item_mse_loss_ghost_resid = self.mse_loss_fn(ghost_out, residual.detach())
        mse_rescaling_factor = (
            per_item_mse_loss / (per_item_mse_loss_ghost_resid + 1e-6)
        ).detach()
        per_item_mse_loss_ghost_resid = (
            mse_rescaling_factor * per_item_mse_loss_ghost_resid
        )

        return per_item_mse_loss_ghost_resid.mean()

    @torch.no_grad()
    def _get_mse_loss_fn(self) -> Any:

        def standard_mse_loss_fn(
            preds: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            return torch.nn.functional.mse_loss(preds, target, reduction="none")

        def batch_norm_mse_loss_fn(
            preds: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            target_centered = target - target.mean(dim=0, keepdim=True)
            normalization = target_centered.norm(dim=-1, keepdim=True)
            return torch.nn.functional.mse_loss(preds, target, reduction="none") / (
                normalization + 1e-6
            )

        if self.cfg.mse_loss_normalization == "dense_batch":
            return batch_norm_mse_loss_fn
        else:
            return standard_mse_loss_fn

    @classmethod
    def load_from_pretrained(  # type: ignore
        cls,
        path: str,
        device: str = "cpu",
        dtype: str = "float32",
    ) -> "TrainingSAE":

        config_path = os.path.join(path, "cfg.json")
        weight_path = os.path.join(path, "sae_weights.safetensors")

        cfg_dict, state_dict = load_pretrained_sae_lens_sae_components(
            config_path, weight_path, device, dtype
        )

        sae_cfg = TrainingSAEConfig.from_dict(cfg_dict)

        sae = cls(sae_cfg)
        sae.load_state_dict(state_dict)

        return sae

    def initialize_weights_complex(self):
        """ """

        if self.cfg.decoder_orthogonal_init:
            self.W_dec.data = nn.init.orthogonal_(self.W_dec.data.T).T

        elif self.cfg.decoder_heuristic_init:
            self.W_dec = nn.Parameter(
                torch.rand(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
            self.initialize_decoder_norm_constant_norm()

        elif self.cfg.normalize_sae_decoder:
            self.set_decoder_norm_to_unit_norm()

        # Then we intialize the encoder weights (either as the transpose of decoder or not)
        if self.cfg.init_encoder_as_decoder_transpose:
            self.W_enc.data = self.W_dec.data.T.clone().contiguous()
        else:
            self.W_enc = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(
                        self.cfg.d_in,
                        self.cfg.d_sae,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
            )

        if self.cfg.normalize_sae_decoder:
            with torch.no_grad():
                # Anthropic normalize this to have unit columns
                self.set_decoder_norm_to_unit_norm()

    ## Initialization Methods
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

    ## Training Utils
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
        self.W_dec.data *= norm  # will break tests but do this for now.

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
