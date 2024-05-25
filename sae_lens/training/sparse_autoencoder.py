"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import json
import os
from typing import Any, Callable, Optional, Tuple

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
from sae_lens.training.config import DTYPE_MAP, LanguageModelSAERunnerConfig

SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"


class SparseAutoencoderBase(HookedRootModule):
    """ """

    # forward pass details.
    d_in: int
    d_sae: int
    activation_fn_str: str
    activation_fn: Callable[[torch.Tensor], torch.Tensor]
    apply_b_dec_to_input: bool
    uses_scaling_factor: bool

    # dataset it was trained on details.
    context_size: int
    model_name: str
    hook_point: str
    hook_point_layer: int
    hook_point_head_index: Optional[int]
    prepend_bos: bool
    dataset_path: str

    # misc
    dtype: torch.dtype
    device: str | torch.device
    sae_lens_training_version: Optional[str]

    # analysis
    use_error_term = False

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        dtype: str | torch.dtype,
        device: str | torch.device,
        model_name: str,
        hook_point: str,
        hook_point_layer: int,
        hook_point_head_index: Optional[int] = None,
        activation_fn: str = "relu",
        apply_b_dec_to_input: bool = True,
        uses_scaling_factor: bool = False,
        sae_lens_training_version: Optional[str] = None,
        prepend_bos: bool = True,
        dataset_path: str = "unknown",
        context_size: int = 256,
    ):
        super().__init__()

        self.d_in = d_in
        self.d_sae = d_sae  # type: ignore
        self.activation_fn_str = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.apply_b_dec_to_input = apply_b_dec_to_input
        self.uses_scaling_factor = uses_scaling_factor

        self.model_name = model_name
        self.hook_point = hook_point
        self.hook_point_layer = hook_point_layer
        self.hook_point_head_index = hook_point_head_index
        self.dataset_path = dataset_path
        self.prepend_bos = prepend_bos
        self.context_size = context_size

        self.dtype = dtype if isinstance(dtype, torch.dtype) else DTYPE_MAP[dtype]
        self.device = device
        self.sae_lens_training_version = sae_lens_training_version

        self.initialize_weights_basic()

        # handle presence / absence of scaling factor.
        if self.uses_scaling_factor:
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
        if self.hook_point.endswith("_z"):

            def reshape_fn_in(x: torch.Tensor):
                self.d_head = x.shape[-1]  # type: ignore
                self.reshape_fn_in = lambda x: einops.rearrange(
                    x, "... n_heads d_head -> ... (n_heads d_head)"
                )
                return einops.rearrange(x, "... n_heads d_head -> ... (n_heads d_head)")

            self.reshape_fn_in = reshape_fn_in

        if self.hook_point.endswith("_z"):
            self.reshape_fn_out = lambda x, d_head: einops.rearrange(
                x, "... (n_heads d_head) -> ... n_heads d_head", d_head=d_head
            )

        self.setup()  # Required for `HookedRootModule`s

    def initialize_weights_basic(self):

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

        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device)
            )
        )

        # methdods which change b_dec as a function of the dataset are implemented after init.
        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
        )

        # scaling factor for fine-tuning (not to be used in initial training)
        # TODO: Make this optional and not included with all SAEs by default (but maintain backwards compatibility)
        if self.uses_scaling_factor:
            self.scaling_factor = nn.Parameter(
                torch.ones(self.d_sae, dtype=self.dtype, device=self.device)
            )

    # Basic Forward Pass Functionality.
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        return sae_out

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:

        # move x to correct dtype
        x = x.to(self.dtype)

        # handle hook z reshaping if needed.
        x = self.reshape_fn_in(x)  # type: ignore

        # apply b_dec_to_input if using that method.
        sae_in = self.hook_sae_input(x - (self.b_dec * self.apply_b_dec_to_input))

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))

        return feature_acts

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """Decodes SAE feature activation tensor into a reconstructed input activation tensor."""
        # "... d_sae, d_sae d_in -> ... d_in",
        sae_out = self.hook_sae_output(
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
        config = self.get_config_dict()

        with open(f"{path}/{SAE_CFG_PATH}", "w") as f:
            json.dump(config, f)

        if sparsity is not None:
            sparsity_in_dict = {"sparsity": sparsity}
            save_file(sparsity_in_dict, f"{path}/{SPARSITY_PATH}")  # type: ignore

    @classmethod
    def load_from_pretrained(
        cls, path: str, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> "SparseAutoencoderBase":

        config_path = os.path.join(path, "cfg.json")
        weight_path = os.path.join(path, "sae_weights.safetensors")

        cfg_dict, state_dict = load_pretrained_sae_lens_sae_components(
            config_path, weight_path, device, dtype
        )

        sae = cls(
            d_in=cfg_dict["d_in"],
            d_sae=cfg_dict["d_sae"],
            dtype=cfg_dict["dtype"],
            device=cfg_dict["device"],
            model_name=cfg_dict["model_name"],
            hook_point=cfg_dict["hook_point"],
            hook_point_layer=cfg_dict["hook_point_layer"],
            hook_point_head_index=cfg_dict["hook_point_head_index"],
            activation_fn=cfg_dict["activation_fn"],
            apply_b_dec_to_input=cfg_dict["apply_b_dec_to_input"],
            uses_scaling_factor=cfg_dict["uses_scaling_factor"],
            prepend_bos=cfg_dict["prepend_bos"],
            dataset_path=cfg_dict["dataset_path"],
            context_size=cfg_dict["context_size"],
        )

        sae.load_state_dict(state_dict)

        return sae

    @classmethod
    def from_pretrained(
        cls, release: str, sae_id: str, device: str = "cpu"
    ) -> Tuple["SparseAutoencoderBase", dict[str, Any]]:
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

        sae = cls(
            d_in=cfg_dict["d_in"],
            d_sae=cfg_dict["d_sae"],
            dtype=cfg_dict["dtype"],
            device=cfg_dict["device"],
            model_name=cfg_dict["model_name"],
            hook_point=cfg_dict["hook_point"],
            hook_point_layer=cfg_dict["hook_point_layer"],
            hook_point_head_index=cfg_dict["hook_point_head_index"],
            activation_fn=(
                cfg_dict["activation_fn"] if "activation_fn" in cfg_dict else "relu"
            ),
            context_size=cfg_dict["context_size"],
            dataset_path=cfg_dict["dataset_path"],
            prepend_bos=cfg_dict["prepend_bos"],
        )
        sae.load_state_dict(state_dict)

        return sae, cfg_dict

    def get_name(self):
        sae_name = (
            f"sparse_autoencoder_{self.model_name}_{self.hook_point}_{self.d_sae}"
        )
        return sae_name

    def get_config_dict(self):
        return {
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "dtype": str(self.dtype),
            "device": str(self.device),
            "model_name": self.model_name,
            "hook_point": self.hook_point,
            "hook_point_layer": self.hook_point_layer,
            "hook_point_head_index": self.hook_point_head_index,
            "activation_fn": self.activation_fn_str,  # use string for serialization
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "uses_scaling_factor": self.uses_scaling_factor,
            "sae_lens_training_version": self.sae_lens_training_version,
            "prepend_bos": self.prepend_bos,
            "dataset_path": self.dataset_path,
            "context_size": self.context_size,
        }


class TrainingSparseAutoencoder(SparseAutoencoderBase):

    l1_coefficient: float
    lp_norm: float
    use_ghost_grads: bool
    normalize_sae_decoder: bool
    noise_scale: float
    decoder_orthogonal_init: bool
    mse_loss_normalization: Optional[str]

    def __init__(self, cfg: LanguageModelSAERunnerConfig):

        super().__init__(
            d_in=cfg.d_in,
            d_sae=cfg.d_sae,  # type: ignore
            dtype=cfg.dtype,
            device=cfg.device,
            model_name=cfg.model_name,
            hook_point=cfg.hook_point,
            hook_point_layer=cfg.hook_point_layer,
            hook_point_head_index=cfg.hook_point_head_index,
            activation_fn=cfg.activation_fn,
            apply_b_dec_to_input=cfg.apply_b_dec_to_input,
            uses_scaling_factor=cfg.finetuning_method is not None,
            sae_lens_training_version=cfg.sae_lens_training_version,
        )

        self.mse_loss_normalization = cfg.mse_loss_normalization
        self.l1_coefficient = cfg.l1_coefficient
        self.lp_norm = cfg.lp_norm
        self.scale_sparsity_penalty_by_decoder_norm = (
            cfg.scale_sparsity_penalty_by_decoder_norm
        )
        self.use_ghost_grads = cfg.use_ghost_grads
        self.noise_scale = cfg.noise_scale

        self.normalize_sae_decoder = cfg.normalize_sae_decoder
        self.decoder_orthogonal_init = cfg.decoder_orthogonal_init
        self.decoder_heuristic_init = cfg.decoder_heuristic_init
        self.init_encoder_as_decoder_transpose = cfg.init_encoder_as_decoder_transpose

        self.initialize_weights_complex()

        # The training SAE will assume that the activation store handles
        # reshaping.
        self.reshape_fn_in = lambda x: x
        self.reshape_fn_out = lambda x, d_head: x

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
        sae_in = self.hook_sae_input(x - (self.b_dec * self.apply_b_dec_to_input))

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        hidden_pre_noised = hidden_pre + (
            torch.randn_like(hidden_pre) * self.noise_scale * self.training
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

    @classmethod
    def load_from_pretrained(  # type: ignore
        cls, path: str, cfg: LanguageModelSAERunnerConfig
    ) -> "TrainingSparseAutoencoder":

        base_sae = super().load_from_pretrained(
            path=path,
            device=cfg.device,  # type: ignore
            dtype=cfg.dtype,
        )

        sae = cls(cfg)
        sae.load_state_dict(base_sae.state_dict())
        return sae

    def initialize_weights_complex(self):
        """ """

        if self.decoder_orthogonal_init:
            self.W_dec.data = nn.init.orthogonal_(self.W_dec.data.T).T

        elif self.decoder_heuristic_init:
            self.W_dec = nn.Parameter(
                torch.rand(self.d_sae, self.d_in, dtype=self.dtype, device=self.device)
            )
            self.initialize_decoder_norm_constant_norm()

        elif self.normalize_sae_decoder:
            self.set_decoder_norm_to_unit_norm()

        # Then we intialize the encoder weights (either as the transpose of decoder or not)
        if self.init_encoder_as_decoder_transpose:
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
