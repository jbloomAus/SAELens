"""Base classes for Sparse Autoencoders (SAEs)."""

import json
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Type, TypeVar

import einops
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from safetensors.torch import save_file
from torch import nn
from transformer_lens.hook_points import HookedRootModule, HookPoint
from typing_extensions import deprecated, overload

from sae_lens import logger
from sae_lens.config import (
    DTYPE_MAP,
    SAE_CFG_FILENAME,
    SAE_WEIGHTS_FILENAME,
    SPARSITY_FILENAME,
    LanguageModelSAERunnerConfig,
)
from sae_lens.loading.pretrained_sae_loaders import (
    NAMED_PRETRAINED_SAE_LOADERS,
    PretrainedSaeDiskLoader,
    PretrainedSaeHuggingfaceLoader,
    get_conversion_loader_name,
    handle_config_defaulting,
    sae_lens_disk_loader,
)
from sae_lens.loading.pretrained_saes_directory import (
    get_config_overrides,
    get_norm_scaling_factor,
    get_pretrained_saes_directory,
    get_repo_id_and_folder_name,
)
from sae_lens.regsitry import get_sae_class, get_sae_training_class

T = TypeVar("T", bound="SAE")


@dataclass
class SAEConfig:
    """Base configuration for SAE models."""

    architecture: str
    d_in: int
    d_sae: int
    dtype: str
    device: str
    model_name: str
    hook_name: str
    hook_layer: int
    hook_head_index: int | None
    activation_fn: str
    activation_fn_kwargs: dict[str, Any]
    apply_b_dec_to_input: bool
    finetuning_scaling_factor: bool
    normalize_activations: str
    context_size: int
    dataset_path: str
    dataset_trust_remote_code: bool
    sae_lens_training_version: str
    model_from_pretrained_kwargs: dict[str, Any] = field(default_factory=dict)
    seqpos_slice: tuple[int, ...] | None = None
    prepend_bos: bool = False
    neuronpedia_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SAEConfig":
        valid_field_names = {field.name for field in fields(cls)}
        valid_config_dict = {
            key: val for key, val in config_dict.items() if key in valid_field_names
        }

        # Ensure seqpos_slice is a tuple
        if (
            "seqpos_slice" in valid_config_dict
            and valid_config_dict["seqpos_slice"] is not None
            and isinstance(valid_config_dict["seqpos_slice"], list)
        ):
            valid_config_dict["seqpos_slice"] = tuple(valid_config_dict["seqpos_slice"])

        return cls(**valid_config_dict)


@dataclass
class TrainStepOutput:
    """Output from a training step."""

    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    hidden_pre: torch.Tensor
    loss: torch.Tensor  # we need to call backwards on this
    losses: dict[str, torch.Tensor]


@dataclass
class TrainStepInput:
    """Input to a training step."""

    sae_in: torch.Tensor
    current_l1_coefficient: float
    dead_neuron_mask: torch.Tensor | None


class SAE(HookedRootModule, ABC):
    """Abstract base class for all SAE architectures."""

    cfg: SAEConfig
    dtype: torch.dtype
    device: torch.device
    use_error_term: bool

    # For type checking only - don't provide default values
    # These will be initialized by subclasses
    W_enc: nn.Parameter
    W_dec: nn.Parameter
    b_dec: nn.Parameter

    def __init__(self, cfg: SAEConfig, use_error_term: bool = False):
        """Initialize the SAE."""
        super().__init__()

        self.cfg = cfg

        if cfg.model_from_pretrained_kwargs:
            warnings.warn(
                "\nThis SAE has non-empty model_from_pretrained_kwargs. "
                "\nFor optimal performance, load the model like so:\n"
                "model = HookedSAETransformer.from_pretrained_no_processing(..., **cfg.model_from_pretrained_kwargs)",
                category=UserWarning,
                stacklevel=1,
            )

        self.dtype = DTYPE_MAP[cfg.dtype]
        self.device = torch.device(cfg.device)
        self.use_error_term = use_error_term

        # Set up activation function
        self.activation_fn = self._get_activation_fn()

        # Initialize weights
        self.initialize_weights()

        # Handle presence / absence of scaling factor
        if self.cfg.finetuning_scaling_factor:
            self.apply_finetuning_scaling_factor = (
                lambda x: x * self.finetuning_scaling_factor
            )
        else:
            self.apply_finetuning_scaling_factor = lambda x: x

        # Set up hooks
        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint()
        self.hook_sae_output = HookPoint()
        self.hook_sae_recons = HookPoint()
        self.hook_sae_error = HookPoint()

        # handle hook_z reshaping if needed.
        if self.cfg.hook_name.endswith("_z"):
            # print(f"Setting up hook_z reshaping for {self.cfg.hook_name}")
            self.turn_on_forward_pass_hook_z_reshaping()
        else:
            # print(f"No hook_z reshaping needed for {self.cfg.hook_name}")
            self.turn_off_forward_pass_hook_z_reshaping()

        # Set up activation normalization
        self._setup_activation_normalization()

        self.setup()  # Required for HookedRootModule

    @torch.no_grad()
    def fold_activation_norm_scaling_factor(self, scaling_factor: float):
        self.W_enc.data *= scaling_factor  # type: ignore
        self.W_dec.data /= scaling_factor  # type: ignore
        self.b_dec.data /= scaling_factor  # type: ignore
        self.cfg.normalize_activations = "none"

    def _get_activation_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get the activation function specified in config."""
        return self._get_activation_fn_static(
            self.cfg.activation_fn, **(self.cfg.activation_fn_kwargs or {})
        )

    @staticmethod
    def _get_activation_fn_static(
        activation_fn: str, **kwargs: Any
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get the activation function from a string specification."""
        if activation_fn == "relu":
            return torch.nn.ReLU()
        if activation_fn == "tanh-relu":

            def tanh_relu(input: torch.Tensor) -> torch.Tensor:
                input = torch.relu(input)
                return torch.tanh(input)

            return tanh_relu
        if activation_fn == "topk":
            if "k" not in kwargs:
                raise ValueError("TopK activation function requires a k value.")
            k = kwargs.get("k", 1)  # Default k to 1 if not provided

            def topk_fn(x: torch.Tensor) -> torch.Tensor:
                topk = torch.topk(x.flatten(start_dim=-1), k=k, dim=-1)
                values = torch.relu(topk.values)
                result = torch.zeros_like(x.flatten(start_dim=-1))
                result.scatter_(-1, topk.indices, values)
                return result.view_as(x)

            return topk_fn
        raise ValueError(f"Unknown activation function: {activation_fn}")

    def _setup_activation_normalization(self):
        """Set up activation normalization functions based on config."""
        if self.cfg.normalize_activations == "constant_norm_rescale":

            def run_time_activation_norm_fn_in(x: torch.Tensor) -> torch.Tensor:
                self.x_norm_coeff = (self.cfg.d_in**0.5) / x.norm(dim=-1, keepdim=True)
                return x * self.x_norm_coeff

            def run_time_activation_norm_fn_out(x: torch.Tensor) -> torch.Tensor:
                x = x / self.x_norm_coeff  # type: ignore
                del self.x_norm_coeff
                return x

            self.run_time_activation_norm_fn_in = run_time_activation_norm_fn_in
            self.run_time_activation_norm_fn_out = run_time_activation_norm_fn_out

        elif self.cfg.normalize_activations == "layer_norm":

            def run_time_activation_ln_in(
                x: torch.Tensor, eps: float = 1e-5
            ) -> torch.Tensor:
                mu = x.mean(dim=-1, keepdim=True)
                x = x - mu
                std = x.std(dim=-1, keepdim=True)
                x = x / (std + eps)
                self.ln_mu = mu
                self.ln_std = std
                return x

            def run_time_activation_ln_out(
                x: torch.Tensor,
                eps: float = 1e-5,  # noqa: ARG001
            ) -> torch.Tensor:
                return x * self.ln_std + self.ln_mu  # type: ignore

            self.run_time_activation_norm_fn_in = run_time_activation_ln_in
            self.run_time_activation_norm_fn_out = run_time_activation_ln_out
        else:
            self.run_time_activation_norm_fn_in = lambda x: x
            self.run_time_activation_norm_fn_out = lambda x: x

    @abstractmethod
    def initialize_weights(self):
        """Initialize model weights."""
        pass

    @abstractmethod
    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """Encode input tensor to feature space."""
        pass

    @abstractmethod
    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """Decode feature activations back to input space."""
        pass

    def turn_on_forward_pass_hook_z_reshaping(self):
        if not self.cfg.hook_name.endswith("_z"):
            raise ValueError("This method should only be called for hook_z SAEs.")

        # print(f"Turning on hook_z reshaping for {self.cfg.hook_name}")

        def reshape_fn_in(x: torch.Tensor):
            # print(f"reshape_fn_in input shape: {x.shape}")
            self.d_head = x.shape[-1]
            # print(f"Setting d_head to: {self.d_head}")
            self.reshape_fn_in = lambda x: einops.rearrange(
                x, "... n_heads d_head -> ... (n_heads d_head)"
            )
            return einops.rearrange(x, "... n_heads d_head -> ... (n_heads d_head)")

        self.reshape_fn_in = reshape_fn_in
        self.reshape_fn_out = lambda x, d_head: einops.rearrange(
            x, "... (n_heads d_head) -> ... n_heads d_head", d_head=d_head
        )
        self.hook_z_reshaping_mode = True
        # print(f"hook_z reshaping turned on, self.d_head={getattr(self, 'd_head', None)}")

    def turn_off_forward_pass_hook_z_reshaping(self):
        self.reshape_fn_in = lambda x: x
        self.reshape_fn_out = lambda x, d_head: x  # noqa: ARG005
        self.d_head = None
        self.hook_z_reshaping_mode = False

    @overload
    def to(
        self: T,
        device: torch.device | str | None = ...,
        dtype: torch.dtype | None = ...,
        non_blocking: bool = ...,
    ) -> T: ...

    @overload
    def to(self: T, dtype: torch.dtype, non_blocking: bool = ...) -> T: ...

    @overload
    def to(self: T, tensor: torch.Tensor, non_blocking: bool = ...) -> T: ...

    def to(self: T, *args: Any, **kwargs: Any) -> T:  # type: ignore
        device_arg = None
        dtype_arg = None

        # Check args
        for arg in args:
            if isinstance(arg, (torch.device, str)):
                device_arg = arg
            elif isinstance(arg, torch.dtype):
                dtype_arg = arg
            elif isinstance(arg, torch.Tensor):
                device_arg = arg.device
                dtype_arg = arg.dtype

        # Check kwargs
        device_arg = kwargs.get("device", device_arg)
        dtype_arg = kwargs.get("dtype", dtype_arg)

        # Update device in config if provided
        if device_arg is not None:
            # Convert device to torch.device if it's a string
            device = (
                torch.device(device_arg) if isinstance(device_arg, str) else device_arg
            )

            # Update the cfg.device
            self.cfg.device = str(device)

            # Update the device property
            self.device = device

        # Update dtype in config if provided
        if dtype_arg is not None:
            # Update the cfg.dtype
            self.cfg.dtype = str(dtype_arg)

            # Update the dtype property
            self.dtype = dtype_arg

        return super().to(*args, **kwargs)

    def process_sae_in(
        self, sae_in: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_in"]:
        # print(f"Input shape to process_sae_in: {sae_in.shape}")
        # print(f"self.cfg.hook_name: {self.cfg.hook_name}")
        # print(f"self.b_dec shape: {self.b_dec.shape}")
        # print(f"Hook z reshaping mode: {getattr(self, 'hook_z_reshaping_mode', False)}")

        sae_in = sae_in.to(self.dtype)

        # print(f"Shape before reshape_fn_in: {sae_in.shape}")
        sae_in = self.reshape_fn_in(sae_in)
        # print(f"Shape after reshape_fn_in: {sae_in.shape}")

        sae_in = self.hook_sae_input(sae_in)
        sae_in = self.run_time_activation_norm_fn_in(sae_in)

        # Here's where the error happens
        bias_term = self.b_dec * self.cfg.apply_b_dec_to_input
        # print(f"Bias term shape: {bias_term.shape}")

        return sae_in - bias_term

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SAE."""
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        if self.use_error_term:
            with torch.no_grad():
                # Recompute without hooks for true error term
                with _disable_hooks(self):
                    feature_acts_clean = self.encode(x)
                    x_reconstruct_clean = self.decode(feature_acts_clean)
                sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            sae_out = sae_out + sae_error

        return self.hook_sae_output(sae_out)

    # overwrite this in subclasses to modify the state_dict in-place before saving
    def process_state_dict_for_saving(self, state_dict: dict[str, Any]) -> None:
        pass

    # overwrite this in subclasses to modify the state_dict in-place after loading
    def process_state_dict_for_loading(self, state_dict: dict[str, Any]) -> None:
        pass

    @torch.no_grad()
    def fold_W_dec_norm(self):
        """Fold decoder norms into encoder."""
        W_dec_norms = self.W_dec.norm(dim=-1).unsqueeze(1)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * W_dec_norms.T

        # Only update b_enc if it exists (standard/jumprelu architectures)
        if hasattr(self, "b_enc") and isinstance(self.b_enc, nn.Parameter):
            self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()

    def get_name(self):
        """Generate a name for this SAE."""
        return f"sae_{self.cfg.model_name}_{self.cfg.hook_name}_{self.cfg.d_sae}"

    def save_model(
        self, path: str | Path, sparsity: torch.Tensor | None = None
    ) -> tuple[Path, Path, Path | None]:
        """Save model weights, config, and optional sparsity tensor to disk."""
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)

        # Generate the weights
        state_dict = self.state_dict()  # Use internal SAE state dict
        self.process_state_dict_for_saving(state_dict)
        model_weights_path = path / SAE_WEIGHTS_FILENAME
        save_file(state_dict, model_weights_path)

        # Save the config
        config = self.cfg.to_dict()
        cfg_path = path / SAE_CFG_FILENAME
        with open(cfg_path, "w") as f:
            json.dump(config, f)

        if sparsity is not None:
            sparsity_in_dict = {"sparsity": sparsity}
            sparsity_path = path / SPARSITY_FILENAME
            save_file(sparsity_in_dict, sparsity_path)
            return model_weights_path, cfg_path, sparsity_path

        return model_weights_path, cfg_path, None

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

        logger.info("Reinitializing b_dec with mean of activations")
        logger.debug(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        logger.debug(f"New distances: {distances.median(0).values.mean().item()}")

        self.b_dec.data = out.to(self.dtype).to(self.device)

    # Class methods for loading models
    @classmethod
    @deprecated("Use load_from_disk instead")
    def load_from_pretrained(
        cls: Type[T], path: str | Path, device: str = "cpu", dtype: str | None = None
    ) -> T:
        return cls.load_from_disk(path, device=device, dtype=dtype)

    @classmethod
    def load_from_disk(
        cls: Type[T],
        path: str | Path,
        device: str = "cpu",
        dtype: str | None = None,
        converter: PretrainedSaeDiskLoader = sae_lens_disk_loader,
    ) -> T:
        overrides = {"dtype": dtype} if dtype is not None else None
        cfg_dict, state_dict = converter(path, device, cfg_overrides=overrides)
        cfg_dict = handle_config_defaulting(cfg_dict)
        sae_config_cls = cls.get_sae_config_class_for_architecture(
            cfg_dict["architecture"]
        )
        sae_cfg = sae_config_cls.from_dict(cfg_dict)
        sae_cls = cls.get_sae_class_for_architecture(sae_cfg.architecture)
        sae = sae_cls(sae_cfg)
        sae.process_state_dict_for_loading(state_dict)
        sae.load_state_dict(state_dict)
        return sae

    @classmethod
    def from_pretrained(
        cls,
        release: str,
        sae_id: str,
        device: str = "cpu",
        force_download: bool = False,
        converter: PretrainedSaeHuggingfaceLoader | None = None,
    ) -> tuple["SAE", dict[str, Any], torch.Tensor | None]:
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

        # Validate release and sae_id
        if release not in sae_directory:
            if "/" not in release:
                raise ValueError(
                    f"Release {release} not found in pretrained SAEs directory, and is not a valid huggingface repo."
                )
        elif sae_id not in sae_directory[release].saes_map:
            # Handle special cases like Gemma Scope
            if (
                "gemma-scope" in release
                and "canonical" not in release
                and f"{release}-canonical" in sae_directory
            ):
                canonical_ids = list(
                    sae_directory[release + "-canonical"].saes_map.keys()
                )
                # Shorten the lengthy string of valid IDs
                if len(canonical_ids) > 5:
                    str_canonical_ids = str(canonical_ids[:5])[:-1] + ", ...]"
                else:
                    str_canonical_ids = str(canonical_ids)
                value_suffix = f" If you don't want to specify an L0 value, consider using release {release}-canonical which has valid IDs {str_canonical_ids}"
            else:
                value_suffix = ""

            valid_ids = list(sae_directory[release].saes_map.keys())
            # Shorten the lengthy string of valid IDs
            if len(valid_ids) > 5:
                str_valid_ids = str(valid_ids[:5])[:-1] + ", ...]"
            else:
                str_valid_ids = str(valid_ids)

            raise ValueError(
                f"ID {sae_id} not found in release {release}. Valid IDs are {str_valid_ids}."
                + value_suffix
            )

        conversion_loader = (
            converter
            or NAMED_PRETRAINED_SAE_LOADERS[get_conversion_loader_name(release)]
        )
        repo_id, folder_name = get_repo_id_and_folder_name(release, sae_id)
        config_overrides = get_config_overrides(release, sae_id)
        config_overrides["device"] = device

        # Load config and weights
        cfg_dict, state_dict, log_sparsities = conversion_loader(
            repo_id=repo_id,
            folder_name=folder_name,
            device=device,
            force_download=force_download,
            cfg_overrides=config_overrides,
        )
        cfg_dict = handle_config_defaulting(cfg_dict)

        # Rename keys to match SAEConfig field names
        renamed_cfg_dict = {}
        rename_map = {
            "hook_point": "hook_name",
            "hook_point_layer": "hook_layer",
            "hook_point_head_index": "hook_head_index",
            "activation_fn": "activation_fn",
        }

        for k, v in cfg_dict.items():
            renamed_cfg_dict[rename_map.get(k, k)] = v

        # Set default values for required fields
        renamed_cfg_dict.setdefault("activation_fn_kwargs", {})
        renamed_cfg_dict.setdefault("seqpos_slice", None)

        # Create SAE with appropriate architecture
        sae_config_cls = cls.get_sae_config_class_for_architecture(
            renamed_cfg_dict["architecture"]
        )
        sae_cfg = sae_config_cls.from_dict(renamed_cfg_dict)
        sae_cls = cls.get_sae_class_for_architecture(sae_cfg.architecture)
        sae = sae_cls(sae_cfg)
        sae.process_state_dict_for_loading(state_dict)
        sae.load_state_dict(state_dict)

        # Apply normalization if needed
        if renamed_cfg_dict.get("normalize_activations") == "expected_average_only_in":
            norm_scaling_factor = get_norm_scaling_factor(release, sae_id)
            if norm_scaling_factor is not None:
                sae.fold_activation_norm_scaling_factor(norm_scaling_factor)
                renamed_cfg_dict["normalize_activations"] = "none"
            else:
                warnings.warn(
                    f"norm_scaling_factor not found for {release} and {sae_id}, but normalize_activations is 'expected_average_only_in'. Skipping normalization folding."
                )

        return sae, renamed_cfg_dict, log_sparsities

    @classmethod
    def from_dict(cls: Type[T], config_dict: dict[str, Any]) -> T:
        """Create an SAE from a config dictionary."""
        sae_cls = cls.get_sae_class_for_architecture(config_dict["architecture"])
        sae_config_cls = cls.get_sae_config_class_for_architecture(
            config_dict["architecture"]
        )
        return sae_cls(sae_config_cls.from_dict(config_dict))

    @classmethod
    def get_sae_class_for_architecture(cls: Type[T], architecture: str) -> Type[T]:
        """Get the SAE class for a given architecture."""
        sae_cls = get_sae_class(architecture)
        if not issubclass(sae_cls, cls):
            raise ValueError(
                f"Loaded SAE is not of type {cls.__name__}. Use {sae_cls.__name__} instead"
            )
        return sae_cls

    # in the future, this can be used to load different config classes for different architectures
    @classmethod
    def get_sae_config_class_for_architecture(
        cls: Type[T],
        architecture: str,  # noqa: ARG003
    ) -> type[SAEConfig]:
        return SAEConfig


@dataclass(kw_only=True)
class TrainingSAEConfig(SAEConfig):
    # Sparsity Loss Calculations
    l1_coefficient: float
    lp_norm: float
    use_ghost_grads: bool
    normalize_sae_decoder: bool
    noise_scale: float
    decoder_orthogonal_init: bool
    mse_loss_normalization: str | None
    jumprelu_init_threshold: float
    jumprelu_bandwidth: float
    decoder_heuristic_init: bool
    init_encoder_as_decoder_transpose: bool
    scale_sparsity_penalty_by_decoder_norm: bool

    @classmethod
    def from_sae_runner_config(
        cls, cfg: LanguageModelSAERunnerConfig
    ) -> "TrainingSAEConfig":
        return cls(
            # base config
            architecture=cfg.architecture,
            d_in=cfg.d_in,
            d_sae=cfg.d_sae,  # type: ignore
            dtype=cfg.dtype,
            device=cfg.device,
            model_name=cfg.model_name,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            hook_head_index=cfg.hook_head_index,
            activation_fn=cfg.activation_fn,
            activation_fn_kwargs=cfg.activation_fn_kwargs,
            apply_b_dec_to_input=cfg.apply_b_dec_to_input,
            finetuning_scaling_factor=cfg.finetuning_method is not None,
            sae_lens_training_version=cfg.sae_lens_training_version,
            context_size=cfg.context_size,
            dataset_path=cfg.dataset_path,
            prepend_bos=cfg.prepend_bos,
            seqpos_slice=tuple(x for x in cfg.seqpos_slice if x is not None)
            if cfg.seqpos_slice is not None
            else None,
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
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
            model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs or {},
            jumprelu_init_threshold=cfg.jumprelu_init_threshold,
            jumprelu_bandwidth=cfg.jumprelu_bandwidth,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAEConfig":
        # remove any keys that are not in the dataclass
        # since we sometimes enhance the config with the whole LM runner config
        valid_field_names = {field.name for field in fields(cls)}
        valid_config_dict = {
            key: val for key, val in config_dict.items() if key in valid_field_names
        }

        # ensure seqpos slice is tuple
        # ensure that seqpos slices is a tuple
        # Ensure seqpos_slice is a tuple
        if "seqpos_slice" in valid_config_dict:
            if isinstance(valid_config_dict["seqpos_slice"], list):
                valid_config_dict["seqpos_slice"] = tuple(
                    valid_config_dict["seqpos_slice"]
                )
            elif not isinstance(valid_config_dict["seqpos_slice"], tuple):
                valid_config_dict["seqpos_slice"] = (valid_config_dict["seqpos_slice"],)

        return TrainingSAEConfig(**valid_config_dict)

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
            "jumprelu_init_threshold": self.jumprelu_init_threshold,
            "jumprelu_bandwidth": self.jumprelu_bandwidth,
        }

    # this needs to exist so we can initialize the parent sae cfg without the training specific
    # parameters. Maybe there's a cleaner way to do this
    def get_base_sae_cfg_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "activation_fn": self.activation_fn,
            "activation_fn_kwargs": self.activation_fn_kwargs,
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "dtype": self.dtype,
            "model_name": self.model_name,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "hook_head_index": self.hook_head_index,
            "device": self.device,
            "context_size": self.context_size,
            "prepend_bos": self.prepend_bos,
            "finetuning_scaling_factor": self.finetuning_scaling_factor,
            "normalize_activations": self.normalize_activations,
            "dataset_path": self.dataset_path,
            "dataset_trust_remote_code": self.dataset_trust_remote_code,
            "sae_lens_training_version": self.sae_lens_training_version,
            "model_from_pretrained_kwargs": self.model_from_pretrained_kwargs,
            "seqpos_slice": self.seqpos_slice,
            "neuronpedia_id": self.neuronpedia_id,
        }


class TrainingSAE(SAE, ABC):
    """Abstract base class for training versions of SAEs."""

    cfg: "TrainingSAEConfig"  # type: ignore

    def __init__(self, cfg: TrainingSAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

        # Turn off hook_z reshaping for training mode - the activation store
        # is expected to handle reshaping before passing data to the SAE
        self.turn_off_forward_pass_hook_z_reshaping()

        self.mse_loss_fn = self._get_mse_loss_fn()

    @abstractmethod
    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        """Encode with access to pre-activation values for training."""
        pass

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        For inference, just encode without returning hidden_pre.
        (training_forward_pass calls encode_with_hidden_pre).
        """
        feature_acts, _ = self.encode_with_hidden_pre(x)
        return feature_acts

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Decodes feature activations back into input space,
        applying optional finetuning scale, hooking, out normalization, etc.
        """
        scaled_features = self.apply_finetuning_scaling_factor(feature_acts)
        sae_out_pre = scaled_features @ self.W_dec + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @abstractmethod
    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Calculate architecture-specific auxiliary loss terms."""
        pass

    def training_forward_pass(
        self,
        step_input: TrainStepInput,
    ) -> TrainStepOutput:
        """Forward pass during training."""
        feature_acts, hidden_pre = self.encode_with_hidden_pre(step_input.sae_in)
        sae_out = self.decode(feature_acts)

        # Calculate MSE loss
        per_item_mse_loss = self.mse_loss_fn(sae_out, step_input.sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        # Calculate architecture-specific auxiliary losses
        aux_losses = self.calculate_aux_loss(
            step_input=step_input,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            sae_out=sae_out,
        )

        # Total loss is MSE plus all auxiliary losses
        total_loss = mse_loss

        # Create losses dictionary with mse_loss
        losses = {"mse_loss": mse_loss}

        # Add architecture-specific losses to the dictionary
        # Make sure aux_losses is a dictionary with string keys and tensor values
        if isinstance(aux_losses, dict):
            losses.update(aux_losses)

        # Sum all losses for total_loss
        if isinstance(aux_losses, dict):
            for loss_value in aux_losses.values():
                total_loss = total_loss + loss_value
        else:
            # Handle case where aux_losses is a tensor
            total_loss = total_loss + aux_losses

        return TrainStepOutput(
            sae_in=step_input.sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=total_loss,
            losses=losses,
        )

    def _get_mse_loss_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get the MSE loss function based on config."""

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
        return standard_mse_loss_fn

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self) -> None:
        """Remove gradient components parallel to decoder directions."""
        # Implement the original logic since this may not be in the base class
        assert self.W_dec.grad is not None

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

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        """Normalize decoder columns to unit norm."""
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def log_histograms(self) -> dict[str, NDArray[Any]]:
        """Log histograms of the weights and biases."""
        W_dec_norm_dist = self.W_dec.detach().float().norm(dim=1).cpu().numpy()
        return {
            "weights/W_dec_norms": W_dec_norm_dist,
        }

    @classmethod
    def get_sae_class_for_architecture(cls: Type[T], architecture: str) -> Type[T]:
        """Get the SAE class for a given architecture."""
        sae_cls = get_sae_training_class(architecture)
        if not issubclass(sae_cls, cls):
            raise ValueError(
                f"Loaded SAE is not of type {cls.__name__}. Use {sae_cls.__name__} instead"
            )
        return sae_cls

    # in the future, this can be used to load different config classes for different architectures
    @classmethod
    def get_sae_config_class_for_architecture(
        cls: Type[T],
        architecture: str,  # noqa: ARG003
    ) -> type[SAEConfig]:
        return TrainingSAEConfig


_blank_hook = nn.Identity()


@contextmanager
def _disable_hooks(sae: SAE):
    """
    Temporarily disable hooks for the SAE. Swaps out all the hooks with a fake modules that does nothing.
    """
    try:
        for hook_name in sae.hook_dict:
            setattr(sae, hook_name, _blank_hook)
        yield
    finally:
        for hook_name, hook in sae.hook_dict.items():
            setattr(sae, hook_name, hook)
