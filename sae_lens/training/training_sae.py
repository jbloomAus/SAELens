"""Most of this is just copied over from Arthur's code and slightly simplified:
https://github.com/ArthurConmy/sae/blob/main/sae/model.py
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import einops
import torch
from jaxtyping import Float
from torch import nn

from sae_lens import logger
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.loading.pretrained_sae_loaders import (
    handle_config_defaulting,
    read_sae_from_disk,
)
from sae_lens.sae import SAE
from sae_lens.saes.gated_sae import GatedTrainingSAE
from sae_lens.saes.jumprelu_sae import JumpReLUTrainingSAE
from sae_lens.saes.sae_base import BaseTrainingSAE, TrainingSAEConfig, TrainStepOutput
from sae_lens.saes.standard_sae import StandardTrainingSAE
from sae_lens.saes.topk_sae import TopKTrainingSAE

SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"


def rectangle(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x)


class Step(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x > threshold).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[None, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        threshold_grad = torch.sum(
            -(1.0 / bandwidth) * rectangle((x - threshold) / bandwidth) * grad_output,
            dim=0,
        )
        return None, threshold_grad, None


class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        threshold: torch.Tensor,
        bandwidth: float,  # noqa: ARG004
    ) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output  # We don't apply STE to x input
        threshold_grad = torch.sum(
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None


def create_training_sae_from_config(
    cfg: TrainingSAEConfig, use_error_term: bool = False
) -> BaseTrainingSAE:
    """
    Factory function to create the appropriate training SAE instance based on architecture.

    Args:
        cfg: Training SAE configuration
        use_error_term: Whether to use the error term in the forward pass

    Returns:
        An instance of the appropriate training SAE class
    """
    architecture = cfg.architecture.lower()

    if architecture == "standard":
        return StandardTrainingSAE(cfg, use_error_term)
    if architecture == "gated":
        return GatedTrainingSAE(cfg, use_error_term)
    if architecture == "jumprelu":
        return JumpReLUTrainingSAE(cfg, use_error_term)
    if architecture == "topk":
        return TopKTrainingSAE(cfg, use_error_term)
    raise ValueError(f"Unsupported architecture: {architecture}")


class TrainingSAE(SAE):
    """
    A facade/factory class for training-specific SAE implementations.
    This class delegates to architecture-specific training implementations
    while maintaining backward compatibility with existing code.
    """

    # Fix typing issue by making _sae explicitly BaseTrainingSAE, not overriding _sae from SAE
    _sae: BaseTrainingSAE

    @property
    def cfg(self) -> TrainingSAEConfig:
        # Remove unnecessary cast
        return self._sae.cfg

    def __init__(
        self,
        cfg: Union[TrainingSAEConfig, LanguageModelSAERunnerConfig],
        use_error_term: bool = False,
    ):
        """Initialize with the appropriate training SAE implementation."""
        # Skip the standard SAE initialization and initialize the HookedRootModule directly
        nn.Module.__init__(self)

        # Convert LanguageModelSAERunnerConfig to TrainingSAEConfig if needed
        if not isinstance(cfg, TrainingSAEConfig):
            cfg = TrainingSAEConfig.from_sae_runner_config(cfg)

        # Create the appropriate training implementation based on architecture
        self._sae = create_training_sae_from_config(cfg, use_error_term)

        # Create property handles for parameters
        self._param_names = []
        # Fix unused variable warning by using _ for param
        for name, _ in self._sae.named_parameters():
            self._param_names.append(name)
            # Use property to dynamically access the parameter from _sae
            setattr(
                self.__class__,
                name,
                property(
                    lambda self, name=name: getattr(self._sae, name),
                    lambda self, value, name=name: setattr(self._sae, name, value),
                ),
            )

        # Forward the hooks and hook dict from the internal implementation
        self.hook_dict = self._sae.hook_dict
        self.setup()  # Required for HookedRootModule

    # Basic delegation methods with training-specific functionality
    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        """Forward to the internal implementation's encode_with_hidden_pre method."""
        return self._sae.encode_with_hidden_pre(x)

    def encode_with_hidden_pre_fn(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        """Forward to the appropriate encode_with_hidden_pre method based on architecture."""
        return self._sae.encode_with_hidden_pre(x)

    def encode_with_hidden_pre_gated(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        """Forward to GatedTrainingSAE's encode_with_hidden_pre method."""
        if not isinstance(self._sae, GatedTrainingSAE):
            raise TypeError("This method is only available for Gated SAEs")
        return self._sae.encode_with_hidden_pre(x)

    def encode_with_hidden_pre_jumprelu(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        """Forward to JumpReLUTrainingSAE's encode_with_hidden_pre method."""
        if not isinstance(self._sae, JumpReLUTrainingSAE):
            raise TypeError("This method is only available for JumpReLU SAEs")
        return self._sae.encode_with_hidden_pre(x)

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: Optional[torch.Tensor] = None,
    ) -> TrainStepOutput:
        """
        Forward to the internal implementation's training_forward_pass.
        No longer modifies loss keys for backward compatibility.
        """
        return self._sae.training_forward_pass(
            sae_in=sae_in,
            current_l1_coefficient=current_l1_coefficient,
            dead_neuron_mask=dead_neuron_mask,
        )

    # Forward additional properties/methods from the internal implementation
    @property
    def threshold(self) -> torch.Tensor:
        """Forward to JumpReLU's threshold property."""
        if not isinstance(self._sae, JumpReLUTrainingSAE):
            raise AttributeError("threshold is only available for JumpReLU SAEs")
        return self._sae.threshold

    @property
    def bandwidth(self) -> float:
        """Forward to JumpReLU's bandwidth property."""
        if not isinstance(self._sae, JumpReLUTrainingSAE):
            raise AttributeError("bandwidth is only available for JumpReLU SAEs")
        return self._sae.bandwidth

    @property
    def mse_loss_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward to the internal implementation's mse_loss_fn."""
        return self._sae.mse_loss_fn

    @mse_loss_fn.setter
    def mse_loss_fn(self, new_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self._sae.mse_loss_fn = new_fn

    def _get_mse_loss_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward to the internal implementation's _get_mse_loss_fn method."""
        return self._sae._get_mse_loss_fn()

    # State dict processing methods
    def process_state_dict_for_saving(self, state_dict: Dict[str, Any]) -> None:
        """Forward to the internal implementation's process_state_dict_for_saving method."""
        if hasattr(self._sae, "process_state_dict_for_saving"):
            method = getattr(self._sae, "process_state_dict_for_saving")
            if callable(method):
                method(state_dict)

    def process_state_dict_for_loading(self, state_dict: Dict[str, Any]) -> None:
        """Forward to the internal implementation's process_state_dict_for_loading method."""
        if hasattr(self._sae, "process_state_dict_for_loading"):
            method = getattr(self._sae, "process_state_dict_for_loading")
            if callable(method):
                method(state_dict)

    # Initialization methods
    def initialize_weights_complex(self) -> None:
        """Replicate the original complex weight initialization logic."""
        if hasattr(self._sae, "initialize_weights_complex"):
            method = getattr(self._sae, "initialize_weights_complex")
            if callable(method):
                method()

    @torch.no_grad()
    def initialize_decoder_norm_constant_norm(self, norm: float = 0.1) -> None:
        """Initialize decoder with constant norm."""
        if hasattr(self._sae, "initialize_decoder_norm_constant_norm"):
            method = getattr(self._sae, "initialize_decoder_norm_constant_norm")
            if callable(method):
                method(norm)

    @torch.no_grad()
    def fold_W_dec_norm(self) -> None:
        """Fold decoder norm into encoder weights."""
        self._sae.fold_W_dec_norm()

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self) -> None:
        """Set decoder norm to unit norm."""
        if hasattr(self._sae, "set_decoder_norm_to_unit_norm"):
            method = getattr(self._sae, "set_decoder_norm_to_unit_norm")
            if callable(method):
                method()

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

    # Backward compatibility class methods
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingSAE":
        """Create a TrainingSAE from a config dictionary."""
        return cls(TrainingSAEConfig.from_dict(config_dict))

    @classmethod
    def load_from_pretrained(
        cls,
        path: Union[str, Path],
        device: str = "cpu",
        dtype: Optional[str] = None,
    ) -> "TrainingSAE":
        """Load a pretrained TrainingSAE from disk."""
        # get the config
        config_path = os.path.join(path, SAE_CFG_PATH)
        with open(config_path) as f:
            cfg_dict = json.load(f)
        cfg_dict = handle_config_defaulting(cfg_dict)
        cfg_dict["device"] = device
        if dtype is not None:
            cfg_dict["dtype"] = dtype

        weight_path = os.path.join(path, SAE_WEIGHTS_PATH)
        cfg_dict, state_dict = read_sae_from_disk(
            cfg_dict=cfg_dict,
            weight_path=weight_path,
            device=device,
        )

        sae_cfg = TrainingSAEConfig.from_dict(cfg_dict)
        sae = cls(sae_cfg)
        sae.process_state_dict_for_loading(state_dict)
        sae._sae.load_state_dict(state_dict)
        return sae

    def initialize_b_dec_with_precalculated(self, origin: torch.Tensor) -> None:
        """Initialize b_dec with precalculated values."""
        out = torch.tensor(origin, dtype=self.dtype, device=self.device)
        self.b_dec.data = out

    def initialize_b_dec_with_mean(self, all_activations: torch.Tensor) -> None:
        """Initialize b_dec with mean of activations."""
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

    def check_cfg_compatibility(self) -> None:
        """Check configuration compatibility."""
        if self.cfg.architecture != "standard" and self.cfg.use_ghost_grads:
            raise ValueError(f"{self.cfg.architecture} SAEs do not support ghost grads")
        if self.cfg.architecture == "gated" and self.use_error_term:
            raise ValueError("Gated SAEs do not support error terms")

    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward to TopKTrainingSAE's calculate_topk_aux_loss."""
        if not isinstance(self._sae, TopKTrainingSAE):
            raise TypeError("TopK aux loss is only available for TopK SAEs")
        return self._sae.calculate_topk_aux_loss(
            sae_in=sae_in,
            sae_out=sae_out,
            hidden_pre=hidden_pre,
            dead_neuron_mask=dead_neuron_mask,
        )


def _calculate_topk_aux_acts(
    k_aux: int,
    hidden_pre: torch.Tensor,
    dead_neuron_mask: torch.Tensor,
) -> torch.Tensor:
    # Don't include living latents in this loss
    auxk_latents = torch.where(dead_neuron_mask[None], hidden_pre, -torch.inf)
    # Top-k dead latents
    auxk_topk = auxk_latents.topk(k_aux, sorted=False)
    # Set the activations to zero for all but the top k_aux dead latents
    auxk_acts = torch.zeros_like(hidden_pre)
    auxk_acts.scatter_(-1, auxk_topk.indices, auxk_topk.values)
    # Set activations to zero for all but top k_aux dead latents
    return auxk_acts
