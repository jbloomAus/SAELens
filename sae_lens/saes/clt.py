from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import nn
from typing_extensions import override

from sae_lens.saes.sae import SAE, SAEConfig


@dataclass
class JumpReLUCLTConfig(SAEConfig):
    """Configuration for JumpReLU CLTs."""

    n_layers: int = 1
    include_skip_connection: bool = False
    d_out: int = None  # type: ignore

    @override
    @classmethod
    def architecture(cls) -> str:
        return "jumprelu_clt"

    @property
    def d_clt(self) -> int:
        return self.d_sae

    def __post_init__(self):
        super().__post_init__()
        if self.d_out is None:
            self.d_out = self.d_in
        if self.apply_b_dec_to_input is True:
            raise ValueError("apply_b_dec_to_input must be False for CLT architectures")


class JumpReLUCLT(SAE[JumpReLUCLTConfig]):
    """Abstract base class for all CLT architectures."""

    dtype: torch.dtype
    device: torch.device

    W_enc: nn.Parameter  # shape n_layers x d_in x d_transcoder
    W_dec: nn.Parameter  # shape n_layers x d_transcoder x n_layers x d_out
    b_enc: nn.Parameter  # shape n_layers x d_transcoder
    b_dec: nn.Parameter  # shape n_layers x d_out
    b_in: nn.Parameter | None = None  # shape n_layers x d_in
    W_skip: nn.Parameter | None = None  # shape n_layers x d_out x d_in
    threshold: nn.Parameter  # shape n_layers x d_transcoder

    def __init__(self, cfg: JumpReLUCLTConfig, use_error_term: bool = False):
        """Initialize the SAE."""
        super().__init__(cfg, use_error_term)

    @override
    def process_sae_in(
        self, sae_in: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_in"]:
        processed_sae_in = super().process_sae_in(sae_in)
        if self.b_in is not None:
            processed_sae_in = processed_sae_in - self.b_in
        return processed_sae_in

    @override
    def initialize_weights(self) -> None:
        self.b_dec = nn.Parameter(
            torch.zeros(
                self.cfg.n_layers, self.cfg.d_out, dtype=self.dtype, device=self.device
            )
        )
        self.b_in = nn.Parameter(
            torch.zeros(
                self.cfg.n_layers, self.cfg.d_in, dtype=self.dtype, device=self.device
            )
        )
        self.W_dec = nn.Parameter(
            torch.zeros(
                self.cfg.n_layers,
                self.cfg.d_sae,
                self.cfg.n_layers,
                self.cfg.d_out,
                dtype=self.dtype,
                device=self.device,
            )
        )

        self.W_enc = nn.Parameter(
            torch.zeros(
                self.cfg.n_layers,
                self.cfg.d_in,
                self.cfg.d_sae,
                dtype=self.dtype,
                device=self.device,
            )
        )
        self.threshold = nn.Parameter(
            torch.zeros(
                self.cfg.n_layers, self.cfg.d_sae, dtype=self.dtype, device=self.device
            )
        )
        self.b_enc = nn.Parameter(
            torch.zeros(
                self.cfg.n_layers, self.cfg.d_sae, dtype=self.dtype, device=self.device
            )
        )

    @override
    def encode(
        self, x: Float[torch.Tensor, "... n_layers d_in"]
    ) -> Float[torch.Tensor, "... n_layers d_sae"]:
        """Encode input tensor to feature space."""
        sae_in = self.process_sae_in(x)
        hidden_pre = torch.einsum("ij,ijk->ik", sae_in, self.W_enc) + self.b_enc
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    @override
    def decode(
        self, feature_acts: Float[torch.Tensor, "... n_layers d_sae"]
    ) -> Float[torch.Tensor, "... n_layers d_out"]:
        """Decode feature activations to input space."""
        sae_out_pre = torch.einsum("ij,ijkl->kl", feature_acts, self.W_dec) + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)
