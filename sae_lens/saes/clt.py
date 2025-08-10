from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import nn
from typing_extensions import override

from sae_lens.saes.sae import SAE, SAEConfig, _disable_hooks


@dataclass
class StandardCLTConfig(SAEConfig):
    """Configuration for JumpReLU CLTs."""

    n_layers: int = 1
    include_skip_connection: bool = False
    include_input_bias: bool = False
    d_out: int = None  # type: ignore
    decode_after_source_layer_only: bool = True

    @override
    @classmethod
    def architecture(cls) -> str:
        return "standard_clt"

    @property
    def d_clt(self) -> int:
        return self.d_sae

    def __post_init__(self):
        super().__post_init__()
        if self.d_out is None:
            self.d_out = self.d_in
        if self.apply_b_dec_to_input is True:
            raise ValueError("apply_b_dec_to_input must be False for CLT architectures")


class StandardCLT(SAE[StandardCLTConfig]):
    """Abstract base class for all CLT architectures."""

    dtype: torch.dtype
    device: torch.device

    W_enc: nn.Parameter  # shape n_layers x d_in x d_clt
    W_dec: nn.ParameterList  # shape n_layers x d_clt x n_layers x d_out
    b_enc: nn.Parameter  # shape n_layers x d_clt
    b_dec: nn.Parameter  # shape n_layers x d_out
    b_in: nn.Parameter | None  # shape n_layers x d_in
    W_skip: nn.Parameter | None  # shape n_layers x d_in x d_out

    def __init__(self, cfg: StandardCLTConfig, use_error_term: bool = False):
        """Initialize the SAE."""
        super().__init__(cfg, use_error_term)

    @override
    def process_sae_in(
        self, sae_in: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_in"]:
        sae_in = sae_in.to(self.dtype)
        sae_in = self.reshape_fn_in(sae_in)
        sae_in = self.hook_sae_input(sae_in)
        sae_in = self.run_time_activation_norm_fn_in(sae_in)

        if self.b_in is not None:
            sae_in = sae_in - self.b_in
        return sae_in

    @override
    def initialize_weights(self) -> None:
        self.b_dec = nn.Parameter(
            torch.zeros(
                self.cfg.n_layers, self.cfg.d_out, dtype=self.dtype, device=self.device
            )
        )
        if self.cfg.include_input_bias:
            self.b_in = nn.Parameter(
                torch.zeros(
                    self.cfg.n_layers,
                    self.cfg.d_in,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        else:
            self.b_in = None
        self.W_dec = torch.nn.ParameterList(  # type: ignore
            [
                torch.nn.Parameter(
                    torch.zeros(
                        self.cfg.d_sae,
                        self.cfg.n_layers - i
                        if self.cfg.decode_after_source_layer_only
                        else self.cfg.n_layers,
                        self.cfg.d_out,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
                for i in range(self.cfg.n_layers)
            ]
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
        self.b_enc = nn.Parameter(
            torch.zeros(
                self.cfg.n_layers, self.cfg.d_sae, dtype=self.dtype, device=self.device
            )
        )

        if self.cfg.include_skip_connection:
            self.W_skip = nn.Parameter(
                torch.zeros(
                    self.cfg.n_layers,
                    self.cfg.d_in,
                    self.cfg.d_out,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        else:
            self.W_skip = None

    @override
    def encode(
        self, x: Float[torch.Tensor, "... n_layers d_in"]
    ) -> Float[torch.Tensor, "... n_layers d_sae"]:
        """Encode input tensor to feature space."""
        sae_in = self.process_sae_in(x)
        hidden_pre = torch.einsum("lbd,ldf->lbf", sae_in, self.W_enc) + self.b_enc
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    @override
    def decode(
        self, feature_acts: Float[torch.Tensor, "... n_layers d_sae"]
    ) -> Float[torch.Tensor, "... n_layers d_out"]:
        """Decode feature activations to input space."""
        pos_ids, layer_ids, _feat_ids, decoder_vectors, _encoder_mapping = (
            self.select_decoder_vectors(feature_acts)
        )
        sae_out_pre = self.compute_reconstruction(pos_ids, layer_ids, decoder_vectors)
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    def select_decoder_vectors(self, features: torch.Tensor):
        # from https://github.com/safety-research/circuit-tracer/blob/main/circuit_tracer/transcoder/cross_layer_transcoder.py
        if not features.is_sparse:
            features = features.to_sparse()
        layer_idx, pos_idx, feat_idx = features.indices()
        activations = features.values()
        n_layers = features.shape[0]
        device = features.device

        pos_ids = []
        layer_ids = []
        feat_ids = []

        decoder_vectors = []
        encoder_mapping = []
        st = 0

        for layer_id in range(n_layers):
            current_layer = layer_idx == layer_id
            if not current_layer.any():
                continue

            current_layer_features = feat_idx[current_layer]
            unique_feats, inv = current_layer_features.unique(return_inverse=True)

            unique_decoders = self.W_dec[layer_id][unique_feats]
            scaled_decoders = (
                unique_decoders[inv] * activations[current_layer, None, None]
            )
            decoder_vectors.append(scaled_decoders.reshape(-1, self.d_model))

            n_output_layers = (
                self.cfg.n_layers - layer_id
                if self.cfg.decode_after_source_layer_only
                else self.cfg.n_layers
            )
            pos_ids.append(pos_idx[current_layer].repeat_interleave(n_output_layers))
            feat_ids.append(current_layer_features.repeat_interleave(n_output_layers))
            layer_ids.append(
                torch.arange(layer_id, self.cfg.n_layers, device=device).repeat(
                    len(current_layer_features)
                )
            )

            source_ids = torch.arange(len(current_layer_features), device=device) + st
            st += len(current_layer_features)
            encoder_mapping.append(torch.repeat_interleave(source_ids, n_output_layers))

        pos_ids = torch.cat(pos_ids, dim=0)
        layer_ids = torch.cat(layer_ids, dim=0)
        feat_ids = torch.cat(feat_ids, dim=0)
        decoder_vectors = torch.cat(decoder_vectors, dim=0)
        encoder_mapping = torch.cat(encoder_mapping, dim=0)

        return pos_ids, layer_ids, feat_ids, decoder_vectors, encoder_mapping

    def compute_reconstruction(
        self,
        pos_ids: torch.Tensor,
        layer_ids: torch.Tensor,
        decoder_vectors: torch.Tensor,
    ) -> torch.Tensor:
        n_pos = int(pos_ids.max()) + 1
        flat_idx = layer_ids * n_pos + pos_ids
        recon = torch.zeros(
            n_pos * self.cfg.n_layers,
            self.cfg.d_out,
            device=decoder_vectors.device,
            dtype=decoder_vectors.dtype,
        ).index_add_(0, flat_idx, decoder_vectors)
        return recon.reshape(self.cfg.n_layers, n_pos, self.cfg.d_out) + self.b_dec

    @override
    def forward(
        self, x: Float[torch.Tensor, "... n_layers d_in"]
    ) -> Float[torch.Tensor, "... n_layers d_out"]:
        features = self.encode(x).to_sparse()
        decoded = self.decode(features)
        if self.W_skip is not None:
            with _disable_hooks(self):
                sae_in = self.process_sae_in(x)
            skip = torch.einsum("ij,ijk->ik", sae_in, self.W_skip)
            decoded = decoded + skip
        return decoded
