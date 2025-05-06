from dataclasses import dataclass, field
from typing import Any, List

import einops
import torch
from jaxtyping import Float

from sae_lens import SAEConfig, SAE
from sae_lens.toolkit.pretrained_sae_loaders import (
    PretrainedSaeDiskLoader,
    handle_config_defaulting,
    sae_lens_disk_loader,
)

@dataclass
class CrosscoderSAEConfig(SAEConfig):
    hook_names: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict() | {
            "hook_names": self.hook_names,
        }

class CrosscoderSAE(SAE):
    """
    Sparse autoencoder that acts on multiple layers of activations.
    """

    def __init__(
            self,
            cfg: CrosscoderSAEConfig,
            use_error_term: bool = False,
            ):
        if cfg.architecture != "standard":
            raise NotImplementedError(
                "TODO(mkbehr): support other architectures")

        super().__init__(cfg=cfg, use_error_term=use_error_term)

        if self.hook_z_reshaping_mode:
            raise NotImplementedError("TODO(mkbehr): support hook_z")

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "CrosscoderSAE":
        return cls(CrosscoderSAEConfig.from_dict(config_dict))

    def input_shape(self):
        return (len(self.cfg.hook_names), self.cfg.d_in)


    def encode_standard(
        self, x: Float[torch.Tensor, "... n_layers d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Calculate SAE features from inputs
        """
        sae_in = self.process_sae_in(x)

        hidden_pre = self.hook_sae_acts_pre(
            einops.einsum(
                sae_in, self.W_enc,
                "... n_layers d_in, n_layers d_in d_sae -> ... d_sae"
                )
            + self.b_enc)
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... n_layers d_in"]:
        """Decodes SAE feature activation tensor into a reconstructed
        input activation tensor."""
        sae_out = self.hook_sae_recons(
            einops.einsum(
                self.apply_finetuning_scaling_factor(feature_acts),
                self.W_dec,
                "... d_sae, d_sae n_layers d_in -> ... n_layers d_in"
            ) + self.b_dec
        )

        # handle run time activation normalization if needed
        # will fail if you call this twice without calling encode in between.
        sae_out = self.run_time_activation_norm_fn_out(sae_out)

        # handle hook z reshaping if needed.
        return self.reshape_fn_out(sae_out, self.d_head)  # type: ignore

    @torch.no_grad()
    def fold_W_dec_norm(self):
        W_dec_norms = self.W_dec.norm(dim=[-2,-1], keepdim=True)
        self.W_dec.data = self.W_dec.data / W_dec_norms
        self.W_enc.data = self.W_enc.data * einops.rearrange(
            W_dec_norms, "d_sae 1 1 -> 1 1 d_sae")
        if self.cfg.architecture == "gated":
            self.r_mag.data = self.r_mag.data * W_dec_norms.squeeze()
            self.b_gate.data = self.b_gate.data * W_dec_norms.squeeze()
            self.b_mag.data = self.b_mag.data * W_dec_norms.squeeze()
        elif self.cfg.architecture == "jumprelu":
            self.threshold.data = self.threshold.data * W_dec_norms.squeeze()
            self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()
        else:
            self.b_enc.data = self.b_enc.data * W_dec_norms.squeeze()

    @torch.no_grad()
    def fold_activation_norm_scaling_factor(
        self, activation_norm_scaling_factor: Float[torch.Tensor, "n_layers"]
    ):
        self.W_enc.data = self.W_enc.data * activation_norm_scaling_factor.reshape((-1,1,1))
        # previously weren't doing this.
        self.W_dec.data = self.W_dec.data / activation_norm_scaling_factor.unsqueeze(-1)
        self.b_dec.data = self.b_dec.data / activation_norm_scaling_factor.unsqueeze(-1)

        # once we normalize, we shouldn't need to scale activations.
        self.cfg.normalize_activations = "none"

    @classmethod
    def load_from_disk(
        cls,
        path: str,
        device: str = "cpu",
        dtype: str | None = None,
        converter: PretrainedSaeDiskLoader = sae_lens_disk_loader,
    ) -> "CrosscoderSAE":
        overrides = {"dtype": dtype} if dtype is not None else None
        cfg_dict, state_dict = converter(path, device, cfg_overrides=overrides)
        cfg_dict = handle_config_defaulting(cfg_dict)
        sae_cfg = CrosscoderSAEConfig.from_dict(cfg_dict)
        sae = cls(sae_cfg)
        sae.process_state_dict_for_loading(state_dict)
        sae.load_state_dict(state_dict)
        return sae
