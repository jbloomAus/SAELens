from dataclasses import dataclass
from typing import Any, List

import einops
import torch
from jaxtyping import Float

from sae_lens import SAEConfig, SAE

@dataclass
class CrosscoderSAEConfig(SAEConfig):
    hook_layers: list[int] = list

    # @classmethod
    # def from_dict(cls, config_dict: dict[str, Any]) -> "CrosscoderSAEConfig":
    #     # TODO(mkbehr) is a new method needed here, or will the superclass's work w/o modification? I think it'll work. test it.
    #     pass

    def to_dict(self) -> dict[str, Any]:
        # TODO(mkbehr) test
        return super().to_dict() | {
            "hook_layers": self.hook_layers,
        }

    def hook_names(self) -> List[str]:
        # TODO(mkbehr): better config setup than putting a magic
        # string in the name
        return [self.hook_name.format(layer)
                for layer in self.hook_layers]


class CrosscoderSAE(SAE):
    """
    TODO(mkbehr): docstring
    """

    # TODO(mkbehr): write
    # - remaining encode methods
    # - hook_z reshaping support

    def __init__(
            self,
            cfg: CrosscoderSAEConfig,
            use_error_term: bool = False,
            ):
        if cfg.architecture != "standard":
            raise NotImplementedError("TODO(mkbehr): support other archs")

        super().__init__(cfg=cfg, use_error_term=use_error_term)

        if self.hook_z_reshaping_mode:
            raise NotImplementedError("TODO(mkbehr): support hook_z")

    def get_name(self):
        # TODO(mkbehr): think about the correct name
        layers = '_'.join([str(l) for l in self.cfg.hook_layers])
        return f"sae_{self.cfg.model_name}_{self.cfg.hook_name}_layers{layers}_{self.cfg.d_sae}"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "CrosscoderSAE":
        return cls(CrosscoderSAEConfig.from_dict(config_dict))

    def input_shape(self):
        return (len(self.cfg.hook_layers), self.cfg.d_in)

    # TODO(mkbehr): in sae.py this is noted to output "... d_sae" but
    # I think that's wrong
    # TODO(mkbehr): I don't think we actually need to change this
    def process_sae_in(
        self, sae_in: Float[torch.Tensor, "... n_layers d_in"]
    ) -> Float[torch.Tensor, "... n_layers d_in"]:
        sae_in = sae_in.to(self.dtype)
        # TODO(mkbehr): n.b. that reshape_fn_in is set to the identity
        # if we're not doing hook_z reshaping
        sae_in = self.reshape_fn_in(sae_in)
        sae_in = self.hook_sae_input(sae_in)
        sae_in = self.run_time_activation_norm_fn_in(sae_in)
        return sae_in - (self.b_dec * self.cfg.apply_b_dec_to_input)

    def encode_standard(
        self, x: Float[torch.Tensor, "... n_layers d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Calculate SAE features from inputs
        """
        # TODO(mkbehr): instead of changing this and the W_enc/b_enc
        # dimensions, we could change reshape_fn_in
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
        # TODO(mkbehr)
        # W_dec: d_sae, n_layers, d_in
        # W_dec_norms: d_sae, 1, 1
        # W_enc: n_layers, d_in, d_sae
        # desired W_enc_norms: 1, 1, d_sae
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

