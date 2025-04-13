import json
import os
from dataclasses import dataclass
from typing import Any

import einops
import torch
from jaxtyping import Float

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.crosscoder_sae import CrosscoderSAE, CrosscoderSAEConfig
from sae_lens.training.training_sae import (
    TrainingSAEConfig,
    TrainingSAE,
    TrainStepOutput,
    )
from sae_lens.toolkit.pretrained_sae_loaders import (
    handle_config_defaulting,
    read_sae_from_disk,
)

SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"


# TODO(mkbehr) will this multiple inheritance work?
@dataclass(kw_only=True)
class TrainingCrosscoderSAEConfig(CrosscoderSAEConfig, TrainingSAEConfig):
    sparsity_penalty_decoder_norm_lp_norm: float = 1

    # TODO(mkbehr): copypasting from TrainingSAEConfig and adding a few
    # params. There should be a better way.
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
            hook_layers=cfg.hook_layers,
            hook_head_index=cfg.hook_head_index,
            activation_fn_str=cfg.activation_fn,
            activation_fn_kwargs=cfg.activation_fn_kwargs,
            apply_b_dec_to_input=cfg.apply_b_dec_to_input,
            finetuning_scaling_factor=cfg.finetuning_method is not None,
            sae_lens_training_version=cfg.sae_lens_training_version,
            context_size=cfg.context_size,
            dataset_path=cfg.dataset_path,
            prepend_bos=cfg.prepend_bos,
            seqpos_slice=cfg.seqpos_slice,
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

    def to_dict(self) -> dict[str, Any]:
        # TODO(mkbehr): double-check this multiple inheritance. seems messy.
        return (TrainingSAE.to_dict(self)
                | CrosscoderSAE.to_dict(self)
                | {
                    "sparsity_penalty_decoder_norm_lp_norm":
                    self.sparsity_penalty_decoder_norm_lp_norm,
                })

    def get_base_sae_cfg_dict(self) -> dict[str, Any]:
        return (TrainingSAEConfig.get_base_sae_cfg_dict(self)
                | { "hook_layers": self.hook_layers })

class TrainingCrosscoderSAE(CrosscoderSAE, TrainingSAE):
    # TODO(mkbehr) future implementation
    # initialize_weights_jumprelu (can maybe just use input shape in trainingsae)
    # encode_with_hidden_pre_{gated,jumprelu}
    # calculate_topk_aux_loss
    # calculate_ghost_grad_loss
    # fold_W_dec_norm for jumprelu

    def __init__(self,
                 cfg: TrainingCrosscoderSAEConfig,
                 use_error_term: bool = False):
        super().__init__(cfg, use_error_term=use_error_term)

    @classmethod
    def from_dict(cls,
                  config_dict: dict[str, Any],
                  use_error_term: bool = False) -> "TrainingSAE":
        return cls(TrainingCrosscoderSAEConfig.from_dict(config_dict),
                   use_error_term = use_error_term)

    # TODO(mkbehr): hacking around multiple inheritance. there's
    # probably a better way.
    @staticmethod
    def base_sae_cfg(cfg: TrainingCrosscoderSAEConfig):
        return CrosscoderSAEConfig.from_dict(cfg.get_base_sae_cfg_dict())

    def check_cfg_compatibility(self):
        if self.cfg.architecture != "standard":
            raise NotImplementedError("TODO(mkbehr): support other archs")
        if not self.cfg.scale_sparsity_penalty_by_decoder_norm:
            raise ValueError("Crosscoders require scale_sparsity_penalty_by_decoder_norm")
        if not self.use_error_term:
            raise NotImplementedError("TODO(mkbehr): support causal crosscoders")
        if self.cfg.use_ghost_grads:
            raise NotImplementedError("TODO(mkbehr): support ghost grads")
        super().check_cfg_compatibility()

    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... n_layers d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        sae_in = self.process_sae_in(x)

        hidden_pre = self.hook_sae_acts_pre(
            einops.einsum(
                sae_in, self.W_enc,
                "... n_layers d_in, n_layers d_in d_sae -> ... d_sae"
                )
            + self.b_enc)
        hidden_pre_noised = hidden_pre + (
            torch.randn_like(hidden_pre) * self.cfg.noise_scale * self.training
        )
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre_noised))

        return feature_acts, hidden_pre_noised

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: torch.Tensor | None = None,
    ) -> TrainStepOutput:
        # do a forward pass to get SAE out, but we also need the
        # hidden pre.
        feature_acts, hidden_pre = self.encode_with_hidden_pre_fn(sae_in)
        sae_out = self.decode(feature_acts)

        # MSE LOSS
        per_item_mse_loss = self.mse_loss_fn(sae_out, sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        losses: dict[str, float | torch.Tensor] = {}

        assert self.cfg.scale_sparsity_penalty_by_decoder_norm
        decoder_norms = self.W_dec.norm(dim=2)
        feature_act_weights = decoder_norms.norm(
            p=self.cfg.sparsity_penalty_decoder_norm_lp_norm,
            dim=1
        )
        weighted_feature_acts = feature_acts * feature_act_weights
        sparsity = weighted_feature_acts.norm(
            p=self.cfg.lp_norm, dim=-1
        )  # sum over the feature dimension

        l1_loss = (current_l1_coefficient * sparsity).mean()
        loss = mse_loss + l1_loss
        if (
            self.cfg.use_ghost_grads
            and self.training
            and dead_neuron_mask is not None
        ):
            ghost_grad_loss = self.calculate_ghost_grad_loss(
                x=sae_in,
                sae_out=sae_out,
                per_item_mse_loss=per_item_mse_loss,
                hidden_pre=hidden_pre,
                dead_neuron_mask=dead_neuron_mask,
            )
            losses["ghost_grad_loss"] = ghost_grad_loss
            loss = loss + ghost_grad_loss
        losses["l1_loss"] = l1_loss

        losses["mse_loss"] = mse_loss

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=loss,
            losses=losses,
        )

    @classmethod
    def load_from_pretrained(
        cls,
        path: str,
        device: str = "cpu",
        dtype: str | None = None,
    ) -> "TrainingCrosscoderSAE":
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
        sae_cfg = TrainingCrosscoderSAEConfig.from_dict(cfg_dict)

        sae = cls(sae_cfg)
        sae.process_state_dict_for_loading(state_dict)
        sae.load_state_dict(state_dict)

        return sae

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=[1,2], keepdim=True)

    @torch.no_grad()
    def initialize_decoder_norm_constant_norm(self, norm: float = 0.1):
        """
        A heuristic proceedure inspired by:
        https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        """
        # TODO: Parameterise this as a function of m and n

        # ensure W_dec norms at unit norm
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=[1,2], keepdim=True)
        self.W_dec.data *= norm  # will break tests but do this for now.

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, n_layers, d_in) shape
        """
        assert self.W_dec.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae n_layers d_in, d_sae n_layers d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae n_layers d_in -> d_sae n_layers d_in",
        )

