import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from torch import nn

from sae_lens.saes.sae import SAE, SAEConfig, TrainingSAE, TrainStepInput


class StandardSAE(SAE):
    """
    StandardSAE is an inference-only implementation of a Sparse Autoencoder (SAE)
    using a simple linear encoder and decoder.

    It implements the required abstract methods from BaseSAE:
      - initialize_weights: sets up simple parameter initializations for W_enc, b_enc, W_dec, and b_dec.
      - encode: computes the feature activations from an input.
      - decode: reconstructs the input from the feature activations.

    The BaseSAE.forward() method automatically calls encode and decode,
    including any error-term processing if configured.
    """

    b_enc: nn.Parameter

    def __init__(self, cfg: SAEConfig, use_error_term: bool = False):
        super().__init__(cfg, use_error_term)

    def initialize_weights(self) -> None:
        # Initialize encoder weights and bias.
        self.b_enc = nn.Parameter(
            torch.zeros(self.cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.b_dec = nn.Parameter(
            torch.zeros(self.cfg.d_in, dtype=self.dtype, device=self.device)
        )

        # Use Kaiming Uniform for W_enc
        w_enc_data = torch.empty(
            self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_enc_data)
        self.W_enc = nn.Parameter(w_enc_data)

        # Use Kaiming Uniform for W_dec
        w_dec_data = torch.empty(
            self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
        )
        nn.init.kaiming_uniform_(w_dec_data)
        self.W_dec = nn.Parameter(w_dec_data)

    def encode(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Encode the input tensor into the feature space.
        For inference, no noise is added.
        """
        # Preprocess the SAE input (casting type, applying hooks, normalization)
        sae_in = self.process_sae_in(x)
        # Compute the pre-activation values
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
        # Apply the activation function (e.g., ReLU, tanh-relu, depending on config)
        return self.hook_sae_acts_post(self.activation_fn(hidden_pre))

    def decode(
        self, feature_acts: Float[torch.Tensor, "... d_sae"]
    ) -> Float[torch.Tensor, "... d_in"]:
        """
        Decode the feature activations back to the input space.
        Now, if hook_z reshaping is turned on, we reverse the flattening.
        """
        # 1) apply finetuning scaling if configured.
        scaled_features = self.apply_finetuning_scaling_factor(feature_acts)
        # 2) linear transform
        sae_out_pre = scaled_features @ self.W_dec + self.b_dec
        # 3) hook reconstruction
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        # 4) optional out-normalization (e.g. constant_norm_rescale or layer_norm)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        # 5) if hook_z is enabled, rearrange back to (..., n_heads, d_head).
        return self.reshape_fn_out(sae_out_pre, self.d_head)


class StandardTrainingSAE(TrainingSAE):
    """
    StandardTrainingSAE is a concrete implementation of BaseTrainingSAE using the "standard" SAE architecture.
    It implements:
      - initialize_weights: basic weight initialization for encoder/decoder.
      - encode: inference encoding (invokes encode_with_hidden_pre).
      - decode: a simple linear decoder.
      - encode_with_hidden_pre: computes pre-activations, adds noise when training, and then activates.
      - calculate_aux_loss: computes a sparsity penalty based on the (optionally scaled) p-norm of feature activations.
    """

    b_enc: nn.Parameter

    def initialize_weights(self) -> None:
        # Basic init
        # In Python MRO, this calls StandardSAE.initialize_weights()
        StandardSAE.initialize_weights(self)  # type: ignore

        # Complex init logic from original TrainingSAE
        if self.cfg.decoder_orthogonal_init:
            self.W_dec.data = nn.init.orthogonal_(self.W_dec.data.T).T

        elif self.cfg.decoder_heuristic_init:
            self.W_dec.data = torch.rand(  # Changed from Parameter to data assignment
                self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
            )
            self.initialize_decoder_norm_constant_norm()

        if self.cfg.init_encoder_as_decoder_transpose:
            self.W_enc.data = self.W_dec.data.T.clone().contiguous()  # type: ignore

        if self.cfg.normalize_sae_decoder:
            with torch.no_grad():
                self.set_decoder_norm_to_unit_norm()

    @torch.no_grad()
    def initialize_decoder_norm_constant_norm(self, norm: float = 0.1):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)  # type: ignore
        self.W_dec.data *= norm

    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:
        # Process the input (including dtype conversion, hook call, and any activation normalization)
        sae_in = self.process_sae_in(x)
        # Compute the pre-activation (and allow for a hook if desired)
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)  # type: ignore
        # Add noise during training for robustness (scaled by noise_scale from the configuration)
        if self.training and self.cfg.noise_scale > 0:
            hidden_pre_noised = (
                hidden_pre + torch.randn_like(hidden_pre) * self.cfg.noise_scale
            )
        else:
            hidden_pre_noised = hidden_pre
        # Apply the activation function (and any post-activation hook)
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre_noised))
        return feature_acts, hidden_pre_noised

    def calculate_aux_loss(
        self,
        step_input: TrainStepInput,
        feature_acts: torch.Tensor,
        hidden_pre: torch.Tensor,
        sae_out: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # The "standard" auxiliary loss is a sparsity penalty on the feature activations
        weighted_feature_acts = feature_acts
        if self.cfg.scale_sparsity_penalty_by_decoder_norm:
            weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)

        # Compute the p-norm (set by cfg.lp_norm) over the feature dimension
        sparsity = weighted_feature_acts.norm(p=self.cfg.lp_norm, dim=-1)
        l1_loss = (step_input.current_l1_coefficient * sparsity).mean()

        return {"l1_loss": l1_loss}

    def log_histograms(self) -> dict[str, NDArray[np.generic]]:
        """Log histograms of the weights and biases."""
        b_e_dist = self.b_enc.detach().float().cpu().numpy()
        return {
            **super().log_histograms(),
            "weights/b_e": b_e_dist,
        }
