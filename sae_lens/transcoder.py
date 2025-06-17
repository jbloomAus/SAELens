from __future__ import annotations

"""Transcoder module: variant of sparse coder that maps *input* activations
from one hook-point (d_in) to *output* activations at another hook-point
(d_out).

The implementation purposefully re-uses as much logic as possible from the
existing SAE class – we only override places where the differing output
size/bias makes a difference.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from sae_lens.sae import SAE, SAEConfig

__all__ = ["TranscoderConfig", "Transcoder"]


# ---------------------------------------------------------------------------
#                               CONFIGURATION
# ---------------------------------------------------------------------------


@dataclass
class TranscoderConfig(SAEConfig):
    """Configuration for a Transcoder.

    Inherits all fields from :class:`~sae_lens.sae.SAEConfig` and adds the
    information needed for the *output* hook.
    """

    # Transcoder-specific sizes
    d_out: int = -1  # will be checked for validity in __post_init__

    # Target (output) hook location
    hook_name_out: str = ""
    hook_layer_out: int = -1
    hook_head_index_out: Optional[int] = None

    # ---------------------------------------------------------------------
    # Serialisation helpers – extend the parent implementation so that the
    # new attributes are saved / restored transparently.
    # ---------------------------------------------------------------------

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TranscoderConfig":
        cfg = super().from_dict(config_dict)  # type: ignore[arg-type]
        # `super()` returns an SAEConfig instance; convert to our subclass.
        return cls(**cfg.__dict__)  # pyright: ignore[reportGeneralTypeIssues]

    def to_dict(self) -> dict[str, Any]:  # type: ignore[override]
        base = super().to_dict()
        base.update(
            {
                "d_out": self.d_out,
                "hook_name_out": self.hook_name_out,
                "hook_layer_out": self.hook_layer_out,
                "hook_head_index_out": self.hook_head_index_out,
            }
        )
        return base


# ---------------------------------------------------------------------------
#                                   MODEL
# ---------------------------------------------------------------------------


class Transcoder(SAE):
    """Sparse auto-encoder variant with different input & output hook dims."""

    cfg: TranscoderConfig  # type: ignore[assignment]

    # We do *not* currently support the error-term variant for transcoders.
    def __init__(self, cfg: TranscoderConfig, use_error_term: bool = False):
        if use_error_term:
            raise NotImplementedError(
                "Error-term mode is not implemented for Transcoder yet."
            )

        # === Runtime Type Check ===
        if not hasattr(cfg, 'd_out') or not hasattr(cfg, 'hook_name_out'):
            raise TypeError(
                "Configuration object seems to be for a standard SAE "
                "(missing 'd_out' or 'hook_name_out'). Please use "
                "SAE.from_pretrained() or load_artifact_from_pretrained()."
            )
        # Check if trying to load SkipTranscoder with base Transcoder class
        # (This relies on a hypothetical config flag or check in the loader)
        # Example: if cfg.architecture == "skip_transcoder" and not isinstance(self, SkipTranscoder):
        #    raise TypeError("Config indicates SkipTranscoder, use SkipTranscoder.from_pretrained()")
        # ==========================

        # Perform the usual SAE initialisation first.
        super().__init__(cfg, use_error_term=False)

        # After the SAE constructor runs, we overwrite / extend the parameters
        # that differ in shape or semantics.
        self._init_transcoder_decoder_params()

    # ------------------------------------------------------------------
    #                         Parameter initialisation
    # ------------------------------------------------------------------

    def _init_transcoder_decoder_params(self) -> None:
        """Create a decoder that maps to *d_out* instead of *d_in*.
        """

        # Re-initialise decoder weight with shape (d_sae, d_out)
        self.W_dec = nn.Parameter(  # type: ignore[assignment]
            torch.nn.init.kaiming_uniform_(
                torch.empty(
                    self.cfg.d_sae,
                    self.cfg.d_out,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        )

    # ------------------------------------------------------------------
    #                            Decoding logic
    # ------------------------------------------------------------------

    def get_sae_recons(
        self, feature_acts: torch.Tensor
    ) -> torch.Tensor:  # type: ignore[override]
        """Return the reconstructed *output* activations.

        The reconstruction shares the same bias ``b_dec`` that is (optionally)
        subtracted from the input during encoding and added back here.
        """

        return (
            self.apply_finetuning_scaling_factor(feature_acts) @ self.W_dec
            + self.b_dec
        )

    # ------------------------------------------------------------------
    #                Activation-norm scaling – no extra parameters
    # ------------------------------------------------------------------

    # We do not introduce additional parameters beyond those handled in the
    # parent SAE class, so we can simply inherit its behaviour.

    # ------------------------------------------------------------------
    #                              Loading
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        release: str,
        sae_id: str,
        device: str = "cpu",
        force_download: bool = False,
        converter: Any | None = None,  # kept for API compatibility, ignored
    ) -> tuple["Transcoder", dict[str, Any], torch.Tensor | None]:
        """
        Load a pretrained Transcoder from the Hugging Face model hub.

        Args:
            release: The release name.
            sae_id: The id of the Transcoder/SAE to load.
            device: The device to load the Transcoder on.
            force_download: Whether to force download.

        Returns:
            A tuple containing the loaded Transcoder, its config dictionary,
            and the log sparsity tensor (or None if not available).
        """
        # This method largely mirrors SAE.from_pretrained but ensures
        # that a TranscoderConfig is instantiated.

        # Imports are inline to avoid circular dependency if Transcoder
        # were imported at the top of pretrained_sae_loaders.py
        from sae_lens.toolkit.pretrained_sae_loaders import (
            NAMED_PRETRAINED_SAE_LOADERS,
            NAMED_PRETRAINED_SAE_CONFIG_GETTERS,
            get_conversion_loader_name,
            handle_config_defaulting,
        )
        from sae_lens.toolkit.pretrained_saes_directory import (
            get_config_overrides,
            get_norm_scaling_factor,
            get_pretrained_saes_directory,
            get_repo_id_and_folder_name,
        )

        # get sae directory
        sae_directory = get_pretrained_saes_directory()

        # Validate release and sae_id (same logic as SAE.from_pretrained)
        if release not in sae_directory:
            msg = (
                "Release {release} not found in pretrained SAEs directory, "
                "and is not a valid huggingface repo."
            )
            raise ValueError(msg.format(release=release))
        elif sae_id not in sae_directory[release].saes_map:
            valid_ids = list(sae_directory[release].saes_map.keys())
            if len(valid_ids) > 5:
                str_valid_ids = str(valid_ids[:5])[:-1] + ", ...]"
            else:
                str_valid_ids = str(valid_ids)
            # Add specific Transcoder hints if applicable (optional)
            raise ValueError(
                (
                    f"ID {sae_id} not found in release {release}. "
                    f"Valid IDs are {str_valid_ids}."
                )
            )

        # --- Fetch release info and potential revision ---
        sae_release_from_dir = sae_directory[release]
        # --------------------------------------------------

        conversion_loader_name = get_conversion_loader_name(release)
        config_getter = (
            NAMED_PRETRAINED_SAE_CONFIG_GETTERS[
                conversion_loader_name
            ]
        )
        conversion_loader = (
            NAMED_PRETRAINED_SAE_LOADERS[
                conversion_loader_name
            ]
        )
        repo_id, folder_name = get_repo_id_and_folder_name(
            release, sae_id
        )
        config_overrides = get_config_overrides(
            release, sae_id
        )
        # Ensure config_overrides is a dict
        if config_overrides is None:
            config_overrides = {}
        config_overrides["device"] = device

        # --- Add revision from directory to overrides ---
        # if sae_release_from_dir.revision is not None:
        #     config_overrides.setdefault("revision", sae_release_from_dir.revision)
        # Safely check for revision attribute
        release_revision = getattr(sae_release_from_dir, "revision", None)
        if release_revision is not None:
            config_overrides.setdefault("revision", release_revision)
        # ------------------------------------------------

        # 1. Get the config dictionary using the appropriate getter
        cfg_dict = config_getter(
            repo_id=repo_id,
            folder_name=folder_name,
            device=device,
            force_download=force_download,
            cfg_overrides=config_overrides,
        )
        cfg_dict = handle_config_defaulting(cfg_dict)

        # 2. **Crucially, instantiate TranscoderConfig**
        transcoder_cfg = TranscoderConfig.from_dict(cfg_dict)

        # 3. Load the state dict using the appropriate loader
        # We reuse cfg_dict here as the loader might need original dict items
        _, state_dict, log_sparsities = conversion_loader(
            repo_id=repo_id,
            folder_name=folder_name,
            device=device,
            force_download=force_download,
            # Pass potentially modified cfg_dict back
            cfg_overrides=cfg_dict,
        )

        # 4. Instantiate the Transcoder
        transcoder = cls(transcoder_cfg)
        transcoder.process_state_dict_for_loading(state_dict)
        transcoder.load_state_dict(state_dict)

        # 5. Handle normalization folding if needed (copied from SAE.from_pretrained)
        if transcoder_cfg.normalize_activations == "expected_average_only_in":
            norm_scaling_factor = get_norm_scaling_factor(release, sae_id)
            if norm_scaling_factor is not None:
                transcoder.fold_activation_norm_scaling_factor(
                    norm_scaling_factor
                )
                # Update cfg_dict as well, as it's returned
                cfg_dict["normalize_activations"] = "none"
            else:
                import warnings  # local import

                msg = (
                    "norm_scaling_factor not found for {rel} and {sid}, "
                    "but normalize_activations is 'expected_average_only_in'. "
                    "Skipping normalization folding."
                ).format(
                    rel=release,
                    sid=sae_id,
                )
                warnings.warn(msg)

        return transcoder, cfg_dict, log_sparsities 


# ---------------------------------------------------------------------------
#                             SKIP  TRANSCODER
# ---------------------------------------------------------------------------


__all__.append("SkipTranscoder")


class SkipTranscoder(Transcoder):
    """Transcoder with an additional learnable *skip* connection.

    Implements

        f(x) = W_dec · JumpReLU(W_enc x + b_enc) + W_skip x + b_dec

    as described in Gemma-Scope (Appx. B).  `W_skip` is initialised to
    zeros, so the model starts as a constant function.
    """

    def __init__(
        self,
        cfg: TranscoderConfig,
        subtract_bias: bool = True,
        use_error_term: bool = False,
    ):
        # Store flag before the parent constructor sets up parameters
        self.subtract_bias = subtract_bias

        super().__init__(cfg, use_error_term=use_error_term)

        # Skip connection weight (initialised to zeros)
        self.W_skip = nn.Parameter(
            torch.zeros(
                self.cfg.d_in,
                self.cfg.d_out,
                dtype=self.dtype,
                device=self.device,
            )
        )

        # Extra hook for inspecting skip contribution
        from transformer_lens.hook_points import HookPoint

        self.hook_skip = HookPoint()

    # ------------------------------------------------------------------
    #                             Forward pass
    # ------------------------------------------------------------------

    def compute_skip(self, input_acts: torch.Tensor) -> torch.Tensor:
        """Compute the linear skip term *x · W_skip^T*."""

        return input_acts @ self.W_skip.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Encode → decode → add skip connection."""

        # Encode & decode (Transcoder logic)
        feature_acts = self.encode(x)
        decoded = self.decode(feature_acts)

        # Compute skip term. Optionally subtract bias first (training uses it).
        skip_input = x - self.b_dec if self.subtract_bias else x
        skip = self.hook_skip(
            self.compute_skip(skip_input)
        )

        return self.hook_sae_output(
            decoded + skip
        ) 