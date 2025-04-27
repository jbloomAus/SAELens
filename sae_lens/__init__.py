# ruff: noqa: E402
__version__ = "5.9.1"

import logging

logger = logging.getLogger(__name__)

from sae_lens.saes.gated_sae import (
    GatedSAE,
    GatedSAEConfig,
    GatedTrainingSAE,
    GatedTrainingSAEConfig,
)
from sae_lens.saes.jumprelu_sae import (
    JumpReLUSAE,
    JumpReLUSAEConfig,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
)
from sae_lens.saes.standard_sae import (
    StandardSAE,
    StandardSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
)
from sae_lens.saes.topk_sae import (
    TopKSAE,
    TopKSAEConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
)

from .analysis.hooked_sae_transformer import HookedSAETransformer
from .cache_activations_runner import CacheActivationsRunner
from .config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
    PretokenizeRunnerConfig,
)
from .evals import run_evals
from .loading.pretrained_sae_loaders import (
    PretrainedSaeDiskLoader,
    PretrainedSaeHuggingfaceLoader,
)
from .pretokenize_runner import PretokenizeRunner, pretokenize_runner
from .regsitry import register_sae_class, register_sae_training_class
from .sae_training_runner import SAETrainingRunner
from .saes.sae import SAE, SAEConfig, TrainingSAE, TrainingSAEConfig
from .training.activations_store import ActivationsStore
from .training.upload_saes_to_huggingface import upload_saes_to_huggingface

register_sae_class("standard", StandardSAE, StandardSAEConfig)
register_sae_training_class("standard", StandardTrainingSAE, StandardTrainingSAEConfig)
register_sae_class("gated", GatedSAE, GatedSAEConfig)
register_sae_training_class("gated", GatedTrainingSAE, GatedTrainingSAEConfig)
register_sae_class("topk", TopKSAE, TopKSAEConfig)
register_sae_training_class("topk", TopKTrainingSAE, TopKTrainingSAEConfig)
register_sae_class("jumprelu", JumpReLUSAE, JumpReLUSAEConfig)
register_sae_training_class("jumprelu", JumpReLUTrainingSAE, JumpReLUTrainingSAEConfig)

__all__ = [
    "SAE",
    "SAEConfig",
    "TrainingSAE",
    "TrainingSAEConfig",
    "HookedSAETransformer",
    "ActivationsStore",
    "LanguageModelSAERunnerConfig",
    "SAETrainingRunner",
    "CacheActivationsRunnerConfig",
    "CacheActivationsRunner",
    "PretokenizeRunnerConfig",
    "PretokenizeRunner",
    "pretokenize_runner",
    "run_evals",
    "upload_saes_to_huggingface",
    "PretrainedSaeHuggingfaceLoader",
    "PretrainedSaeDiskLoader",
    "register_sae_class",
    "register_sae_training_class",
]
