# ruff: noqa: E402
__version__ = "6.0.0-rc.5"

import logging

logger = logging.getLogger(__name__)

from sae_lens.saes import (
    SAE,
    GatedSAE,
    GatedSAEConfig,
    GatedTrainingSAE,
    GatedTrainingSAEConfig,
    JumpReLUSAE,
    JumpReLUSAEConfig,
    JumpReLUTrainingSAE,
    JumpReLUTrainingSAEConfig,
    SAEConfig,
    StandardSAE,
    StandardSAEConfig,
    StandardTrainingSAE,
    StandardTrainingSAEConfig,
    TopKSAE,
    TopKSAEConfig,
    TopKTrainingSAE,
    TopKTrainingSAEConfig,
    TrainingSAE,
    TrainingSAEConfig,
)

from .analysis.hooked_sae_transformer import HookedSAETransformer
from .cache_activations_runner import CacheActivationsRunner
from .config import (
    CacheActivationsRunnerConfig,
    LanguageModelSAERunnerConfig,
    LoggingConfig,
    PretokenizeRunnerConfig,
)
from .evals import run_evals
from .llm_sae_training_runner import LanguageModelSAETrainingRunner, SAETrainingRunner
from .loading.pretrained_sae_loaders import (
    PretrainedSaeDiskLoader,
    PretrainedSaeHuggingfaceLoader,
)
from .pretokenize_runner import PretokenizeRunner, pretokenize_runner
from .registry import register_sae_class, register_sae_training_class
from .training.activations_store import ActivationsStore
from .training.upload_saes_to_huggingface import upload_saes_to_huggingface

__all__ = [
    "SAE",
    "SAEConfig",
    "TrainingSAE",
    "TrainingSAEConfig",
    "HookedSAETransformer",
    "ActivationsStore",
    "LanguageModelSAERunnerConfig",
    "LanguageModelSAETrainingRunner",
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
    "StandardSAE",
    "StandardSAEConfig",
    "StandardTrainingSAE",
    "StandardTrainingSAEConfig",
    "GatedSAE",
    "GatedSAEConfig",
    "GatedTrainingSAE",
    "GatedTrainingSAEConfig",
    "TopKSAE",
    "TopKSAEConfig",
    "TopKTrainingSAE",
    "TopKTrainingSAEConfig",
    "JumpReLUSAE",
    "JumpReLUSAEConfig",
    "JumpReLUTrainingSAE",
    "JumpReLUTrainingSAEConfig",
    "SAETrainingRunner",
    "LoggingConfig",
]


register_sae_class("standard", StandardSAE, StandardSAEConfig)
register_sae_training_class("standard", StandardTrainingSAE, StandardTrainingSAEConfig)
register_sae_class("gated", GatedSAE, GatedSAEConfig)
register_sae_training_class("gated", GatedTrainingSAE, GatedTrainingSAEConfig)
register_sae_class("topk", TopKSAE, TopKSAEConfig)
register_sae_training_class("topk", TopKTrainingSAE, TopKTrainingSAEConfig)
register_sae_class("jumprelu", JumpReLUSAE, JumpReLUSAEConfig)
register_sae_training_class("jumprelu", JumpReLUTrainingSAE, JumpReLUTrainingSAEConfig)
