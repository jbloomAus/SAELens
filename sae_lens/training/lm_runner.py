from typing import Any, cast

import torch
import wandb

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.geometric_median import compute_geometric_median
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from sae_lens.training.train_sae_on_language_model import (
    train_sae_group_on_language_model,
)


def language_model_sae_runner(cfg: LanguageModelSAERunnerConfig):
    """ """
    if cfg.from_pretrained_path is not None:
        (
            model,
            sparse_autoencoder,
            activations_loader,
        ) = LMSparseAutoencoderSessionloader.load_pretrained_sae(
            cfg.from_pretrained_path
        )
        cfg = sparse_autoencoder.cfg
    else:
        loader = LMSparseAutoencoderSessionloader(cfg)
        model, sparse_autoencoder, activations_loader = (
            loader.load_sae_training_group_session()
        )
        _init_sae_group_b_decs(sparse_autoencoder, activations_loader)

    if cfg.log_to_wandb:
        wandb.init(
            project=cfg.wandb_project,
            config=cast(Any, cfg),
            name=cfg.run_name,
            id=cfg.wandb_id,
        )

    # Compile model and SAE
    # torch.compile can provide significant speedups (10-20% in testing)
    # using max-autotune gives the best speedups but:
    # (a) increases VRAM usage,
    # (b) can't be used on both SAE and LM (some issue with cudagraphs), and
    # (c) takes some time to compile
    # optimal settings seem to be:
    # use max-autotune on SAE and max-autotune-no-cudagraphs on LM
    # (also pylance seems to really hate this)
    if cfg.compile_llm:
        model = torch.compile(
            model,  # pyright: ignore [reportPossiblyUnboundVariable]
            mode=cfg.llm_compilation_mode,
        )

    if cfg.compile_sae:
        for (
            k
        ) in (
            sparse_autoencoder.autoencoders.keys()  # pyright: ignore [reportPossiblyUnboundVariable]
        ):
            sae = sparse_autoencoder.autoencoders[  # pyright: ignore [reportPossiblyUnboundVariable]
                k
            ]
            sae = torch.compile(sae, mode=cfg.sae_compilation_mode)
            sparse_autoencoder.autoencoders[k] = sae  # type: ignore # pyright: ignore [reportPossiblyUnboundVariable]

    # train SAE
    sparse_autoencoder = train_sae_group_on_language_model(
        model=model,  # pyright: ignore [reportPossiblyUnboundVariable] # type: ignore
        sae=sparse_autoencoder,  # pyright: ignore [reportPossiblyUnboundVariable]
        activation_store=activations_loader,  # pyright: ignore [reportPossiblyUnboundVariable]
        batch_size=cfg.train_batch_size_tokens,
        n_checkpoints=cfg.n_checkpoints,
        feature_sampling_window=cfg.feature_sampling_window,
        use_wandb=cfg.log_to_wandb,
        wandb_log_frequency=cfg.wandb_log_frequency,
        eval_every_n_wandb_logs=cfg.eval_every_n_wandb_logs,
        autocast=cfg.autocast,
        n_eval_batches=cfg.n_eval_batches,
        eval_batch_size_prompts=cfg.eval_batch_size_prompts,
    ).sae

    if cfg.log_to_wandb:
        wandb.finish()

    return sparse_autoencoder


def _init_sae_group_b_decs(
    sae: SparseAutoencoder,
    activation_store: ActivationsStore,
) -> None:
    """
    extract all activations at a certain layer and use for sae b_dec initialization
    """

    if sae.cfg.b_dec_init_method == "geometric_median":
        layer_acts = activation_store.storage_buffer.detach()[:, 0, :]
        # get geometric median of the activations if we're using those.
        median = compute_geometric_median(
            layer_acts,
            maxiter=100,
        ).median
        sae.initialize_b_dec_with_precalculated(median)
    elif sae.cfg.b_dec_init_method == "mean":
        layer_acts = activation_store.storage_buffer.detach().cpu()[:, 0, :]
        sae.initialize_b_dec_with_mean(layer_acts)
