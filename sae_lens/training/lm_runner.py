import traceback
from typing import Any, cast

import torch
import wandb

from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.load_model import load_model
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.training.train_sae_on_language_model import (
    load_checkpoint,
    train_sae_group_on_language_model,
)


def language_model_sae_runner(cfg: LanguageModelSAERunnerConfig):
    """ """
    training_run_state = None
    train_contexts = None

    if cfg.resume:
        try:
            checkpoint_path = cfg.get_resume_checkpoint_path()
            model = load_model(
                model_class_name=cfg.model_class_name,
                model_name=cfg.model_name,
                device=cfg.device,
            )
            model.to(cfg.device)
            (
                training_run_state,
                activations_loader,
                sparse_autoencoder,
                train_contexts,
            ) = load_checkpoint(
                checkpoint_path=checkpoint_path,
                cfg=cfg,
                model=model,
                batch_size=cfg.train_batch_size,
            )
        # no checkpoints found, don't resume
        except FileNotFoundError:
            print(traceback.format_exc())
            print("failed to find checkpoint to resume from, setting resume to False")
            cfg.resume = False

    if not cfg.resume:
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

    if cfg.log_to_wandb:
        resume = None
        if cfg.resume:
            resume = "allow"
        wandb.init(
            project=cfg.wandb_project,
            config=cast(Any, cfg),
            name=cfg.run_name,
            resume=resume,
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
        sae_group=sparse_autoencoder,  # pyright: ignore [reportPossiblyUnboundVariable]
        activation_store=activations_loader,  # pyright: ignore [reportPossiblyUnboundVariable]
        train_contexts=train_contexts,
        training_run_state=training_run_state,
        batch_size=cfg.train_batch_size,
        n_checkpoints=cfg.n_checkpoints,
        feature_sampling_window=cfg.feature_sampling_window,
        use_wandb=cfg.log_to_wandb,
        wandb_log_frequency=cfg.wandb_log_frequency,
        eval_every_n_wandb_logs=cfg.eval_every_n_wandb_logs,
        autocast=cfg.autocast,
        n_eval_batches=cfg.n_eval_batches,
        n_eval_seqs=cfg.n_eval_seqs,
    ).sae_group

    if cfg.log_to_wandb:
        wandb.finish()

    return sparse_autoencoder
