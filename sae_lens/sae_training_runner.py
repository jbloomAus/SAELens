import json
import os
import signal
from typing import Any, cast

import torch
import wandb
from safetensors.torch import save_file

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.geometric_median import compute_geometric_median
from sae_lens.training.load_model import load_model
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.sparse_autoencoder import (
    SAE_CFG_PATH,
    SAE_WEIGHTS_PATH,
    SPARSITY_PATH,
    TrainingSparseAutoencoder,
)


class InterruptedException(Exception):
    pass


def interrupt_callback(sig_num: Any, stack_frame: Any):
    raise InterruptedException()


class SAETrainingRunner:

    cfg: LanguageModelSAERunnerConfig
    model: torch.nn.Module
    sae: TrainingSparseAutoencoder
    activations_store: ActivationsStore

    def __init__(self, cfg: LanguageModelSAERunnerConfig):
        self.cfg = cfg

        self.model = load_model(
            self.cfg.model_class_name,
            self.cfg.model_name,
            device=self.cfg.device,
            model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
        )

        self.activations_store = ActivationsStore.from_config(
            self.model,
            self.cfg,
        )

        if self.cfg.from_pretrained_path is not None:
            self.sae = TrainingSparseAutoencoder.load_from_pretrained(
                self.cfg.from_pretrained_path, self.cfg.device  # type: ignore
            )
        else:
            self.sae = TrainingSparseAutoencoder(self.cfg)
            self._init_sae_group_b_decs()

    def run(self):
        """ """

        if self.cfg.log_to_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                config=cast(Any, self.cfg),
                name=self.cfg.run_name,
                id=self.cfg.wandb_id,
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
        if self.cfg.compile_llm:
            self.model = torch.compile(
                self.model,
                mode=self.cfg.llm_compilation_mode,
            )  # type: ignore

        if self.cfg.compile_sae:
            self.sae = torch.compile(
                self.sae, mode=self.cfg.sae_compilation_mode
            )  # type: ignore

        trainer = SAETrainer(
            model=self.model,  # type: ignore
            sae=self.sae,  # type: ignore
            activation_store=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,  # type: ignore
            cfg=self.cfg,
        )

        sparse_autoencoder = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.log_to_wandb:
            wandb.finish()

        return sparse_autoencoder

    def run_trainer_with_interruption_handling(self, trainer: SAETrainer):
        try:
            # signal handlers (if preempted)
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)

            # train SAE
            sparse_autoencoder = trainer.fit()

        except (KeyboardInterrupt, InterruptedException):
            print("interrupted, saving progress")
            checkpoint_name = trainer.n_training_tokens
            self.save_checkpoint(trainer, checkpoint_name=checkpoint_name)
            print("done saving")
            raise

        return sparse_autoencoder

    # TODO: move this into the SAE trainer class.
    def _init_sae_group_b_decs(
        self,
    ) -> None:
        """
        extract all activations at a certain layer and use for sae b_dec initialization
        """

        if self.cfg.b_dec_init_method == "geometric_median":
            layer_acts = self.activations_store.storage_buffer.detach()[:, 0, :]
            # get geometric median of the activations if we're using those.
            median = compute_geometric_median(
                layer_acts,
                maxiter=100,
            ).median
            self.sae.initialize_b_dec_with_precalculated(median)  # type: ignore
        elif self.cfg.b_dec_init_method == "mean":
            layer_acts = self.activations_store.storage_buffer.detach().cpu()[:, 0, :]
            self.sae.initialize_b_dec_with_mean(layer_acts)  # type: ignore

    def save_checkpoint(
        self,
        trainer,  # type: ignore
        checkpoint_name: int | str,
        wandb_aliases: list[str] | None = None,
    ) -> str:

        checkpoint_path = f"{trainer.cfg.checkpoint_path}/{checkpoint_name}"

        os.makedirs(checkpoint_path, exist_ok=True)

        path = f"{checkpoint_path}"
        os.makedirs(path, exist_ok=True)

        if self.sae.normalize_sae_decoder:
            self.sae.set_decoder_norm_to_unit_norm()
        self.sae.save_model(path)

        # let's over write the cfg file with the trainer cfg, which is a super set of the original cfg.
        # and should not cause issues but give us more info about SAEs we trained in SAE Lens.
        config = trainer.cfg.to_dict()
        with open(f"{path}/cfg.json", "w") as f:
            json.dump(config, f)

        log_feature_sparsities = {"sparsity": trainer.log_feature_sparsity}

        log_feature_sparsity_path = f"{path}/{SPARSITY_PATH}"
        save_file(log_feature_sparsities, log_feature_sparsity_path)

        if trainer.cfg.log_to_wandb and os.path.exists(log_feature_sparsity_path):
            model_artifact = wandb.Artifact(
                f"{self.sae.get_name()}",
                type="model",
                metadata=dict(trainer.cfg.__dict__),
            )
            model_artifact.add_file(f"{path}/{SAE_WEIGHTS_PATH}")
            model_artifact.add_file(f"{path}/{SAE_CFG_PATH}")

            wandb.log_artifact(model_artifact, aliases=wandb_aliases)

            sparsity_artifact = wandb.Artifact(
                f"{self.sae.get_name()}_log_feature_sparsity",
                type="log_feature_sparsity",
                metadata=dict(trainer.cfg.__dict__),
            )
            sparsity_artifact.add_file(log_feature_sparsity_path)
            wandb.log_artifact(sparsity_artifact)

        return checkpoint_path
