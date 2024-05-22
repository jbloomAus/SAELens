import signal
from typing import Any, cast

import torch
import wandb

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.checkpointing import save_checkpoint
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.geometric_median import compute_geometric_median
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from sae_lens.training.sparse_autoencoder import SparseAutoencoder
from sae_lens.training.train_sae_on_language_model import SAETrainer


class InterruptedException(Exception):
    pass


def interrupt_callback(sig_num: Any, stack_frame: Any):
    raise InterruptedException()


class SAETrainingRunner:

    cfg: LanguageModelSAERunnerConfig
    model: torch.nn.Module
    sparse_autoencoder: SparseAutoencoder
    activations_store: ActivationsStore

    def __init__(self, cfg: LanguageModelSAERunnerConfig):
        self.cfg = cfg

        if self.cfg.from_pretrained_path is not None:
            (
                self.model,
                self.sparse_autoencoder,
                self.activations_store,
            ) = LMSparseAutoencoderSessionloader.load_pretrained_sae(
                self.cfg.from_pretrained_path
            )
            self.cfg = self.sparse_autoencoder.cfg
        else:
            loader = LMSparseAutoencoderSessionloader(self.cfg)
            self.model, self.sparse_autoencoder, self.activations_store = (
                loader.load_sae_training_group_session()
            )
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
            self.sparse_autoencoder = torch.compile(
                self.sparse_autoencoder, mode=self.cfg.sae_compilation_mode
            )  # type: ignore

        trainer = SAETrainer(
            model=self.model,  # type: ignore
            sae=self.sparse_autoencoder,  # type: ignore
            activation_store=self.activations_store,
            save_checkpoint_fn=save_checkpoint,  # type: ignore
            cfg=self.cfg,
        )

        try:
            # signal handlers (if preempted)
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)

            # train SAE
            sparse_autoencoder = trainer.fit()

        except (KeyboardInterrupt, InterruptedException):
            print("interrupted, saving progress")
            checkpoint_name = trainer.n_training_tokens
            save_checkpoint(trainer, checkpoint_name=checkpoint_name)
            print("done saving")
            raise

        if self.cfg.log_to_wandb:
            wandb.finish()

        return sparse_autoencoder

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
            self.sparse_autoencoder.initialize_b_dec_with_precalculated(median)  # type: ignore
        elif self.cfg.b_dec_init_method == "mean":
            layer_acts = self.activations_store.storage_buffer.detach().cpu()[:, 0, :]
            self.sparse_autoencoder.initialize_b_dec_with_mean(layer_acts)  # type: ignore
