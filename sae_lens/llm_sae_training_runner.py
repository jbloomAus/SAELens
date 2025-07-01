import json
import signal
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, cast

import torch
import wandb
from simple_parsing import ArgumentParser
from transformer_lens.hook_points import HookedRootModule
from typing_extensions import deprecated

from sae_lens import logger
from sae_lens.config import HfDataset, LanguageModelSAERunnerConfig
from sae_lens.constants import ACTIVATIONS_STORE_STATE_FILENAME, RUNNER_CFG_FILENAME
from sae_lens.evals import EvalConfig, run_evals
from sae_lens.load_model import load_model
from sae_lens.saes.sae import (
    T_TRAINING_SAE,
    T_TRAINING_SAE_CONFIG,
    TrainingSAE,
    TrainingSAEConfig,
)
from sae_lens.training.activation_scaler import ActivationScaler
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.types import DataProvider


class InterruptedException(Exception):
    pass


def interrupt_callback(sig_num: Any, stack_frame: Any):  # noqa: ARG001
    raise InterruptedException()


@dataclass
class LLMSaeEvaluator(Generic[T_TRAINING_SAE]):
    model: HookedRootModule
    activations_store: ActivationsStore
    eval_batch_size_prompts: int | None
    n_eval_batches: int
    model_kwargs: dict[str, Any]

    def __call__(
        self,
        sae: T_TRAINING_SAE,
        data_provider: DataProvider,
        activation_scaler: ActivationScaler,
    ) -> dict[str, Any]:
        ignore_tokens = set()
        if self.activations_store.exclude_special_tokens is not None:
            ignore_tokens = set(self.activations_store.exclude_special_tokens.tolist())

        eval_config = EvalConfig(
            batch_size_prompts=self.eval_batch_size_prompts,
            n_eval_reconstruction_batches=self.n_eval_batches,
            n_eval_sparsity_variance_batches=self.n_eval_batches,
            compute_ce_loss=True,
            compute_l2_norms=True,
            compute_sparsity_metrics=True,
            compute_variance_metrics=True,
        )

        eval_metrics, _ = run_evals(
            sae=sae,
            activation_store=self.activations_store,
            model=self.model,
            activation_scaler=activation_scaler,
            eval_config=eval_config,
            ignore_tokens=ignore_tokens,
            model_kwargs=self.model_kwargs,
        )  # not calculating featurwise metrics here.

        # Remove eval metrics that are already logged during training
        eval_metrics.pop("metrics/explained_variance", None)
        eval_metrics.pop("metrics/explained_variance_std", None)
        eval_metrics.pop("metrics/l0", None)
        eval_metrics.pop("metrics/l1", None)
        eval_metrics.pop("metrics/mse", None)

        # Remove metrics that are not useful for wandb logging
        eval_metrics.pop("metrics/total_tokens_evaluated", None)

        return eval_metrics


class LanguageModelSAETrainingRunner:
    """
    Class to run the training of a Sparse Autoencoder (SAE) on a TransformerLens model.
    """

    cfg: LanguageModelSAERunnerConfig[Any]
    model: HookedRootModule
    sae: TrainingSAE[Any]
    activations_store: ActivationsStore

    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig[T_TRAINING_SAE_CONFIG],
        override_dataset: HfDataset | None = None,
        override_model: HookedRootModule | None = None,
        override_sae: TrainingSAE[Any] | None = None,
    ):
        if override_dataset is not None:
            logger.warning(
                f"You just passed in a dataset which will override the one specified in your configuration: {cfg.dataset_path}. As a consequence this run will not be reproducible via configuration alone."
            )
        if override_model is not None:
            logger.warning(
                f"You just passed in a model which will override the one specified in your configuration: {cfg.model_name}. As a consequence this run will not be reproducible via configuration alone."
            )

        self.cfg = cfg

        if override_model is None:
            self.model = load_model(
                self.cfg.model_class_name,
                self.cfg.model_name,
                device=self.cfg.device,
                model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
            )
        else:
            self.model = override_model

        self.activations_store = ActivationsStore.from_config(
            self.model,
            self.cfg,
            override_dataset=override_dataset,
        )

        if override_sae is None:
            if self.cfg.from_pretrained_path is not None:
                self.sae = TrainingSAE.load_from_disk(
                    self.cfg.from_pretrained_path, self.cfg.device
                )
            else:
                self.sae = TrainingSAE.from_dict(
                    TrainingSAEConfig.from_dict(
                        self.cfg.get_training_sae_cfg_dict(),
                    ).to_dict()
                )
        else:
            self.sae = override_sae

    def run(self):
        """
        Run the training of the SAE.
        """

        if self.cfg.logger.log_to_wandb:
            wandb.init(
                project=self.cfg.logger.wandb_project,
                entity=self.cfg.logger.wandb_entity,
                config=cast(Any, self.cfg),
                name=self.cfg.logger.run_name,
                id=self.cfg.logger.wandb_id,
            )

        evaluator = LLMSaeEvaluator(
            model=self.model,
            activations_store=self.activations_store,
            eval_batch_size_prompts=self.cfg.eval_batch_size_prompts,
            n_eval_batches=self.cfg.n_eval_batches,
            model_kwargs=self.cfg.model_kwargs,
        )

        trainer = SAETrainer(
            sae=self.sae,
            data_provider=self.activations_store,
            evaluator=evaluator,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg.to_sae_trainer_config(),
        )

        self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.logger.log_to_wandb:
            wandb.finish()

        return sae

    def _compile_if_needed(self):
        # Compile model and SAE
        #  torch.compile can provide significant speedups (10-20% in testing)
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
            backend = "aot_eager" if self.cfg.device == "mps" else "inductor"

            self.sae.training_forward_pass = torch.compile(  # type: ignore
                self.sae.training_forward_pass,
                mode=self.cfg.sae_compilation_mode,
                backend=backend,
            )  # type: ignore

    def run_trainer_with_interruption_handling(
        self, trainer: SAETrainer[TrainingSAE[TrainingSAEConfig], TrainingSAEConfig]
    ):
        try:
            # signal handlers (if preempted)
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)

            # train SAE
            sae = trainer.fit()

        except (KeyboardInterrupt, InterruptedException):
            logger.warning("interrupted, saving progress")
            checkpoint_path = Path(self.cfg.checkpoint_path) / str(
                trainer.n_training_samples
            )
            self.save_checkpoint(checkpoint_path)
            logger.info("done saving")
            raise

        return sae

    def save_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> None:
        self.activations_store.save(
            str(checkpoint_path / ACTIVATIONS_STORE_STATE_FILENAME)
        )

        runner_config = self.cfg.to_dict()
        with open(checkpoint_path / RUNNER_CFG_FILENAME, "w") as f:
            json.dump(runner_config, f)


def _parse_cfg_args(
    args: Sequence[str],
) -> LanguageModelSAERunnerConfig[TrainingSAEConfig]:
    if len(args) == 0:
        args = ["--help"]
    parser = ArgumentParser(exit_on_error=False)
    parser.add_arguments(LanguageModelSAERunnerConfig, dest="cfg")
    return parser.parse_args(args).cfg


# moved into its own function to make it easier to test
def _run_cli(args: Sequence[str]):
    cfg = _parse_cfg_args(args)
    LanguageModelSAETrainingRunner(cfg=cfg).run()


if __name__ == "__main__":
    _run_cli(args=sys.argv[1:])


@deprecated("Use LanguageModelSAETrainingRunner instead")
class SAETrainingRunner(LanguageModelSAETrainingRunner):
    pass
