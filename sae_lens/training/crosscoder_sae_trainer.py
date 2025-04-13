from typing import Any

import torch
import wandb
from tqdm import tqdm

from sae_lens.evals import run_evals
from sae_lens.training.sae_trainer import SAETrainer, _unwrap_item
from sae_lens.training.training_sae import TrainingSAE, TrainStepOutput

# TODO(mkbehr): probably too much copypasting here

class CrosscoderSAETrainer(SAETrainer):
    def fit(self) -> TrainingSAE:
        pbar = tqdm(total=self.cfg.total_training_tokens, desc="Training SAE")

        self.activations_store.set_norm_scaling_factor_if_needed()

        # Train loop
        while self.n_training_tokens < self.cfg.total_training_tokens:
            # Do a training step.
            layer_acts = self.activations_store.next_batch().to(
                self.sae.device
            )
            self.n_training_tokens += self.cfg.train_batch_size_tokens

            step_output = self._train_step(sae=self.sae, sae_in=layer_acts)

            if self.cfg.log_to_wandb:
                self._log_train_step(step_output)
                self._run_and_log_evals()

            self._checkpoint_if_needed()
            self.n_training_steps += 1
            self._update_pbar(step_output, pbar)

            ### If n_training_tokens > sae_group.cfg.training_tokens, then we should switch to fine-tuning (if we haven't already)
            self._begin_finetuning_if_needed()

        # fold the estimated norm scaling factor into the sae weights
        if self.activations_store.estimated_norm_scaling_factor is not None:
            self.sae.fold_activation_norm_scaling_factor(
                self.activations_store.estimated_norm_scaling_factor
            )
            self.activations_store.estimated_norm_scaling_factor = None

        # save final sae group to checkpoints folder
        self.save_checkpoint(
            trainer=self,
            checkpoint_name=f"final_{self.n_training_tokens}",
            wandb_aliases=["final_model"],
        )

        pbar.close()
        return self.sae

    @torch.no_grad()
    def _build_train_step_log_dict(
        self,
        output: TrainStepOutput,
        n_training_tokens: int,
    ) -> dict[str, Any]:
        sae_in = output.sae_in
        sae_out = output.sae_out
        feature_acts = output.feature_acts
        loss = output.loss.item()

        # metrics for currents acts
        l0 = (feature_acts > 0).float().sum(-1).mean()
        current_learning_rate = self.optimizer.param_groups[0]["lr"]

        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=(-2, -1)).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum((-2, -1))
        explained_variance = 1 - per_token_l2_loss / total_variance

        log_dict = {
            # losses
            "losses/overall_loss": loss,
            # variance explained
            "metrics/explained_variance": explained_variance.mean().item(),
            "metrics/explained_variance_std": explained_variance.std().item(),
            "metrics/l0": l0.item(),
            # sparsity
            "sparsity/mean_passes_since_fired": self.n_forward_passes_since_fired.mean().item(),
            "sparsity/dead_features": self.dead_neurons.sum().item(),
            "details/current_learning_rate": current_learning_rate,
            "details/current_l1_coefficient": self.current_l1_coefficient,
            "details/n_training_tokens": n_training_tokens,
        }
        for loss_name, loss_value in output.losses.items():
            loss_item = _unwrap_item(loss_value)
            # special case for l1 loss, which we normalize by the l1 coefficient
            if loss_name == "l1_loss":
                log_dict[f"losses/{loss_name}"] = (
                    loss_item / self.current_l1_coefficient
                )
                log_dict[f"losses/raw_{loss_name}"] = loss_item
            else:
                log_dict[f"losses/{loss_name}"] = loss_item

        return log_dict

    @torch.no_grad()
    def _run_and_log_evals(self):
        # record loss frequently, but not all the time.
        if (self.n_training_steps + 1) % (
            self.cfg.wandb_log_frequency * self.cfg.eval_every_n_wandb_logs
        ) == 0:
            self.sae.eval()
            ignore_tokens = set()
            if self.activations_store.exclude_special_tokens is not None:
                ignore_tokens = set(
                    self.activations_store.exclude_special_tokens.tolist()
                )
            eval_metrics, _ = run_evals(
                sae=self.sae,
                activation_store=self.activations_store,
                model=self.model,
                eval_config=self.trainer_eval_config,
                ignore_tokens=ignore_tokens,
                model_kwargs=self.cfg.model_kwargs,
            )  # not calculating featurwise metrics here.

            # Remove eval metrics that are already logged during training
            eval_metrics.pop("metrics/explained_variance", None)
            eval_metrics.pop("metrics/explained_variance_std", None)
            eval_metrics.pop("metrics/l0", None)
            eval_metrics.pop("metrics/l1", None)
            eval_metrics.pop("metrics/mse", None)

            # Remove metrics that are not useful for wandb logging
            eval_metrics.pop("metrics/total_tokens_evaluated", None)

            W_dec_norm_dist = self.sae.W_dec.detach().float().norm(dim=(1,2)).cpu().numpy()
            eval_metrics["weights/W_dec_norms"] = wandb.Histogram(W_dec_norm_dist)  # type: ignore

            if self.sae.cfg.architecture == "standard":
                b_e_dist = self.sae.b_enc.detach().float().cpu().numpy()
                eval_metrics["weights/b_e"] = wandb.Histogram(b_e_dist)  # type: ignore
            elif self.sae.cfg.architecture == "gated":
                b_gate_dist = self.sae.b_gate.detach().float().cpu().numpy()
                eval_metrics["weights/b_gate"] = wandb.Histogram(b_gate_dist)  # type: ignore
                b_mag_dist = self.sae.b_mag.detach().float().cpu().numpy()
                eval_metrics["weights/b_mag"] = wandb.Histogram(b_mag_dist)  # type: ignore

            wandb.log(
                eval_metrics,
                step=self.n_training_steps,
            )
            self.sae.train()
