import contextlib
import os
import pickle
import random
import signal
from dataclasses import dataclass, field, fields
from typing import Any, Optional, cast

import numpy as np
import torch
import wandb
from safetensors.torch import save_file
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule

from sae_lens import __version__
from sae_lens.training.activations_store import ActivationsStore, HfDataset
from sae_lens.training.config import LanguageModelSAERunnerConfig
from sae_lens.training.evals import run_evals
from sae_lens.training.geometric_median import compute_geometric_median
from sae_lens.training.optim import L1Scheduler, get_lr_scheduler
from sae_lens.training.sae_group import SparseAutoencoderDictionary
from sae_lens.training.sparse_autoencoder import (
    SAE_CFG_PATH,
    SAE_WEIGHTS_PATH,
    SPARSITY_PATH,
    SparseAutoencoder,
)

# used to map between parameters which are updated during finetuning and the config str.
FINETUNING_PARAMETERS = {
    "scale": ["scaling_factor"],
    "decoder": ["scaling_factor", "W_dec", "b_dec"],
    "unrotated_decoder": ["scaling_factor", "b_dec"],
}


def _log_feature_sparsity(
    feature_sparsity: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    return torch.log10(feature_sparsity + eps).detach().cpu()


@dataclass
class SAETrainContext:
    """
    Context to track during training for a single SAE
    """

    act_freq_scores: torch.Tensor
    n_forward_passes_since_fired: torch.Tensor
    n_frac_active_tokens: int
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    l1_scheduler: L1Scheduler
    finetuning: bool = False

    @property
    def feature_sparsity(self) -> torch.Tensor:
        return self.act_freq_scores / self.n_frac_active_tokens

    @property
    def log_feature_sparsity(self) -> torch.Tensor:
        return _log_feature_sparsity(self.feature_sparsity)

    def begin_finetuning(self, sae: SparseAutoencoder):

        # finetuning method should be set in the config
        # if not, then we don't finetune
        if not isinstance(sae.cfg.finetuning_method, str):
            return

        for name, param in sae.named_parameters():
            if name in FINETUNING_PARAMETERS[sae.cfg.finetuning_method]:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.finetuning = True

    def state_dict(self) -> dict[str, torch.Tensor]:
        state_dict = {}
        for attr in fields(self):
            value = getattr(self, attr.name)
            # serializable fields
            if hasattr(value, "state_dict"):
                state_dict[attr.name] = value.state_dict()
            else:
                state_dict[attr.name] = value
        return state_dict

    @classmethod
    def load(cls, path: str, sae: SparseAutoencoder, total_training_steps: int):
        with open(path, "rb") as f:
            state_dict = pickle.load(f)
        attached_ctx = _build_train_context(
            sae=sae, total_training_steps=total_training_steps
        )
        for attr in fields(attached_ctx):
            value = getattr(attached_ctx, attr.name)
            # optimizer and scheduler, this attaches them properly
            if hasattr(value, "state_dict"):
                value.load_state_dict(state_dict[attr.name])
                state_dict[attr.name] = value
        ctx = cls(**state_dict)  # pyright: ignore [reportArgumentType]
        # if fine tuning, we need to set sae requires grad properly
        if ctx.finetuning:
            ctx.begin_finetuning(sae=sae)
        return ctx

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)


@dataclass
class SAETrainingRunState:
    """
    Training run state for all SAES
    includes n_training_steps
    n_training_tokens
    started_fine_tuning
    and rng states
    """

    n_training_steps: int = 0
    n_training_tokens: int = 0
    started_fine_tuning: bool = False
    checkpoint_paths: list[str] = field(default_factory=list)
    torch_state: Optional[torch.Tensor] = None
    torch_cuda_state: Optional[list[torch.Tensor]] = None
    numpy_state: Optional[
        dict[str, Any]
        | tuple[str, np.ndarray[Any, np.dtype[np.uint32]], int, int, float]
    ] = None
    random_state: Optional[Any] = None

    def __post_init__(self):
        if self.torch_state is None:
            self.torch_state = torch.get_rng_state()
        if self.torch_cuda_state is None:
            self.torch_cuda_state = torch.cuda.get_rng_state_all()
        if self.numpy_state is None:
            self.numpy_state = np.random.get_state()
        if self.random_state is None:
            self.random_state = random.getstate()

    def set_random_state(self):
        assert self.torch_state is not None
        torch.random.set_rng_state(self.torch_state)
        assert self.torch_cuda_state is not None
        torch.cuda.set_rng_state_all(self.torch_cuda_state)
        assert self.numpy_state is not None
        np.random.set_state(self.numpy_state)
        assert self.random_state is not None
        random.setstate(self.random_state)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            attr_dict = pickle.load(f)
        return cls(**attr_dict)

    def save(self, path: str):
        attr_dict = {**self.__dict__}
        with open(path, "wb") as f:
            pickle.dump(attr_dict, f)


@dataclass
class TrainSAEGroupOutput:
    sae_group: SparseAutoencoderDictionary
    checkpoint_paths: list[str]
    log_feature_sparsities: dict[str, torch.Tensor]


def train_sae_on_language_model(
    model: HookedRootModule,
    sae_group: SparseAutoencoderDictionary,
    activation_store: ActivationsStore,
    batch_size: int = 1024,
    n_checkpoints: int = 0,
    feature_sampling_window: int = 1000,  # how many training steps between resampling the features / considiring neurons dead
    dead_feature_threshold: float = 1e-8,  # how infrequently a feature has to be active to be considered dead
    use_wandb: bool = False,
    wandb_log_frequency: int = 50,
    eval_every_n_wandb_logs: int = 100,
    autocast: bool = False,
    n_eval_batches: int = 10,
    n_eval_seqs: int | None = None,
) -> SparseAutoencoderDictionary:
    """
    @deprecated Use `train_sae_group_on_language_model` instead. This method is kept for backward compatibility.
    """
    return train_sae_group_on_language_model(
        model=model,
        sae_group=sae_group,
        activation_store=activation_store,
        batch_size=batch_size,
        n_checkpoints=n_checkpoints,
        feature_sampling_window=feature_sampling_window,
        use_wandb=use_wandb,
        wandb_log_frequency=wandb_log_frequency,
        eval_every_n_wandb_logs=eval_every_n_wandb_logs,
        autocast=autocast,
        n_eval_batches=n_eval_batches,
        n_eval_seqs=n_eval_seqs,
    ).sae_group


def get_total_training_tokens(sae_group: SparseAutoencoderDictionary) -> int:
    return sae_group.cfg.training_tokens + sae_group.cfg.finetuning_tokens


def train_sae_group_on_language_model(
    model: HookedRootModule,
    sae_group: SparseAutoencoderDictionary,
    activation_store: ActivationsStore,
    train_contexts: Optional[dict[str, SAETrainContext]] = None,
    training_run_state: Optional[SAETrainingRunState] = None,
    batch_size: int = 1024,
    n_checkpoints: int = 0,
    feature_sampling_window: int = 1000,  # how many training steps between resampling the features / considiring neurons dead
    use_wandb: bool = False,
    wandb_log_frequency: int = 50,
    eval_every_n_wandb_logs: int = 100,
    autocast: bool = False,
    n_eval_batches: int = 10,
    n_eval_seqs: int | None = None,
) -> TrainSAEGroupOutput:
    total_training_tokens = get_total_training_tokens(sae_group=sae_group)
    _update_sae_lens_training_version(sae_group)
    total_training_steps = total_training_tokens // batch_size

    checkpoint_thresholds = []
    if n_checkpoints > 0:
        checkpoint_thresholds = list(
            range(0, total_training_tokens, total_training_tokens // n_checkpoints)
        )[1:]

    all_layers = sae_group.cfg.hook_point_layer
    if not isinstance(all_layers, list):
        all_layers = [all_layers]

    pbar = tqdm(total=total_training_tokens, desc="Training SAE")

    # not resuming
    if training_run_state is None and train_contexts is None:
        train_contexts = {
            name: _build_train_context(sae, total_training_steps)
            for name, sae in sae_group.autoencoders.items()
        }
        training_run_state = SAETrainingRunState()
        _init_sae_group_b_decs(sae_group, activation_store, all_layers)
    # resuming
    else:
        if train_contexts is None:
            raise ValueError(
                "train_contexts is None, when resuming, pass in training_run_state and train_contexts"
            )
        if training_run_state is None:
            raise ValueError(
                "training_run_state is None, when resuming, pass in training_run_state and train_contexts"
            )
        pbar.update(training_run_state.n_training_tokens)
        training_run_state.set_random_state()

    class InterruptedException(Exception):
        pass

    def interrupt_callback(sig_num: Any, stack_frame: Any):
        raise InterruptedException()

    try:
        # signal handlers (if preempted)
        signal.signal(signal.SIGINT, interrupt_callback)
        signal.signal(signal.SIGTERM, interrupt_callback)

        while training_run_state.n_training_tokens < total_training_tokens:
            # Do a training step.
            layer_acts = activation_store.next_batch()
            training_run_state.n_training_tokens += batch_size

            mse_losses: list[torch.Tensor] = []
            l1_losses: list[torch.Tensor] = []

            for name, sparse_autoencoder in sae_group.autoencoders.items():
                ctx = train_contexts[name]
                wandb_suffix = _wandb_log_suffix(sae_group.cfg, sparse_autoencoder.cfg)
                step_output = _train_step(
                    sparse_autoencoder=sparse_autoencoder,
                    layer_acts=layer_acts,
                    ctx=ctx,
                    feature_sampling_window=feature_sampling_window,
                    use_wandb=use_wandb,
                    n_training_steps=training_run_state.n_training_steps,
                    all_layers=all_layers,
                    batch_size=batch_size,
                    wandb_suffix=wandb_suffix,
                    autocast=autocast,
                )
                mse_losses.append(step_output.mse_loss)
                l1_losses.append(step_output.l1_loss)

                if use_wandb:
                    with torch.no_grad():
                        if (
                            training_run_state.n_training_steps + 1
                        ) % wandb_log_frequency == 0:
                            wandb.log(
                                _build_train_step_log_dict(
                                    sparse_autoencoder,
                                    step_output,
                                    ctx,
                                    wandb_suffix,
                                    training_run_state.n_training_tokens,
                                ),
                                step=training_run_state.n_training_steps,
                            )

                        # record loss frequently, but not all the time.
                        if (training_run_state.n_training_steps + 1) % (
                            wandb_log_frequency * eval_every_n_wandb_logs
                        ) == 0:
                            sparse_autoencoder.eval()
                            run_evals(
                                sparse_autoencoder,
                                activation_store,
                                model,
                                training_run_state.n_training_steps,
                                suffix=wandb_suffix,
                                n_eval_batches=n_eval_batches,
                                n_eval_seqs=n_eval_seqs,
                            )
                            sparse_autoencoder.train()

            # checkpoint if at checkpoint frequency
            if (
                checkpoint_thresholds
                and training_run_state.n_training_tokens > checkpoint_thresholds[0]
            ):
                _save_checkpoint(
                    sae_group,
                    activation_store=activation_store,
                    train_contexts=train_contexts,
                    training_run_state=training_run_state,
                    checkpoint_name=training_run_state.n_training_tokens,
                )
                checkpoint_thresholds.pop(0)

            ###############

            training_run_state.n_training_steps += 1
            if training_run_state.n_training_steps % 100 == 0:
                pbar.set_description(
                    f"{training_run_state.n_training_steps}| MSE Loss {torch.stack(mse_losses).mean().item():.3f} | L1 {torch.stack(l1_losses).mean().item():.3f}"
                )
            pbar.update(batch_size)

            ### If n_training_tokens > sae_group.cfg.training_tokens, then we should switch to fine-tuning (if we haven't already)
            if (not training_run_state.started_fine_tuning) and (
                training_run_state.n_training_tokens > sae_group.cfg.training_tokens
            ):
                training_run_state.started_fine_tuning = True
                for name, sparse_autoencoder in sae_group.autoencoders.items():
                    ctx = train_contexts[name]
                    # this should turn grads on for the scaling factor and other parameters.
                    ctx.begin_finetuning(sae_group.autoencoders[name])

    except (KeyboardInterrupt, InterruptedException):
        print("interrupted, saving progress")
        checkpoint_name = training_run_state.n_training_tokens
        _save_checkpoint(
            sae_group,
            activation_store=activation_store,
            train_contexts=train_contexts,
            training_run_state=training_run_state,
            checkpoint_name=checkpoint_name,
        )
        print("done saving")
        raise
    # save final sae group to checkpoints folder
    _save_checkpoint(
        sae_group,
        activation_store=activation_store,
        train_contexts=train_contexts,
        training_run_state=training_run_state,
        checkpoint_name=f"final_{training_run_state.n_training_tokens}",
        wandb_aliases=["final_model"],
    )

    log_feature_sparsities = {
        name: ctx.log_feature_sparsity for name, ctx in train_contexts.items()
    }

    return TrainSAEGroupOutput(
        sae_group=sae_group,
        checkpoint_paths=training_run_state.checkpoint_paths,
        log_feature_sparsities=log_feature_sparsities,
    )


def _wandb_log_suffix(cfg: Any, hyperparams: Any):
    # Create a mapping from cfg list keys to their corresponding hyperparams attributes
    key_mapping = {
        "hook_point_layer": "layer",
        "l1_coefficient": "coeff",
        "lp_norm": "l",
        "lr": "lr",
    }

    # Generate the suffix by iterating over the keys that have list values in cfg
    suffix = "".join(
        f"_{key_mapping.get(key, key)}{getattr(hyperparams, key, '')}"
        for key, value in vars(cfg).items()
        if isinstance(value, list)
    )
    return suffix


def _build_train_context(
    sae: SparseAutoencoder, total_training_steps: int
) -> SAETrainContext:
    assert not isinstance(sae.cfg.lr, list), "lr must not be a list for a single SAE"
    assert not isinstance(
        sae.cfg.lr_end, list
    ), "lr_end must not be a list for a single SAE"
    assert not isinstance(
        sae.cfg.lr_scheduler_name, list
    ), "lr_scheduler_name must not be a list for a single SAE"
    assert not isinstance(
        sae.cfg.lr_warm_up_steps, list
    ), "lr_warm_up_steps must not be a list for a single SAE"
    assert not isinstance(
        sae.cfg.lr_decay_steps, list
    ), "lr_decay_steps must not be a list for a single SAE"
    assert not isinstance(
        sae.cfg.n_restart_cycles, list
    ), "n_restart_cycles must not be a list for a single SAE"

    act_freq_scores = torch.zeros(
        cast(int, sae.cfg.d_sae),
        device=sae.cfg.device,
    )
    n_forward_passes_since_fired = torch.zeros(
        cast(int, sae.cfg.d_sae),
        device=sae.cfg.device,
    )
    n_frac_active_tokens = 0

    # we don't train the scaling factor (initially)
    # set requires grad to false for the scaling factor
    for name, param in sae.named_parameters():
        if "scaling_factor" in name:
            param.requires_grad = False

    optimizer = Adam(
        sae.parameters(),
        lr=sae.cfg.lr,
        betas=(
            sae.cfg.adam_beta1,  # type: ignore
            sae.cfg.adam_beta2,  # type: ignore
        ),
    )
    assert sae.cfg.lr_end is not None  # this is set in config post-init
    lr_scheduler = get_lr_scheduler(
        sae.cfg.lr_scheduler_name,
        lr=sae.cfg.lr,
        optimizer=optimizer,
        warm_up_steps=sae.cfg.lr_warm_up_steps,
        decay_steps=sae.cfg.lr_decay_steps,
        training_steps=total_training_steps,
        lr_end=sae.cfg.lr_end,
        num_cycles=sae.cfg.n_restart_cycles,
    )

    l1_scheduler = L1Scheduler(
        l1_warm_up_steps=sae.cfg.l1_warm_up_steps,  # type: ignore
        total_steps=total_training_steps,
        sparse_autoencoder=sae,
    )

    return SAETrainContext(
        act_freq_scores=act_freq_scores,
        n_forward_passes_since_fired=n_forward_passes_since_fired,
        n_frac_active_tokens=n_frac_active_tokens,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        l1_scheduler=l1_scheduler,
    )


def _init_sae_group_b_decs(
    sae_group: SparseAutoencoderDictionary,
    activation_store: ActivationsStore,
    all_layers: list[int],
) -> None:
    """
    extract all activations at a certain layer and use for sae b_dec initialization
    """
    geometric_medians = {}
    for _, sae in sae_group:
        hyperparams = sae.cfg
        sae_layer_id = all_layers.index(sae.hook_point_layer)
        if hyperparams.b_dec_init_method == "geometric_median":
            layer_acts = activation_store.storage_buffer.detach()[:, sae_layer_id, :]
            # get geometric median of the activations if we're using those.
            if sae_layer_id not in geometric_medians:
                median = compute_geometric_median(
                    layer_acts,
                    maxiter=100,
                ).median
                geometric_medians[sae_layer_id] = median
            sae.initialize_b_dec_with_precalculated(geometric_medians[sae_layer_id])
        elif hyperparams.b_dec_init_method == "mean":
            layer_acts = activation_store.storage_buffer.detach().cpu()[
                :, sae_layer_id, :
            ]
            sae.initialize_b_dec_with_mean(layer_acts)


def _update_sae_lens_training_version(sae_group: SparseAutoencoderDictionary) -> None:
    """
    Make sure we record the version of SAELens used for the training run
    """
    sae_group.cfg.sae_lens_training_version = __version__
    for sae in sae_group.autoencoders.values():
        sae.cfg.sae_lens_training_version = __version__


@dataclass
class TrainStepOutput:
    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    loss: torch.Tensor
    mse_loss: torch.Tensor
    l1_loss: torch.Tensor
    ghost_grad_loss: torch.Tensor
    ghost_grad_neuron_mask: torch.Tensor


def _train_step(
    sparse_autoencoder: SparseAutoencoder,
    layer_acts: torch.Tensor,
    ctx: SAETrainContext,
    feature_sampling_window: int,  # how many training steps between resampling the features / considiring neurons dead
    use_wandb: bool,
    n_training_steps: int,
    all_layers: list[int],
    batch_size: int,
    wandb_suffix: str,
    autocast: bool = True,
) -> TrainStepOutput:
    assert sparse_autoencoder.cfg.d_sae is not None  # keep pyright happy
    layer_id = all_layers.index(sparse_autoencoder.hook_point_layer)
    sae_in = layer_acts[:, layer_id, :]

    sparse_autoencoder.train()
    # Make sure the W_dec is still zero-norm
    if sparse_autoencoder.normalize_sae_decoder:
        sparse_autoencoder.set_decoder_norm_to_unit_norm()

    # log and then reset the feature sparsity every feature_sampling_window steps
    if (n_training_steps + 1) % feature_sampling_window == 0:
        feature_sparsity = ctx.feature_sparsity
        log_feature_sparsity = _log_feature_sparsity(feature_sparsity)

        if use_wandb:
            wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
            wandb.log(
                {
                    f"metrics/mean_log10_feature_sparsity{wandb_suffix}": log_feature_sparsity.mean().item(),
                    f"plots/feature_density_line_chart{wandb_suffix}": wandb_histogram,
                    f"sparsity/below_1e-5{wandb_suffix}": (feature_sparsity < 1e-5)
                    .sum()
                    .item(),
                    f"sparsity/below_1e-6{wandb_suffix}": (feature_sparsity < 1e-6)
                    .sum()
                    .item(),
                },
                step=n_training_steps,
            )

        ctx.act_freq_scores = torch.zeros(
            sparse_autoencoder.cfg.d_sae, device=sparse_autoencoder.cfg.device
        )
        ctx.n_frac_active_tokens = 0

    ghost_grad_neuron_mask = (
        ctx.n_forward_passes_since_fired > sparse_autoencoder.cfg.dead_feature_window
    ).bool()

    # Setup autocast if using
    scaler = torch.cuda.amp.GradScaler(enabled=autocast)
    if autocast:
        autocast_if_enabled = torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=autocast,
        )
    else:
        autocast_if_enabled = contextlib.nullcontext()

    # Forward and Backward Passes
    # for documentation on autocasting see:
    # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    with autocast_if_enabled:
        (
            sae_out,
            feature_acts,
            loss,
            mse_loss,
            l1_loss,
            ghost_grad_loss,
        ) = sparse_autoencoder(
            sae_in,
            ghost_grad_neuron_mask,
        )

    did_fire = (feature_acts > 0).float().sum(-2) > 0
    ctx.n_forward_passes_since_fired += 1
    ctx.n_forward_passes_since_fired[did_fire] = 0

    with torch.no_grad():
        # Calculate the sparsities, and add it to a list, calculate sparsity metrics
        ctx.act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
        ctx.n_frac_active_tokens += batch_size

    # Scaler will rescale gradients if autocast is enabled
    scaler.scale(loss).backward()  # loss.backward() if not autocasting
    scaler.unscale_(ctx.optimizer)  # needed to clip correctly
    # TODO: Work out if grad norm clipping should be in config / how to test it.
    torch.nn.utils.clip_grad_norm_(sparse_autoencoder.parameters(), 1.0)
    scaler.step(ctx.optimizer)  # just ctx.optimizer.step() if not autocasting
    scaler.update()

    if sparse_autoencoder.normalize_sae_decoder:
        sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()

    ctx.optimizer.zero_grad()

    ctx.lr_scheduler.step()
    ctx.l1_scheduler.step()

    return TrainStepOutput(
        sae_in=sae_in,
        sae_out=sae_out,
        feature_acts=feature_acts,
        loss=loss,
        mse_loss=mse_loss,
        l1_loss=l1_loss,
        ghost_grad_loss=ghost_grad_loss,
        ghost_grad_neuron_mask=ghost_grad_neuron_mask,
    )


def _build_train_step_log_dict(
    sparse_autoencoder: SparseAutoencoder,
    output: TrainStepOutput,
    ctx: SAETrainContext,
    wandb_suffix: str,
    n_training_tokens: int,
) -> dict[str, Any]:
    sae_in = output.sae_in
    sae_out = output.sae_out
    feature_acts = output.feature_acts
    mse_loss = output.mse_loss
    l1_loss = output.l1_loss
    ghost_grad_loss = output.ghost_grad_loss
    loss = output.loss
    ghost_grad_neuron_mask = output.ghost_grad_neuron_mask

    # metrics for currents acts
    l0 = (feature_acts > 0).float().sum(-1).mean()
    current_learning_rate = ctx.optimizer.param_groups[0]["lr"]

    per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
    total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
    explained_variance = 1 - per_token_l2_loss / total_variance

    if isinstance(ghost_grad_loss, torch.Tensor):
        ghost_grad_loss = ghost_grad_loss.item()
    return {
        # losses
        f"losses/mse_loss{wandb_suffix}": mse_loss.item(),
        f"losses/l1_loss{wandb_suffix}": l1_loss.item()
        / sparse_autoencoder.l1_coefficient,  # normalize by l1 coefficient
        f"losses/ghost_grad_loss{wandb_suffix}": ghost_grad_loss,
        f"losses/overall_loss{wandb_suffix}": loss.item(),
        # variance explained
        f"metrics/explained_variance{wandb_suffix}": explained_variance.mean().item(),
        f"metrics/explained_variance_std{wandb_suffix}": explained_variance.std().item(),
        f"metrics/l0{wandb_suffix}": l0.item(),
        # sparsity
        f"sparsity/mean_passes_since_fired{wandb_suffix}": ctx.n_forward_passes_since_fired.mean().item(),
        f"sparsity/dead_features{wandb_suffix}": ghost_grad_neuron_mask.sum().item(),
        f"details/current_learning_rate{wandb_suffix}": current_learning_rate,
        f"details/current_l1_coefficient{wandb_suffix}": sparse_autoencoder.l1_coefficient,
        "details/n_training_tokens": n_training_tokens,
    }


ACTIVATION_STORE_PATH = "activation_store.safetensors"
TRAINING_RUN_STATE_PATH = "training_run_state.pkl"
SAE_CONTEXT_PATH = "ctx.safetensors"


def load_checkpoint(
    checkpoint_path: str,
    cfg: LanguageModelSAERunnerConfig,
    model: HookedRootModule,
    batch_size: int,
    dataset: HfDataset | None = None,
) -> tuple[
    SAETrainingRunState,
    ActivationsStore,
    SparseAutoencoderDictionary,
    dict[str, SAETrainContext],
]:
    training_run_state_path = f"{checkpoint_path}/{TRAINING_RUN_STATE_PATH}"
    training_run_state = SAETrainingRunState.load(training_run_state_path)

    activation_store_path = f"{checkpoint_path}/{ACTIVATION_STORE_PATH}"
    activation_store = ActivationsStore.load(
        activation_store_path, model=model, cfg=cfg, dataset=dataset
    )

    sae_group = SparseAutoencoderDictionary.load_from_pretrained(
        checkpoint_path, device=str(cfg.device)
    )

    total_training_steps = get_total_training_tokens(sae_group=sae_group) // batch_size

    train_contexts = {}
    for name, sae in sae_group.autoencoders.items():
        path = f"{checkpoint_path}/{name}"
        ctx_path = f"{path}/{SAE_CONTEXT_PATH}"
        train_contexts[name] = SAETrainContext.load(
            ctx_path, sae=sae, total_training_steps=total_training_steps
        )

    # overwrite sae gruop cfg with our new cfg in case we want to change things
    sae_group.cfg = cfg
    # TODO: individual saes don't get new cfgs, maybe they should idk its messy bc of _init_autoencoders stuff
    return training_run_state, activation_store, sae_group, train_contexts


def _save_checkpoint(
    sae_group: SparseAutoencoderDictionary,
    activation_store: ActivationsStore,
    train_contexts: dict[str, SAETrainContext],
    training_run_state: SAETrainingRunState,
    checkpoint_name: int | str,
    wandb_aliases: list[str] | None = None,
) -> str:

    checkpoint_path = f"{sae_group.cfg.checkpoint_path}/{checkpoint_name}"
    training_run_state.checkpoint_paths.append(checkpoint_path)
    os.makedirs(checkpoint_path, exist_ok=True)

    training_run_state_path = f"{checkpoint_path}/{TRAINING_RUN_STATE_PATH}"
    training_run_state.save(training_run_state_path)
    if sae_group.cfg.log_to_wandb:
        training_run_state_artifact = wandb.Artifact(
            f"{sae_group.get_name()}_training_run_state",
            type="training_run_state",
            metadata=dict(sae_group.cfg.__dict__),
        )
        training_run_state_artifact.add_file(training_run_state_path)
        # TODO: should these have aliases=wandb_aliases?
        wandb.log_artifact(training_run_state_artifact)

    activation_store_path = f"{checkpoint_path}/{ACTIVATION_STORE_PATH}"
    activation_store.save(activation_store_path)
    if sae_group.cfg.log_to_wandb and sae_group.cfg.log_activations_store_to_wandb:
        activation_store_artifact = wandb.Artifact(
            f"{sae_group.get_name()}_activations_store",
            type="activation_store",
            metadata=dict(sae_group.cfg.__dict__),
        )
        activation_store_artifact.add_file(activation_store_path)
        wandb.log_artifact(activation_store_artifact)

    for name, sae in sae_group.autoencoders.items():

        ctx = train_contexts[name]
        path = f"{checkpoint_path}/{name}"
        os.makedirs(path, exist_ok=True)
        ctx_path = f"{path}/{SAE_CONTEXT_PATH}"
        ctx.save(ctx_path)

        if sae.normalize_sae_decoder:
            sae.set_decoder_norm_to_unit_norm()
        sae.save_model(path)
        log_feature_sparsities = {"sparsity": ctx.log_feature_sparsity}

        log_feature_sparsity_path = f"{path}/{SPARSITY_PATH}"
        save_file(log_feature_sparsities, log_feature_sparsity_path)

        if sae_group.cfg.log_to_wandb and os.path.exists(log_feature_sparsity_path):
            model_artifact = wandb.Artifact(
                f"{sae_group.get_name()}",
                type="model",
                metadata=dict(sae_group.cfg.__dict__),
            )
            model_artifact.add_file(f"{path}/{SAE_WEIGHTS_PATH}")
            model_artifact.add_file(f"{path}/{SAE_CFG_PATH}")
            if sae_group.cfg.log_optimizer_state_to_wandb:
                model_artifact.add_file(ctx_path)
            wandb.log_artifact(model_artifact, aliases=wandb_aliases)

            sparsity_artifact = wandb.Artifact(
                f"{sae_group.get_name()}_log_feature_sparsity",
                type="log_feature_sparsity",
                metadata=dict(sae_group.cfg.__dict__),
            )
            sparsity_artifact.add_file(log_feature_sparsity_path)
            wandb.log_artifact(sparsity_artifact)

    return checkpoint_path
