import os
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import torch
import wandb

from sae_lens import __version__

DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}


@dataclass
class LanguageModelSAERunnerConfig:
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    # Data Generating Function (Model + Training Distibuion)
    model_name: str = "gelu-2l"
    model_class_name: str = "HookedTransformer"
    hook_point: str = "blocks.{layer}.hook_mlp_out"
    hook_point_eval: str = "blocks.{layer}.attn.pattern"
    hook_point_layer: int | list[int] = 0
    hook_point_head_index: Optional[int] = None
    dataset_path: str = "NeelNanda/c4-tokenized-2b"
    streaming: bool = True
    is_dataset_tokenized: bool = True
    context_size: int = 128
    use_cached_activations: bool = False
    cached_activations_path: Optional[str] = (
        None  # Defaults to "activations/{dataset}/{model}/{full_hook_name}_{hook_point_head_index}"
    )

    # SAE Parameters
    d_in: int = 512
    d_sae: Optional[int] = None
    b_dec_init_method: str = "geometric_median"
    expansion_factor: int | list[int] = 4
    activation_fn: str = "relu"  # relu, tanh-relu
    normalize_sae_decoder: bool = True
    noise_scale: float = 0.0
    from_pretrained_path: Optional[str] = None
    apply_b_dec_to_input: bool = True
    decoder_orthogonal_init: bool = False
    decoder_heuristic_init: bool = False
    init_encoder_as_decoder_transpose: bool = False

    # Activation Store Parameters
    n_batches_in_buffer: int = 20
    training_tokens: int = 2_000_000
    finetuning_tokens: int = 0
    store_batch_size: int = 32
    train_batch_size: int = 4096
    normalize_activations: bool = False

    # Misc
    device: str | torch.device = "cpu"
    seed: int = 42
    dtype: str | torch.dtype = "float32"  # type: ignore #
    prepend_bos: bool = True

    # Performance - see compilation section of lm_runner.py for info
    autocast: bool = False  # autocast to autocast_dtype during training
    compile_llm: bool = False  # use torch.compile on the LLM
    llm_compilation_mode: str | None = None  # which torch.compile mode to use
    compile_sae: bool = False  # use torch.compile on the SAE
    sae_compilation_mode: str | None = None

    # Training Parameters

    ## Batch size
    train_batch_size: int = 4096

    ## Adam
    adam_beta1: float | list[float] = 0
    adam_beta2: float | list[float] = 0.999

    ## Loss Function
    mse_loss_normalization: Optional[str] = None
    l1_coefficient: float | list[float] = 1e-3
    lp_norm: float | list[float] = 1
    scale_sparsity_penalty_by_decoder_norm: bool = False
    l1_warm_up_steps: int | list[int] = 0

    ## Learning Rate Schedule
    lr: float | list[float] = 3e-4
    lr_scheduler_name: str | list[str] = (
        "constant"  # constant, cosineannealing, cosineannealingwarmrestarts
    )
    lr_warm_up_steps: int | list[int] = 500
    lr_end: float | list[float] | None = (
        None  # only used for cosine annealing, default is lr / 10
    )
    lr_decay_steps: int | list[int] = 0
    n_restart_cycles: int | list[int] = 1  # used only for cosineannealingwarmrestarts

    ## FineTuning
    finetuning_method: Optional[str] = None  # scale, decoder or unrotated_decoder

    # Resampling protocol args
    use_ghost_grads: bool | list[bool] = (
        False  # want to change this to true on some timeline.
    )
    feature_sampling_window: int = 2000
    dead_feature_window: int = 1000  # unless this window is larger feature sampling,

    dead_feature_threshold: float = 1e-8

    # Evals
    n_eval_batches: int = 10
    n_eval_seqs: int | None = None  # useful if evals cause OOM

    # WANDB
    log_to_wandb: bool = True
    log_activations_store_to_wandb: bool = False
    log_optimizer_state_to_wandb: bool = False
    wandb_project: str = "mats_sae_training_language_model"
    wandb_id: Optional[str] = None
    run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_log_frequency: int = 10
    eval_every_n_wandb_logs: int = 100  # logs every 1000 steps.

    # Misc
    resume: bool = False
    n_checkpoints: int = 0
    checkpoint_path: str = "checkpoints"
    verbose: bool = True
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    sae_lens_version: str = field(default_factory=lambda: __version__)
    sae_lens_training_version: str = field(default_factory=lambda: __version__)

    def __post_init__(self):
        if self.use_cached_activations and self.cached_activations_path is None:
            self.cached_activations_path = _default_cached_activations_path(
                self.dataset_path,
                self.model_name,
                self.hook_point,
                self.hook_point_head_index,
            )

        if not isinstance(self.expansion_factor, list):
            self.d_sae = self.d_in * self.expansion_factor
        self.tokens_per_buffer = (
            self.train_batch_size * self.context_size * self.n_batches_in_buffer
        )

        if self.run_name is None:
            self.run_name = f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.training_tokens:3.3e}"

        if self.b_dec_init_method not in ["geometric_median", "mean", "zeros"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.b_dec_init_method}"
            )

        if self.normalize_sae_decoder and self.decoder_heuristic_init:
            raise ValueError(
                "You can't normalize the decoder and use heuristic initialization."
            )

        if self.normalize_sae_decoder and self.scale_sparsity_penalty_by_decoder_norm:
            raise ValueError(
                "Weighting loss by decoder norm makes no sense if you are normalizing the decoder weight norms to 1"
            )

        if isinstance(self.dtype, str) and self.dtype not in DTYPE_MAP:
            raise ValueError(
                f"dtype must be one of {list(DTYPE_MAP.keys())}. Got {self.dtype}"
            )
        elif isinstance(self.dtype, str):
            self.dtype: torch.dtype = DTYPE_MAP[self.dtype]

        # if we use decoder fine tuning, we can't be applying b_dec to the input
        if (self.finetuning_method == "decoder") and (self.apply_b_dec_to_input):
            raise ValueError(
                "If we are fine tuning the decoder, we can't be applying b_dec to the input.\nSet apply_b_dec_to_input to False."
            )

        self.device: str | torch.device = torch.device(self.device)

        if self.lr_end is None:
            if isinstance(self.lr, list):
                self.lr_end = [lr / 10 for lr in self.lr]
            else:
                self.lr_end = self.lr / 10
        unique_id = self.wandb_id
        if unique_id is None:
            unique_id = cast(
                Any, wandb
            ).util.generate_id()  # not sure why this type is erroring
        self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}"

        if self.verbose:
            print(
                f"Run name: {self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.training_tokens:3.3e}"
            )
            # Print out some useful info:
            n_tokens_per_buffer = (
                self.store_batch_size * self.context_size * self.n_batches_in_buffer
            )
            print(f"n_tokens_per_buffer (millions): {n_tokens_per_buffer / 10 ** 6}")
            n_contexts_per_buffer = self.store_batch_size * self.n_batches_in_buffer
            print(
                f"Lower bound: n_contexts_per_buffer (millions): {n_contexts_per_buffer / 10 ** 6}"
            )

            total_training_steps = (
                self.training_tokens + self.finetuning_tokens
            ) // self.train_batch_size
            print(f"Total training steps: {total_training_steps}")

            total_wandb_updates = total_training_steps // self.wandb_log_frequency
            print(f"Total wandb updates: {total_wandb_updates}")

            # how many times will we sample dead neurons?
            # assert self.dead_feature_window <= self.feature_sampling_window, "dead_feature_window must be smaller than feature_sampling_window"
            n_feature_window_samples = (
                total_training_steps // self.feature_sampling_window
            )
            print(
                f"n_tokens_per_feature_sampling_window (millions): {(self.feature_sampling_window * self.context_size * self.train_batch_size) / 10 ** 6}"
            )
            print(
                f"n_tokens_per_dead_feature_window (millions): {(self.dead_feature_window * self.context_size * self.train_batch_size) / 10 ** 6}"
            )
            print(
                f"We will reset the sparsity calculation {n_feature_window_samples} times."
            )
            # print("Number tokens in dead feature calculation window: ", self.dead_feature_window * self.train_batch_size)
            print(
                f"Number tokens in sparsity calculation window: {self.feature_sampling_window * self.train_batch_size:.2e}"
            )

        if not isinstance(self.use_ghost_grads, list) and self.use_ghost_grads:
            print("Using Ghost Grads.")

    def get_checkpoints_by_step(self) -> tuple[dict[int, str], bool]:
        """
        Returns (dict, is_done)
        where dict is [steps] = path
        for each checkpoint, and
        is_done is True if there is a "final_{steps}" checkpoint
        """
        is_done = False
        checkpoints = [
            f
            for f in os.listdir(self.checkpoint_path)
            if os.path.isdir(os.path.join(self.checkpoint_path, f))
        ]
        mapped_to_steps = {}
        for c in checkpoints:
            try:
                steps = int(c)
            except ValueError:
                if c.startswith("final"):
                    steps = int(c.split("_")[1])
                    is_done = True
                else:
                    continue  # ignore this directory
            full_path = os.path.join(self.checkpoint_path, c)
            mapped_to_steps[steps] = full_path
        return mapped_to_steps, is_done

    def get_resume_checkpoint_path(self) -> str:
        """
        Gets the checkpoint path with the most steps
        raises StopIteration if the model is done (there is a final_{steps} directoryh
        raises FileNotFoundError if there are no checkpoints found
        """
        mapped_to_steps, is_done = self.get_checkpoints_by_step()
        if is_done:
            raise StopIteration("Finished training model")
        if len(mapped_to_steps) == 0:
            raise FileNotFoundError("no checkpoints available to resume from")
        else:
            max_step = max(list(mapped_to_steps.keys()))
            checkpoint_dir = mapped_to_steps[max_step]
            print(f"resuming from step {max_step} at path {checkpoint_dir}")
            return mapped_to_steps[max_step]


@dataclass
class CacheActivationsRunnerConfig:
    """
    Configuration for caching activations of an LLM.
    """

    # Data Generating Function (Model + Training Distibuion)
    model_name: str = "gelu-2l"
    model_class_name: str = "HookedTransformer"
    hook_point: str = "blocks.{layer}.hook_mlp_out"
    hook_point_layer: int | list[int] = 0
    hook_point_head_index: Optional[int] = None
    dataset_path: str = "NeelNanda/c4-tokenized-2b"
    streaming: bool = True
    is_dataset_tokenized: bool = True
    context_size: int = 128
    new_cached_activations_path: Optional[str] = (
        None  # Defaults to "activations/{dataset}/{model}/{full_hook_name}_{hook_point_head_index}"
    )
    # dont' specify this since you don't want to load from disk with the cache runner.
    cached_activations_path: Optional[str] = None
    # SAE Parameters
    d_in: int = 512

    # Activation Store Parameters
    n_batches_in_buffer: int = 20
    training_tokens: int = 2_000_000
    store_batch_size: int = 32
    train_batch_size: int = 4096
    normalize_activations: bool = False

    # Misc
    device: str | torch.device = "cpu"
    seed: int = 42
    dtype: str | torch.dtype = "float32"
    prepend_bos: bool = True

    # Activation caching stuff
    shuffle_every_n_buffers: int = 10
    n_shuffles_with_last_section: int = 10
    n_shuffles_in_entire_dir: int = 10
    n_shuffles_final: int = 100
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Autofill cached_activations_path unless the user overrode it
        if self.new_cached_activations_path is None:
            self.new_cached_activations_path = _default_cached_activations_path(
                self.dataset_path,
                self.model_name,
                self.hook_point,
                self.hook_point_head_index,
            )


def _default_cached_activations_path(
    dataset_path: str,
    model_name: str,
    hook_point: str,
    hook_point_head_index: int | None,
) -> str:
    path = f"activations/{dataset_path.replace('/', '_')}/{model_name.replace('/', '_')}/{hook_point}"
    if hook_point_head_index is not None:
        path += f"_{hook_point_head_index}"
    return path
