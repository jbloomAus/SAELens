import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

import simple_parsing
import torch
import wandb
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

from sae_lens import __version__, logger
from sae_lens.constants import DTYPE_MAP
from sae_lens.saes.sae import TrainingSAEConfig  # Import the base class

if TYPE_CHECKING:
    # No need to import T_TRAINING_SAE_CONFIG here anymore
    pass
    # from sae_lens.saes.sae import T_TRAINING_SAE_CONFIG

# Define the TypeVar with the bound
T_TRAINING_SAE_CONFIG = TypeVar("T_TRAINING_SAE_CONFIG", bound=TrainingSAEConfig)

HfDataset = DatasetDict | Dataset | IterableDatasetDict | IterableDataset


# calling this "json_dict" so error messages will reference "json_dict" being invalid
def json_dict(s: str) -> Any:
    res = json.loads(s)
    if res is not None and not isinstance(res, dict):
        raise ValueError(f"Expected a dictionary, got {type(res)}")
    return res


def dict_field(default: dict[str, Any] | None, **kwargs: Any) -> Any:  # type: ignore
    """
    Helper to wrap simple_parsing.helpers.dict_field so we can load JSON fields from the command line.
    """
    if default is None:
        return simple_parsing.helpers.field(default=None, type=json_dict, **kwargs)
    return simple_parsing.helpers.dict_field(default, type=json_dict, **kwargs)


@dataclass
class LoggingConfig:
    # WANDB
    log_to_wandb: bool = True
    log_activations_store_to_wandb: bool = False
    log_optimizer_state_to_wandb: bool = False
    wandb_project: str = "sae_lens_training"
    wandb_id: str | None = None
    run_name: str | None = None
    wandb_entity: str | None = None
    wandb_log_frequency: int = 10
    eval_every_n_wandb_logs: int = 100  # logs every 100 steps.

    def log(
        self,
        trainer: Any,  # avoid import cycle from importing SAETrainer
        weights_path: Path | str,
        cfg_path: Path | str,
        sparsity_path: Path | str | None,
        wandb_aliases: list[str] | None = None,
    ) -> None:
        # Avoid wandb saving errors such as:
        #   ValueError: Artifact name may only contain alphanumeric characters, dashes, underscores, and dots. Invalid name: sae_google/gemma-2b_etc
        sae_name = trainer.sae.get_name().replace("/", "__")

        # save model weights and cfg
        model_artifact = wandb.Artifact(
            sae_name,
            type="model",
            metadata=dict(trainer.cfg.__dict__),
        )
        model_artifact.add_file(str(weights_path))
        model_artifact.add_file(str(cfg_path))
        wandb.log_artifact(model_artifact, aliases=wandb_aliases)

        # save log feature sparsity
        sparsity_artifact = wandb.Artifact(
            f"{sae_name}_log_feature_sparsity",
            type="log_feature_sparsity",
            metadata=dict(trainer.cfg.__dict__),
        )
        if sparsity_path is not None:
            sparsity_artifact.add_file(str(sparsity_path))
        wandb.log_artifact(sparsity_artifact)


@dataclass
class LanguageModelSAERunnerConfig(Generic[T_TRAINING_SAE_CONFIG]):
    """
    Configuration for training a sparse autoencoder on a language model.

    Args:
        architecture (str): The architecture to use, either "standard", "gated", "topk", or "jumprelu".
        model_name (str): The name of the model to use. This should be the name of the model in the Hugging Face model hub.
        model_class_name (str): The name of the class of the model to use. This should be either `HookedTransformer` or `HookedMamba`.
        hook_name (str): The name of the hook to use. This should be a valid TransformerLens hook.
        hook_eval (str): NOT CURRENTLY IN USE. The name of the hook to use for evaluation.
        hook_layer (int): The index of the layer to hook. Used to stop forward passes early and speed up processing.
        hook_head_index (int, optional): When the hook if for an activatio with a head index, we can specify a specific head to use here.
        dataset_path (str): A Hugging Face dataset path.
        dataset_trust_remote_code (bool): Whether to trust remote code when loading datasets from Huggingface.
        streaming (bool): Whether to stream the dataset. Streaming large datasets is usually practical.
        is_dataset_tokenized (bool): NOT IN USE. We used to use this but now automatically detect if the dataset is tokenized.
        context_size (int): The context size to use when generating activations on which to train the SAE.
        use_cached_activations (bool): Whether to use cached activations. This is useful when doing sweeps over the same activations.
        cached_activations_path (str, optional): The path to the cached activations.
        d_in (int): The input dimension of the SAE.
        d_sae (int, optional): The output dimension of the SAE. If None, defaults to `d_in * expansion_factor`.
        b_dec_init_method (str): The method to use to initialize the decoder bias. Zeros is likely fine.
        expansion_factor (int): The expansion factor. Larger is better but more computationally expensive. Default is 4.
        activation_fn (str): The activation function to use. Relu is standard.
        normalize_sae_decoder (bool): Whether to normalize the SAE decoder. Unit normed decoder weights used to be preferred.
        noise_scale (float): Using noise to induce sparsity is supported but not recommended.
        from_pretrained_path (str, optional): The path to a pretrained SAE. We can finetune an existing SAE if needed.
        apply_b_dec_to_input (bool): Whether to apply the decoder bias to the input. Not currently advised.
        decoder_orthogonal_init (bool): Whether to use orthogonal initialization for the decoder. Not currently advised.
        decoder_heuristic_init (bool): Whether to use heuristic initialization for the decoder. See Anthropic April Update.
        init_encoder_as_decoder_transpose (bool): Whether to initialize the encoder as the transpose of the decoder. See Anthropic April Update.
        n_batches_in_buffer (int): The number of batches in the buffer. When not using cached activations, a buffer in ram is used. The larger it is, the better shuffled the activations will be.
        training_tokens (int): The number of training tokens.
        finetuning_tokens (int): The number of finetuning tokens. See [here](https://www.lesswrong.com/posts/3JuSjTZyMzaSeTxKk/addressing-feature-suppression-in-saes)
        store_batch_size_prompts (int): The batch size for storing activations. This controls how many prompts are in the batch of the language model when generating actiations.
        train_batch_size_tokens (int): The batch size for training. This controls the batch size of the SAE Training loop.
        normalize_activations (str): Activation Normalization Strategy. Either none, expected_average_only_in (estimate the average activation norm and divide activations by it following Antrhopic April update -> this can be folded post training and set to None), or constant_norm_rescale (at runtime set activation norm to sqrt(d_in) and then scale up the SAE output).
        seqpos_slice (tuple): Determines slicing of activations when constructing batches during training. The slice should be (start_pos, end_pos, optional[step_size]), e.g. for Othello we sometimes use (5, -5). Note, step_size > 0.
        device (str): The device to use. Usually cuda.
        act_store_device (str): The device to use for the activation store. CPU is advised in order to save vram.
        seed (int): The seed to use.
        dtype (str): The data type to use.
        prepend_bos (bool): Whether to prepend the beginning of sequence token. You should use whatever the model was trained with.
        jumprelu_init_threshold (float): The threshold to initialize for training JumpReLU SAEs.
        jumprelu_bandwidth (float): Bandwidth for training JumpReLU SAEs.
        autocast (bool): Whether to use autocast during training. Saves vram.
        autocast_lm (bool): Whether to use autocast during activation fetching.
        compile_llm (bool): Whether to compile the LLM.
        llm_compilation_mode (str): The compilation mode to use for the LLM.
        compile_sae (bool): Whether to compile the SAE.
        sae_compilation_mode (str): The compilation mode to use for the SAE.
        adam_beta1 (float): The beta1 parameter for Adam.
        adam_beta2 (float): The beta2 parameter for Adam.
        mse_loss_normalization (str): The normalization to use for the MSE loss.
        l1_coefficient (float): The L1 coefficient.
        lp_norm (float): The Lp norm.
        scale_sparsity_penalty_by_decoder_norm (bool): Whether to scale the sparsity penalty by the decoder norm.
        l1_warm_up_steps (int): The number of warm-up steps for the L1 loss.
        lr (float): The learning rate.
        lr_scheduler_name (str): The name of the learning rate scheduler to use.
        lr_warm_up_steps (int): The number of warm-up steps for the learning rate.
        lr_end (float): The end learning rate if lr_decay_steps is set. Default is lr / 10.
        lr_decay_steps (int): The number of decay steps for the learning rate.
        n_restart_cycles (int): The number of restart cycles for the cosine annealing warm restarts scheduler.
        finetuning_method (str): The method to use for finetuning.
        use_ghost_grads (bool): Whether to use ghost gradients.
        feature_sampling_window (int): The feature sampling window.
        dead_feature_window (int): The dead feature window.
        dead_feature_threshold (float): The dead feature threshold.
        n_eval_batches (int): The number of evaluation batches.
        eval_batch_size_prompts (int): The batch size for evaluation.
        log_to_wandb (bool): Whether to log to Weights & Biases.
        log_activations_store_to_wandb (bool): NOT CURRENTLY USED. Whether to log the activations store to Weights & Biases.
        log_optimizer_state_to_wandb (bool): NOT CURRENTLY USED. Whether to log the optimizer state to Weights & Biases.
        wandb_project (str): The Weights & Biases project to log to.
        wandb_id (str): The Weights & Biases ID.
        run_name (str): The name of the run.
        wandb_entity (str): The Weights & Biases entity.
        wandb_log_frequency (int): The frequency to log to Weights & Biases.
        eval_every_n_wandb_logs (int): The frequency to evaluate.
        resume (bool): Whether to resume training.
        n_checkpoints (int): The number of checkpoints.
        checkpoint_path (str): The path to save checkpoints.
        verbose (bool): Whether to print verbose output.
        model_kwargs (dict[str, Any]): Additional keyword arguments for the model.
        model_from_pretrained_kwargs (dict[str, Any]): Additional keyword arguments for the model from pretrained.
        exclude_special_tokens (bool | list[int]): Whether to exclude special tokens from the activations.
    """

    sae: T_TRAINING_SAE_CONFIG

    # Data Generating Function (Model + Training Distibuion)
    model_name: str = "gelu-2l"
    model_class_name: str = "HookedTransformer"
    hook_name: str = "blocks.0.hook_mlp_out"
    hook_eval: str = "NOT_IN_USE"
    hook_layer: int = 0
    hook_head_index: int | None = None
    dataset_path: str = ""
    dataset_trust_remote_code: bool = True
    streaming: bool = True
    is_dataset_tokenized: bool = True
    context_size: int = 128
    use_cached_activations: bool = False
    cached_activations_path: str | None = (
        None  # Defaults to "activations/{dataset}/{model}/{full_hook_name}_{hook_head_index}"
    )

    # SAE Parameters
    from_pretrained_path: str | None = None

    # Activation Store Parameters
    n_batches_in_buffer: int = 20
    training_tokens: int = 2_000_000
    store_batch_size_prompts: int = 32
    seqpos_slice: tuple[int | None, ...] = (None,)

    # Misc
    device: str = "cpu"
    act_store_device: str = "with_model"  # will be set by post init if with_model
    seed: int = 42
    dtype: str = "float32"  # type: ignore #
    prepend_bos: bool = True

    # Performance - see compilation section of lm_runner.py for info
    autocast: bool = False  # autocast to autocast_dtype during training
    autocast_lm: bool = False  # autocast lm during activation fetching
    compile_llm: bool = False  # use torch.compile on the LLM
    llm_compilation_mode: str | None = None  # which torch.compile mode to use
    compile_sae: bool = False  # use torch.compile on the SAE
    sae_compilation_mode: str | None = None

    # Training Parameters

    ## Batch size
    train_batch_size_tokens: int = 4096

    ## Adam
    adam_beta1: float = 0.0
    adam_beta2: float = 0.999

    ## Learning Rate Schedule
    lr: float = 3e-4
    lr_scheduler_name: str = (
        "constant"  # constant, cosineannealing, cosineannealingwarmrestarts
    )
    lr_warm_up_steps: int = 0
    lr_end: float | None = None  # only used for cosine annealing, default is lr / 10
    lr_decay_steps: int = 0
    n_restart_cycles: int = 1  # used only for cosineannealingwarmrestarts

    # Resampling protocol args
    dead_feature_window: int = 1000  # unless this window is larger feature sampling,
    feature_sampling_window: int = 2000
    dead_feature_threshold: float = 1e-8

    # Evals
    n_eval_batches: int = 10
    eval_batch_size_prompts: int | None = None  # useful if evals cause OOM

    logger: LoggingConfig = field(default_factory=LoggingConfig)

    # Misc
    n_checkpoints: int = 0
    checkpoint_path: str = "checkpoints"
    verbose: bool = True
    model_kwargs: dict[str, Any] = dict_field(default={})
    model_from_pretrained_kwargs: dict[str, Any] | None = dict_field(default=None)
    sae_lens_version: str = field(default_factory=lambda: __version__)
    sae_lens_training_version: str = field(default_factory=lambda: __version__)
    exclude_special_tokens: bool | list[int] = False

    def __post_init__(self):
        if self.use_cached_activations and self.cached_activations_path is None:
            self.cached_activations_path = _default_cached_activations_path(
                self.dataset_path,
                self.model_name,
                self.hook_name,
                self.hook_head_index,
            )
        self.tokens_per_buffer = (
            self.train_batch_size_tokens * self.context_size * self.n_batches_in_buffer
        )

        if self.logger.run_name is None:
            self.logger.run_name = f"{self.sae.architecture()}-{self.sae.d_sae}-LR-{self.lr}-Tokens-{self.training_tokens:3.3e}"

        if self.model_from_pretrained_kwargs is None:
            if self.model_class_name == "HookedTransformer":
                self.model_from_pretrained_kwargs = {"center_writing_weights": False}
            else:
                self.model_from_pretrained_kwargs = {}

        if self.act_store_device == "with_model":
            self.act_store_device = self.device

        if self.lr_end is None:
            self.lr_end = self.lr / 10

        unique_id = self.logger.wandb_id
        if unique_id is None:
            unique_id = cast(
                Any, wandb
            ).util.generate_id()  # not sure why this type is erroring
        self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}"

        if self.verbose:
            logger.info(
                f"Run name: {self.sae.architecture()}-{self.sae.d_sae}-LR-{self.lr}-Tokens-{self.training_tokens:3.3e}"
            )
            # Print out some useful info:
            n_tokens_per_buffer = (
                self.store_batch_size_prompts
                * self.context_size
                * self.n_batches_in_buffer
            )
            logger.info(
                f"n_tokens_per_buffer (millions): {n_tokens_per_buffer / 10**6}"
            )
            n_contexts_per_buffer = (
                self.store_batch_size_prompts * self.n_batches_in_buffer
            )
            logger.info(
                f"Lower bound: n_contexts_per_buffer (millions): {n_contexts_per_buffer / 10**6}"
            )

            total_training_steps = (
                self.training_tokens
            ) // self.train_batch_size_tokens
            logger.info(f"Total training steps: {total_training_steps}")

            total_wandb_updates = (
                total_training_steps // self.logger.wandb_log_frequency
            )
            logger.info(f"Total wandb updates: {total_wandb_updates}")

            # how many times will we sample dead neurons?
            # assert self.dead_feature_window <= self.feature_sampling_window, "dead_feature_window must be smaller than feature_sampling_window"
            n_feature_window_samples = (
                total_training_steps // self.feature_sampling_window
            )
            logger.info(
                f"n_tokens_per_feature_sampling_window (millions): {(self.feature_sampling_window * self.context_size * self.train_batch_size_tokens) / 10**6}"
            )
            logger.info(
                f"n_tokens_per_dead_feature_window (millions): {(self.dead_feature_window * self.context_size * self.train_batch_size_tokens) / 10**6}"
            )
            logger.info(
                f"We will reset the sparsity calculation {n_feature_window_samples} times."
            )
            # logger.info("Number tokens in dead feature calculation window: ", self.dead_feature_window * self.train_batch_size_tokens)
            logger.info(
                f"Number tokens in sparsity calculation window: {self.feature_sampling_window * self.train_batch_size_tokens:.2e}"
            )

        if self.context_size < 0:
            raise ValueError(
                f"The provided context_size is {self.context_size} is negative. Expecting positive context_size."
            )

        _validate_seqpos(seqpos=self.seqpos_slice, context_size=self.context_size)

        if isinstance(self.exclude_special_tokens, list) and not all(
            isinstance(x, int) for x in self.exclude_special_tokens
        ):
            raise ValueError("exclude_special_tokens list must contain only integers")

    @property
    def total_training_tokens(self) -> int:
        return self.training_tokens

    @property
    def total_training_steps(self) -> int:
        return self.total_training_tokens // self.train_batch_size_tokens

    # def get_base_sae_cfg_dict(self) -> dict[str, Any]:
    #     return self.sae.to_dict()

    def get_training_sae_cfg_dict(self) -> dict[str, Any]:
        return self.sae.to_dict()

    def to_dict(self) -> dict[str, Any]:
        # Make a shallow copy of config's dictionary
        d = dict(self.__dict__)

        d["logger"] = asdict(self.logger)

        # Overwrite fields that might not be JSON-serializable
        d["dtype"] = str(self.dtype)
        d["device"] = str(self.device)
        d["act_store_device"] = str(self.act_store_device)
        return d

    def to_json(self, path: str) -> None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path + "cfg.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "LanguageModelSAERunnerConfig[Any]":
        with open(path + "cfg.json") as f:
            cfg = json.load(f)

        # ensure that seqpos slices is a tuple
        # Ensure seqpos_slice is a tuple
        if "seqpos_slice" in cfg:
            if isinstance(cfg["seqpos_slice"], list):
                cfg["seqpos_slice"] = tuple(cfg["seqpos_slice"])
            elif not isinstance(cfg["seqpos_slice"], tuple):
                cfg["seqpos_slice"] = (cfg["seqpos_slice"],)

        return cls(**cfg)


@dataclass
class CacheActivationsRunnerConfig:
    """
    Configuration for creating and caching activations of an LLM.

    Args:
        dataset_path (str): The path to the Hugging Face dataset. This may be tokenized or not.
        model_name (str): The name of the model to use.
        model_batch_size (int): How many prompts are in the batch of the language model when generating activations.
        hook_name (str): The name of the hook to use.
        hook_layer (int): The layer of the final hook. Currently only support a single hook, so this should be the same as hook_name.
        d_in (int): Dimension of the model.
        total_training_tokens (int): Total number of tokens to process.
        context_size (int): Context size to process. Can be left as -1 if the dataset is tokenized.
        model_class_name (str): The name of the class of the model to use. This should be either `HookedTransformer` or `HookedMamba`.
        new_cached_activations_path (str, optional): The path to save the activations.
        shuffle (bool): Whether to shuffle the dataset.
        seed (int): The seed to use for shuffling.
        dtype (str): Datatype of activations to be stored.
        device (str): The device for the model.
        buffer_size_gb (float): The buffer size in GB. This should be < 2GB.
        hf_repo_id (str, optional): The Hugging Face repository id to save the activations to.
        hf_num_shards (int, optional): The number of shards to save the activations to.
        hf_revision (str): The revision to save the activations to.
        hf_is_private_repo (bool): Whether the Hugging Face repository is private.
        model_kwargs (dict): Keyword arguments for `model.run_with_cache`.
        model_from_pretrained_kwargs (dict): Keyword arguments for the `from_pretrained` method of the model.
        compile_llm (bool): Whether to compile the LLM.
        llm_compilation_mode (str): The torch.compile mode to use.
        prepend_bos (bool): Whether to prepend the beginning of sequence token. You should use whatever the model was trained with.
        seqpos_slice (tuple): Determines slicing of activations when constructing batches during training. The slice should be (start_pos, end_pos, optional[step_size]), e.g. for Othello we sometimes use (5, -5). Note, step_size > 0.
        streaming (bool): Whether to stream the dataset. Streaming large datasets is usually practical.
        autocast_lm (bool): Whether to use autocast during activation fetching.
        dataset_trust_remote_code (bool): Whether to trust remote code when loading datasets from Huggingface.
    """

    dataset_path: str
    model_name: str
    model_batch_size: int
    hook_name: str
    hook_layer: int
    d_in: int
    training_tokens: int

    context_size: int = -1  # Required if dataset is not tokenized
    model_class_name: str = "HookedTransformer"
    # defaults to "activations/{dataset}/{model}/{hook_name}
    new_cached_activations_path: str | None = None
    shuffle: bool = True
    seed: int = 42
    dtype: str = "float32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    buffer_size_gb: float = 2.0  # HF datasets writer have problems with shards > 2GB

    # Huggingface Integration
    hf_repo_id: str | None = None
    hf_num_shards: int | None = None
    hf_revision: str = "main"
    hf_is_private_repo: bool = False

    # Model
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    model_from_pretrained_kwargs: dict[str, Any] = field(default_factory=dict)
    compile_llm: bool = False
    llm_compilation_mode: str | None = None  # which torch.compile mode to use

    # Activation Store
    prepend_bos: bool = True
    seqpos_slice: tuple[int | None, ...] = (None,)
    streaming: bool = True
    autocast_lm: bool = False
    dataset_trust_remote_code: bool | None = None

    def __post_init__(self):
        # Automatically determine context_size if dataset is tokenized
        if self.context_size == -1:
            ds = load_dataset(self.dataset_path, split="train", streaming=True)
            assert isinstance(ds, IterableDataset)
            first_sample = next(iter(ds))
            toks = first_sample.get("tokens") or first_sample.get("input_ids") or None
            if toks is None:
                raise ValueError(
                    "Dataset is not tokenized. Please specify context_size."
                )
            token_length = len(toks)
            self.context_size = token_length

        if self.context_size == -1:
            raise ValueError("context_size is still -1 after dataset inspection.")

        if self.seqpos_slice is not None:
            _validate_seqpos(
                seqpos=self.seqpos_slice,
                context_size=self.context_size,
            )

        if self.new_cached_activations_path is None:
            self.new_cached_activations_path = _default_cached_activations_path(  # type: ignore
                self.dataset_path, self.model_name, self.hook_name, None
            )

    @property
    def sliced_context_size(self) -> int:
        if self.seqpos_slice is not None:
            return len(range(self.context_size)[slice(*self.seqpos_slice)])
        return self.context_size

    @property
    def bytes_per_token(self) -> int:
        return self.d_in * DTYPE_MAP[self.dtype].itemsize

    @property
    def n_tokens_in_buffer(self) -> int:
        # Calculate raw tokens per buffer based on memory constraints
        _tokens_per_buffer = int(self.buffer_size_gb * 1e9) // self.bytes_per_token
        # Round down to nearest multiple of batch_token_size
        return _tokens_per_buffer - (_tokens_per_buffer % self.n_tokens_in_batch)

    @property
    def n_tokens_in_batch(self) -> int:
        return self.model_batch_size * self.sliced_context_size

    @property
    def n_batches_in_buffer(self) -> int:
        return self.n_tokens_in_buffer // self.n_tokens_in_batch

    @property
    def n_seq_in_dataset(self) -> int:
        return self.training_tokens // self.sliced_context_size

    @property
    def n_seq_in_buffer(self) -> int:
        return self.n_tokens_in_buffer // self.sliced_context_size

    @property
    def n_buffers(self) -> int:
        return math.ceil(self.training_tokens / self.n_tokens_in_buffer)


def _default_cached_activations_path(
    dataset_path: str,
    model_name: str,
    hook_name: str,
    hook_head_index: int | None,
) -> str:
    path = f"activations/{dataset_path.replace('/', '_')}/{model_name.replace('/', '_')}/{hook_name}"
    if hook_head_index is not None:
        path += f"_{hook_head_index}"
    return path


def _validate_seqpos(seqpos: tuple[int | None, ...], context_size: int) -> None:
    # Ensure that the step-size is larger or equal to 1
    if len(seqpos) == 3:
        step_size = seqpos[2] or 1
        if step_size <= 1:
            raise ValueError(
                f"Ensure the step_size={seqpos[2]} for sequence slicing is at least 1."
            )
    # Ensure that the choice of seqpos doesn't end up with an empty list
    if len(list(range(context_size))[slice(*seqpos)]) == 0:
        raise ValueError(
            f"The slice {seqpos} results in an empty range. Please adjust your seqpos or context_size."
        )


@dataclass
class PretokenizeRunnerConfig:
    tokenizer_name: str = "gpt2"
    dataset_path: str = ""
    dataset_name: str | None = None
    dataset_trust_remote_code: bool | None = None
    split: str | None = "train"
    data_files: list[str] | None = None
    data_dir: str | None = None
    num_proc: int = 4
    context_size: int = 128
    column_name: str = "text"
    shuffle: bool = True
    seed: int | None = None
    streaming: bool = False
    pretokenize_batch_size: int | None = 1000

    # special tokens
    begin_batch_token: int | Literal["bos", "eos", "sep"] | None = "bos"
    begin_sequence_token: int | Literal["bos", "eos", "sep"] | None = None
    sequence_separator_token: int | Literal["bos", "eos", "sep"] | None = "bos"

    # if saving locally, set save_path
    save_path: str | None = None

    # if saving to huggingface, set hf_repo_id
    hf_repo_id: str | None = None
    hf_num_shards: int = 64
    hf_revision: str = "main"
    hf_is_private_repo: bool = False
