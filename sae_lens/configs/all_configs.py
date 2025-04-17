# @dataclass
# class ModelConfig:
#     """Configuration for the model and hooks"""

#     model_name: str = "gelu-2l"
#     model_class_name: str = "HookedTransformer"
#     hook_name: str = "blocks.0.hook_mlp_out"
#     hook_eval: str = "NOT_IN_USE"
#     hook_layer: int = 0
#     hook_head_index: Optional[int] = None
#     model_kwargs: dict[str, Any] = field(default_factory=dict)
#     model_from_pretrained_kwargs: dict[str, Any] | None = None
#     prepend_bos: bool = True


# @dataclass
# class DataConfig:
#     """Configuration for dataset and activation storage"""

#     dataset_path: str = ""
#     dataset_trust_remote_code: bool = True
#     streaming: bool = True
#     is_dataset_tokenized: bool = True
#     context_size: int = 128
#     use_cached_activations: bool = False
#     cached_activations_path: Optional[str] = None
#     n_batches_in_buffer: int = 20
#     store_batch_size_prompts: int = 32
#     seqpos_slice: tuple[int | None, ...] = (None,)


# @dataclass
# class SAEArchitectureConfig:
#     """Configuration for SAE architecture and initialization"""

#     architecture: Literal["standard", "gated", "jumprelu", "topk"] = "standard"
#     d_in: int = 512
#     d_sae: Optional[int] = None
#     expansion_factor: Optional[int] = None
#     activation_fn: str = None  # type: ignore
#     activation_fn_kwargs: dict[str, Any] = None  # type: ignore
#     normalize_sae_decoder: bool = True
#     noise_scale: float = 0.0
#     from_pretrained_path: Optional[str] = None
#     apply_b_dec_to_input: bool = True
#     decoder_orthogonal_init: bool = False
#     decoder_heuristic_init: bool = False
#     init_encoder_as_decoder_transpose: bool = False
#     b_dec_init_method: str = "geometric_median"
#     normalize_activations: str = "none"


# @dataclass
# class TrainingConfig:
#     """Configuration for training parameters"""

#     training_tokens: int = 2_000_000
#     finetuning_tokens: int = 0
#     train_batch_size_tokens: int = 4096
#     adam_beta1: float = 0.0
#     adam_beta2: float = 0.999
#     lr: float = 3e-4
#     lr_scheduler_name: str = "constant"
#     lr_warm_up_steps: int = 0
#     lr_end: Optional[float] = None
#     lr_decay_steps: int = 0
#     n_restart_cycles: int = 1
#     l1_coefficient: float = 1e-3
#     lp_norm: float = 1
#     l1_warm_up_steps: int = 0
#     mse_loss_normalization: Optional[str] = None
#     scale_sparsity_penalty_by_decoder_norm: bool = False


# @dataclass
# class FeatureResamplingConfig:
#     """Configuration for feature resampling and dead feature handling"""

#     use_ghost_grads: bool = False
#     feature_sampling_window: int = 2000
#     dead_feature_window: int = 1000
#     dead_feature_threshold: float = 1e-8


# @dataclass
# class SystemConfig:
#     """Configuration for system and performance settings"""

#     device: str = "cpu"
#     act_store_device: str = "with_model"
#     seed: int = 42
#     dtype: str = "float32"
#     autocast: bool = False
#     autocast_lm: bool = False
#     compile_llm: bool = False
#     llm_compilation_mode: str | None = None
#     compile_sae: bool = False
#     sae_compilation_mode: str | None = None


# @dataclass
# class LoggingConfig:
#     """Configuration for logging and checkpointing"""

#     log_to_wandb: bool = True
#     log_activations_store_to_wandb: bool = False
#     log_optimizer_state_to_wandb: bool = False
#     wandb_project: str = "mats_sae_training_language_model"
#     wandb_id: Optional[str] = None
#     run_name: Optional[str] = None
#     wandb_entity: Optional[str] = None
#     wandb_log_frequency: int = 10
#     eval_every_n_wandb_logs: int = 100
#     n_eval_batches: int = 10
#     eval_batch_size_prompts: int | None = None
#     verbose: bool = True
#     resume: bool = False
#     n_checkpoints: int = 0
#     checkpoint_path: str = "checkpoints"


# @dataclass
# class LanguageModelSAERunnerConfig:
#     """Main configuration class that combines all sub-configurations"""

#     model: ModelConfig = field(default_factory=ModelConfig)
#     data: DataConfig = field(default_factory=DataConfig)
#     sae: SAEArchitectureConfig = field(default_factory=SAEArchitectureConfig)
#     training: TrainingConfig = field(default_factory=TrainingConfig)
#     resampling: FeatureResamplingConfig = field(default_factory=FeatureResamplingConfig)
#     system: SystemConfig = field(default_factory=SystemConfig)
#     logging: LoggingConfig = field(default_factory=LoggingConfig)

#     # Version tracking
#     sae_lens_version: str = field(default_factory=lambda: __version__)
#     sae_lens_training_version: str = field(default_factory=lambda: __version__)
