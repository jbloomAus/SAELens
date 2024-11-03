import torch

from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.sae_training_runner import SAETrainingRunner

# os.environ["WANDB_MODE"] = "offline"  # turn this off if you want to see the output


# The way to run this with this command:
# poetry run py.test tests/benchmark/test_language_model_sae_runner.py --profile-svg -s
def test_language_model_sae_runner():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # total_training_steps = 20_000
    total_training_steps = 500
    batch_size = 4096
    total_training_tokens = total_training_steps * batch_size
    print(f"Total Training Tokens: {total_training_tokens}")

    lr_warm_up_steps = 0
    lr_decay_steps = 40_000
    print(f"lr_decay_steps: {lr_decay_steps}")
    l1_warmup_steps = 10_000
    print(f"l1_warmup_steps: {l1_warmup_steps}")

    cfg = LanguageModelSAERunnerConfig(
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        d_in=512,
        dataset_path="NeelNanda/c4-tokenized-2b",
        context_size=256,
        is_dataset_tokenized=True,
        prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
        # How big do we want our SAE to be?
        expansion_factor=16,
        # Dataset / Activation Store
        # When we do a proper test
        # training_tokens= 820_000_000, # 200k steps * 4096 batch size ~ 820M tokens (doable overnight on an A100)
        # For now.
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=4096,
        # Loss Function
        ## Reconstruction Coefficient.
        mse_loss_normalization=None,  # MSE Loss Normalization is not mentioned (so we use stanrd MSE Loss). But not we take an average over the batch.
        ## Anthropic does not mention using an Lp norm other than L1.
        l1_coefficient=5,
        lp_norm=1.0,
        # Instead, they multiply the L1 loss contribution
        # from each feature of the activations by the decoder norm of the corresponding feature.
        scale_sparsity_penalty_by_decoder_norm=True,
        # Learning Rate
        lr_scheduler_name="constant",  # we set this independently of warmup and decay steps.
        l1_warm_up_steps=l1_warmup_steps,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_warm_up_steps,
        ## No ghost grad term.
        use_ghost_grads=False,
        # Initialization / Architecture
        apply_b_dec_to_input=False,
        # encoder bias zero's. (I'm not sure what it is by default now)
        # decoder bias zero's.
        b_dec_init_method="zeros",
        normalize_sae_decoder=False,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,
        # Optimizer
        lr=4e-5,
        ## adam optimizer has no weight decay by default so worry about this.
        adam_beta1=0.9,
        adam_beta2=0.999,
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=64,
        store_batch_size_prompts=16,
        normalize_activations="none",
        # Feature Store
        feature_sampling_window=1000,
        dead_feature_window=1000,
        dead_feature_threshold=1e-4,
        # performance enhancement:
        compile_sae=False,
        # WANDB
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        wandb_project="benchmark",
        wandb_log_frequency=100,
        # Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="float32",
    )

    # look at the next cell to see some instruction for what to do while this is running.
    sae = SAETrainingRunner(cfg).run()

    assert sae is not None
    # know whether or not this works by looking at the dashboard!


def test_language_model_sae_runner_gated():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # total_training_steps = 20_000
    total_training_steps = 500
    batch_size = 4096
    total_training_tokens = total_training_steps * batch_size
    print(f"Total Training Tokens: {total_training_tokens}")

    lr_warm_up_steps = 0
    lr_decay_steps = 40_000
    print(f"lr_decay_steps: {lr_decay_steps}")
    l1_warmup_steps = 10_000
    print(f"l1_warmup_steps: {l1_warmup_steps}")

    cfg = LanguageModelSAERunnerConfig(
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        architecture="gated",
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        d_in=512,
        dataset_path="NeelNanda/c4-tokenized-2b",
        context_size=256,
        is_dataset_tokenized=True,
        prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
        # How big do we want our SAE to be?
        expansion_factor=16,
        # Dataset / Activation Store
        # When we do a proper test
        # training_tokens= 820_000_000, # 200k steps * 4096 batch size ~ 820M tokens (doable overnight on an A100)
        # For now.
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=4096,
        # Loss Function
        ## Reconstruction Coefficient.
        mse_loss_normalization=None,  # MSE Loss Normalization is not mentioned (so we use stanrd MSE Loss). But not we take an average over the batch.
        ## Anthropic does not mention using an Lp norm other than L1.
        l1_coefficient=5,
        lp_norm=1.0,
        # Instead, they multiply the L1 loss contribution
        # from each feature of the activations by the decoder norm of the corresponding feature.
        scale_sparsity_penalty_by_decoder_norm=True,
        # Learning Rate
        lr_scheduler_name="constant",  # we set this independently of warmup and decay steps.
        l1_warm_up_steps=l1_warmup_steps,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_warm_up_steps,
        ## No ghost grad term.
        use_ghost_grads=False,
        # Initialization / Architecture
        apply_b_dec_to_input=False,
        # encoder bias zero's. (I'm not sure what it is by default now)
        # decoder bias zero's.
        b_dec_init_method="zeros",
        normalize_sae_decoder=False,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,
        # Optimizer
        lr=4e-5,
        ## adam optimizer has no weight decay by default so worry about this.
        adam_beta1=0.9,
        adam_beta2=0.999,
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=64,
        store_batch_size_prompts=16,
        normalize_activations="none",
        # Feature Store
        feature_sampling_window=1000,
        dead_feature_window=1000,
        dead_feature_threshold=1e-4,
        # performance enhancement:
        compile_sae=False,
        # WANDB
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        wandb_project="benchmark",
        wandb_log_frequency=100,
        # Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="float32",
    )

    # look at the next cell to see some instruction for what to do while this is running.
    sae = SAETrainingRunner(cfg).run()

    assert sae is not None
    # know whether or not this works by looking at the dashboard!


def test_language_model_sae_runner_top_k():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # total_training_steps = 20_000
    total_training_steps = 500
    batch_size = 4096
    total_training_tokens = total_training_steps * batch_size
    print(f"Total Training Tokens: {total_training_tokens}")

    lr_warm_up_steps = 0
    lr_decay_steps = 40_000
    print(f"lr_decay_steps: {lr_decay_steps}")
    l1_warmup_steps = 10_000
    print(f"l1_warmup_steps: {l1_warmup_steps}")

    cfg = LanguageModelSAERunnerConfig(
        activation_fn="topk",
        activation_fn_kwargs={"k": 32},
        normalize_activations="layer_norm",
        # Pick a tiny model to make this easier.
        model_name="gelu-1l",
        ## MLP Layer 0 ##
        hook_name="blocks.0.hook_mlp_out",
        hook_layer=0,
        d_in=512,
        dataset_path="NeelNanda/c4-tokenized-2b",
        context_size=256,
        is_dataset_tokenized=True,
        prepend_bos=True,  # I used to train GPT2 SAEs with a prepended-bos but no longer think we should do this.
        # How big do we want our SAE to be?
        expansion_factor=16,
        training_tokens=total_training_tokens,  # For initial testing I think this is a good number.
        train_batch_size_tokens=4096,
        # Loss Function
        ## Reconstruction Coefficient.
        mse_loss_normalization=None,  # MSE Loss Normalization is not mentioned (so we use stanrd MSE Loss). But not we take an average over the batch.
        ## Anthropic does not mention using an Lp norm other than L1.
        l1_coefficient=5,
        lp_norm=1.0,
        # Instead, they multiply the L1 loss contribution
        # from each feature of the activations by the decoder norm of the corresponding feature.
        scale_sparsity_penalty_by_decoder_norm=False,
        # Learning Rate
        lr_scheduler_name="constant",  # we set this independently of warmup and decay steps.
        l1_warm_up_steps=l1_warmup_steps,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_warm_up_steps,
        ## No ghost grad term.
        apply_b_dec_to_input=True,
        b_dec_init_method="geometric_median",
        normalize_sae_decoder=True,
        decoder_heuristic_init=False,
        init_encoder_as_decoder_transpose=True,
        # Optimizer
        lr=4e-5,
        ## adam optimizer has no weight decay by default so worry about this.
        adam_beta1=0.9,
        adam_beta2=0.999,
        # Buffer details won't matter in we cache / shuffle our activations ahead of time.
        n_batches_in_buffer=64,
        store_batch_size_prompts=16,
        # Feature Store
        feature_sampling_window=1000,
        dead_feature_window=1000,
        dead_feature_threshold=1e-4,
        # performance enhancement:
        compile_sae=False,
        # WANDB
        log_to_wandb=True,  # always use wandb unless you are just testing code.
        wandb_project="benchmark",
        wandb_log_frequency=100,
        # Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="float32",
    )

    # look at the next cell to see some instruction for what to do while this is running.
    sae = SAETrainingRunner(cfg).run()

    assert sae is not None
    # know whether or not this works by looking at the dashboard!


def test_language_model_sae_runner_othellogpt():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # total_training_steps = 20_000
    total_training_steps = 500
    batch_size = 4096
    total_training_tokens = total_training_steps * batch_size

    lr_warm_up_steps = 0
    lr_decay_steps = 40_000
    l1_warmup_steps = 10_000

    cfg = LanguageModelSAERunnerConfig(
        # Data Generating Function (Model + Training Distibuion)
        model_name="othello-gpt",  # othello-gpt model
        hook_name="blocks.6.hook_resid_pre",  # A valid hook point (see more details here: https://neelnanda-io.github.io/TransformerLens/generated/demos/Main_Demo.html#Hook-Points)
        hook_layer=6,  # Only one layer in the model.
        d_in=512,  # the width of the mlp output.
        dataset_path="taufeeque/othellogpt",  # this is a tokenized language dataset on Huggingface for OthelloGPT games.
        is_dataset_tokenized=True,
        streaming=True,  # we could pre-download the token dataset if it was small.
        # SAE Parameters
        mse_loss_normalization=None,  # We won't normalize the mse loss,
        expansion_factor=16,  # the width of the SAE. Larger will result in better stats but slower training.
        b_dec_init_method="geometric_median",  # The geometric median can be used to initialize the decoder weights.
        apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
        normalize_sae_decoder=False,
        scale_sparsity_penalty_by_decoder_norm=True,
        decoder_heuristic_init=True,
        init_encoder_as_decoder_transpose=True,
        normalize_activations="expected_average_only_in",
        # Training Parameters
        lr=0.00003,  # lower the better, we'll go fairly high to speed up the tutorial.
        adam_beta1=0.9,  # adam params (default, but once upon a time we experimented with these.)
        adam_beta2=0.999,
        lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
        lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
        lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
        l1_coefficient=0.001,  # will control how sparse the feature activations are
        l1_warm_up_steps=l1_warmup_steps,  # this can help avoid too many dead features initially.
        lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
        train_batch_size_tokens=batch_size,
        context_size=59,  # will control the length of the prompts we feed to the model. Larger is better but slower. so for the tutorial we'll use a short one.
        seqpos_slice=(5, -5),
        # Activation Store Parameters
        n_batches_in_buffer=32,  # controls how many activations we store / shuffle.
        training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
        store_batch_size_prompts=32,
        # Resampling protocol
        use_ghost_grads=False,  # we don't use ghost grads anymore.
        feature_sampling_window=500,  # this controls our reporting of feature sparsity stats
        dead_feature_window=1000000,  # would effect resampling or ghost grads if we were using it.
        dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
        # WANDB
        log_to_wandb=False,  # always use wandb unless you are just testing code.
        wandb_project="benchmark",
        wandb_log_frequency=100,
        eval_every_n_wandb_logs=20,
        # Misc
        device=device,
        seed=42,
        n_checkpoints=0,
        checkpoint_path="checkpoints",
        dtype="torch.float32",
    )

    # look at the next cell to see some instruction for what to do while this is running.
    sae = SAETrainingRunner(cfg).run()

    assert sae is not None
    # know whether or not this works by looking at the dashboard!
