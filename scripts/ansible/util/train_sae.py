import inspect
import os
import sys

import torch
import yaml

from sae_lens.training.config import DTYPE_MAP, LanguageModelSAERunnerConfig
from sae_lens.training.lm_runner import language_model_sae_runner

# sys.path.append("..")


if len(sys.argv) > 1:
    job_name = sys.argv[1]
    print(f"Train SAE Job Name: {job_name}")
else:
    raise ValueError("Error: One argument required - the Train SAE Job Name")

if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
    device = "mps"
    torch.mps.empty_cache()
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# load the yaml file as config
# load only the keys that are in LanguageModelSAERunnerConfig
# TODO: this is a hacky way of importing
with open(f"./jobs/train_sae/{job_name}/{job_name}.yml", "r") as file:
    config_yaml = yaml.load(file, Loader=yaml.FullLoader)

    config_params = inspect.signature(LanguageModelSAERunnerConfig).parameters
    filtered_data = {k: v for k, v in config_yaml.items() if k in config_params}
    config = LanguageModelSAERunnerConfig(**filtered_data)

    if type(config.dtype) != torch.dtype:
        config.dtype = DTYPE_MAP[config.dtype]  # type: ignore
    config.device = device

if config is None:
    raise ValueError("Error: The config is not loaded.")

print(f"l1_warm_up_steps: {config.l1_warm_up_steps}")
print(f"lr_warm_up_steps: {config.lr_warm_up_steps}")
print(f"lr_decay_steps: {config.lr_decay_steps}")

cached_activations_path = config.cached_activations_path
if cached_activations_path is None:
    raise ValueError("Error: The cached_activations_path is not set.")

sparse_autoencoder_dictionary = language_model_sae_runner(config)
