import torch

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}


SPARSITY_FILENAME = "sparsity.safetensors"
SAE_WEIGHTS_FILENAME = "sae_weights.safetensors"
SAE_CFG_FILENAME = "cfg.json"
RUNNER_CFG_FILENAME = "runner_cfg.json"
SPARSIFY_WEIGHTS_FILENAME = "sae.safetensors"
ACTIVATIONS_STORE_STATE_FILENAME = "activations_store_state.safetensors"
ACTIVATION_SCALER_CFG_FILENAME = "activation_scaler.json"
