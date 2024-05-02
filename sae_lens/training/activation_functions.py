from typing import Callable

import torch


def get_activation_fn(activation_fn: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation_fn == "relu":
        return torch.nn.ReLU()
    elif activation_fn == "tanh-relu":
        return tanh_relu
    elif activation_fn == "tanh":
        return torch.tanh
    elif activation_fn == "identity":
        return identity
    else:
        raise ValueError(f"Unknown activation function: {activation_fn}")


def tanh_relu(input: torch.Tensor) -> torch.Tensor:
    input = torch.relu(input)
    input = torch.tanh(input)
    return input


def identity(input: torch.Tensor) -> torch.Tensor:
    return input
