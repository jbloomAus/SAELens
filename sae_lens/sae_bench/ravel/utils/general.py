import collections
import datetime
import json
import os
import pickle as pkl
import random
import re
from typing import Any, Tuple
from zoneinfo import ZoneInfo

import datasets
import h5py
import numpy as np
import torch
from datasets import Dataset
from nnsight import NNsight
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.generate_ravel_instance import RAVELMetadata
from utils.generation_utils import generate_batched

REPO_DIR = f"ravel"
SRC_DIR = os.path.join(REPO_DIR, "src")
MODEL_DIR = os.path.join(REPO_DIR, "models")
DATA_DIR = os.path.join(REPO_DIR, "data")


def setup_environment():
    """
    Set up the environment by creating necessary directories and setting the random seed.
    """
    for d in [MODEL_DIR, DATA_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)

    import sys

    sys.path.append(REPO_DIR)
    sys.path.append(SRC_DIR)

    set_seed(0)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(
    model_id: str, model_name: str
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the model and tokenizer.

    Args:
        model_id (str): The ID of the model to load.
        model_name (str): The name of the model.

    Returns:
        tuple: The loaded model and tokenizer.
    """
    with open("/share/u/can/src/hf.txt", "r") as f:
        hf_token = f.read().strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.set_grad_enabled(False)  # avoid blowing up mem
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=MODEL_DIR,
        token=hf_token,
        device_map=device,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=MODEL_DIR,
        token=hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return hf_model, tokenizer


def wrap_model_nnsight(hf_model: AutoModelForCausalLM) -> Any:
    """
    Wrap the model with NNsight.

    Args:
        hf_model (AutoModelForCausalLM): The model to wrap.

    Returns:
        NNsight: The wrapped model.
    """
    return NNsight(hf_model)
