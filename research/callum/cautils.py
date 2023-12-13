"""
Main set of utils to import at the start of all scripts and notebooks
"""

import sys
import warnings

import torch
import torch as t

warnings.warn("Setting grad enabled false...")
t.set_grad_enabled(False)

import gzip
import itertools
import pickle
import sys
import time
from copy import copy, deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import torch.nn.functional as F
from datasets import load_dataset
from jaxtyping import Bool, Float, Int, jaxtyped
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader, Dataset

import wandb

if str(__file__).startswith("/code"): # Hofvarpnir seems annoying...
    from tqdm import tqdm
else:
    from tqdm.auto import tqdm

# pio.kaleido.scope.mathjax = None # Fixes PDF bug here https://github.com/plotly/plotly.py/issues/3469
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import ctime

import einops
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import HTML, clear_output, display
from plotly.subplots import make_subplots
from rich import print as rprint
from rich.table import Column, Table
from transformer_lens import ActivationCache, FactoredMatrix, HookedTransformer, utils

try:
    import circuitsvis as cv
except ModuleNotFoundError:
    warnings.warn("No circuitsvis found")
import gc

import transformer_lens
from transformer_lens import *
