from collections.abc import Iterator

import torch

DataProvider = Iterator[torch.Tensor]
