import pickle
import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class SAETrainingRunState:
    """
    The SAE Training state exists to enable interupted training to be resumed.

    When interupted, or saving a checkpoint, the SAE training state can be
    saved to enable resuming training from the same point later.

    Note: Whereas we attempt to ensure that loading an SAE from a previous checkpoint
    will be possible as versions of the package change, we do not guarantee that
    for resuming training. This is because the training state is far more flexible.
    """

    torch_state: Optional[torch.Tensor] = None
    torch_cuda_state: Optional[list[torch.Tensor]] = None
    numpy_state: Optional[
        dict[str, Any]
        | tuple[str, np.ndarray[Any, np.dtype[np.uint32]], int, int, float]
    ] = None
    random_state: Optional[Any] = None

    def __post_init__(self):
        if self.torch_state is None:
            self.torch_state = torch.get_rng_state()
        if self.torch_cuda_state is None:
            self.torch_cuda_state = torch.cuda.get_rng_state_all()
        if self.numpy_state is None:
            self.numpy_state = np.random.get_state()
        if self.random_state is None:
            self.random_state = random.getstate()

    def set_random_state(self):
        assert self.torch_state is not None
        torch.random.set_rng_state(self.torch_state)
        assert self.torch_cuda_state is not None
        torch.cuda.set_rng_state_all(self.torch_cuda_state)
        assert self.numpy_state is not None
        np.random.set_state(self.numpy_state)
        assert self.random_state is not None
        random.setstate(self.random_state)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            attr_dict = pickle.load(f)
        return cls(**attr_dict)

    def save(self, path: str):
        attr_dict = {**self.__dict__}
        with open(path, "wb") as f:
            pickle.dump(attr_dict, f)
