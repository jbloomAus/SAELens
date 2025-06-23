from dataclasses import dataclass
from statistics import mean

import torch
from tqdm import tqdm

from sae_lens.training.types import DataProvider


@dataclass
class ActivationScaler:
    scaling_factor: float | None = None

    def scale(self, acts: torch.Tensor) -> torch.Tensor:
        if self.scaling_factor is None:
            return acts
        return acts * self.scaling_factor

    def unscale(self, acts: torch.Tensor) -> torch.Tensor:
        if self.scaling_factor is None:
            return acts
        return acts / self.scaling_factor

    def __call__(self, acts: torch.Tensor) -> torch.Tensor:
        return self.scale(acts)

    @torch.no_grad()
    def _calculate_mean_norm(
        self, data_provider: DataProvider, n_batches_for_norm_estimate: int = int(1e3)
    ) -> float:
        norms_per_batch: list[float] = []
        for _ in tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            acts = next(data_provider)
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        return mean(norms_per_batch)

    def estimate_scaling_factor(
        self,
        d_in: int,
        data_provider: DataProvider,
        n_batches_for_norm_estimate: int = int(1e3),
    ):
        mean_norm = self._calculate_mean_norm(
            data_provider, n_batches_for_norm_estimate
        )
        self.scaling_factor = (d_in**0.5) / mean_norm
