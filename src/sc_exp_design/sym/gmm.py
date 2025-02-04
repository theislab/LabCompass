from collections.abc import Sequence

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from sc_exp_design.types import TensorLike

__all__ = ["GaussianMixtureModel"]


class GaussianMixtureModel:
    """"""

    def __init__(
        self,
        params: Sequence[dict[str, TensorLike]],
        weights: Sequence[float] | None = None,
    ) -> None:
        """"""
        # uniform weights
        if weights is None:
            weights = [1 / len(params) for _ in params]
        assert len(params) == len(weights)
        self.params = params
        self.weights = weights
        self.__init_distributions()

    @property
    def num_components(self) -> int:
        """"""
        return len(self.params)

    @property
    def dimensionality(
        self,
    ) -> int:
        """"""
        mean = self.params[0]["mean"]
        return mean.shape

    def __init_distributions(
        self,
    ) -> None:
        """"""
        self.distributions = [MultivariateNormal(params["mean"], params["cov"]) for params in self.params]

    def sample(
        self,
        num_samples: int | None = None,
        comps: np.ndarray | None = None,
    ) -> TensorLike:
        """"""
        # overriding the num_samples argument if
        # the components are explicitly passed
        if comps is not None:
            num_samples = comps.shape[0]
        else:
            # only one sample by default
            if num_samples is None:
                num_samples = 1
            # ode the components
            # using the weights
            comps = np.random.choice(self.num_components, size=num_samples, p=self.weights)
        # allocating memory
        samples = torch.zeros((num_samples, *self.dimensionality))
        # converting components array to list
        if isinstance(comps, np.ndarray | torch.Tensor):
            comps = comps.tolist()
        # iterating over the samples (TODO: vectorize this loop)
        for sample_id, sample_comp in enumerate(comps):
            distribution = self.distributions[sample_comp]
            sample = distribution.sample()
            samples[sample_id] = sample
        return samples

    def log_prob(
        self,
        samples: TensorLike,
    ) -> TensorLike:
        """"""
        ind_log_probs = [comp.log_prob(samples) * self.weights[idx] for idx, comp in enumerate(self.distributions)]
        ind_log_probs = torch.stack(ind_log_probs, axis=0)
        return torch.sum(ind_log_probs, axis=0)
