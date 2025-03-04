from collections.abc import Sequence

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.distributions.categorical import Categorical

from sc_exp_design.types import TensorLike

__all__ = ["GaussianMixtureModel"]


class GaussianMixtureModel:
    """
    A class for representing and sampling from a Gaussian Mixture Model (GMM).

    The model consists of multiple Gaussian distributions, each with its own mean and covariance matrix.
    The distributions are weighted, and the class allows sampling from the mixture, as well as calculating
    the log probability of given samples under the mixture model.    
    """

    def __init__(
        self,
        params: Sequence[dict[str, TensorLike]],
        weights: Sequence[float] | None = None,
    ) -> None:
        """
        Initializes the Gaussian Mixture Model with the given parameters and weights.

        Args:
            params (Sequence[dict[str, TensorLike]]): A sequence of dictionaries containing "mean" and "cov"
                                                      for each Gaussian component.
            weights (Sequence[float] | None, optional): A sequence of weights for the components. If None,
                                                       uniform weights are used.
        """
        # uniform weights
        if weights is None:
            weights = [1 / len(params) for _ in params]
        assert len(params) == len(weights)
        self.params = params
        self.weights = weights
        self.__init_distributions()

    @property
    def num_components(self) -> int:
        """
        Returns the number of Gaussian components in the mixture.

        Returns:
            int: The number of components.
        """
        return len(self.params)

    @property
    def dimensionality(
        self,
    ) -> int:
        """
        Returns the dimensionality of the samples (i.e., the shape of the mean vectors).

        Returns:
            int: The dimensionality of the samples.
        """
        mean = self.params[0]["mean"]
        return mean.shape

    def __init_distributions(
        self,
    ) -> None:
        """
        Initializes the list of `MultivariateNormal` distributions for each component
        in the Gaussian mixture model.
        """
        self.distributions = [MultivariateNormal(params["mean"], params["cov"]) for params in self.params]

    def sample(
        self,
        num_samples: int | None = None,
        comps: np.ndarray | None = None,
    ) -> TensorLike:
        """
        Samples from the Gaussian mixture model. If no components are specified, samples are drawn based on the
        mixture weights.

        Args:
            num_samples (int, optional): The number of samples to generate. If None, a single sample is returned.
            comps (np.ndarray | None, optional): If provided, it specifies which components to sample from for each sample.

        Returns:
            TensorLike: A tensor containing the sampled data points from the mixture model.
        """
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
        """
        Computes the log probability of the given samples under the Gaussian mixture model.

        Args:
            samples (TensorLike): A tensor containing the samples for which the log probability is calculated.

        Returns:
            TensorLike: The log probabilities of the samples.
        """
        ind_log_probs = [comp.log_prob(samples) * self.weights[idx] for idx, comp in enumerate(self.distributions)]
        ind_log_probs = torch.stack(ind_log_probs, axis=0)
        return torch.sum(ind_log_probs, axis=0)

class AnnotatedGaussianMixtureModel(GaussianMixtureModel):
    """
    A Gaussian Mixture Model (GMM) with categorical annotations.

    This class extends the standard Gaussian Mixture Model by incorporating categorical labels
    for each sample based on a learned linear transformation of the feature space.
    """

    def __init__(
        self,
        params,
        n_cat,  
        cat_logit_lm,
        weights=None,
    ) -> None:
        """
        Initializes the annotated Gaussian Mixture Model with given parameters, category information, 
        and a linear mapping for category logits.

        Args:
            params (Sequence[dict[str, TensorLike]]): A sequence of dictionaries containing "mean" and "cov" 
                                                      for each Gaussian component.
            n_cat (int): The number of categorical labels.
            cat_logit_lm (TensorLike): A matrix (n_features x n_categories) representing the logits for categories.
            weights (Sequence[float] | None, optional): A sequence of weights for the Gaussian components. 
                                                        If None, uniform weights are used.
        """
        super(AnnotatedGaussianMixtureModel, self).__init__(params, weights)
        self.n_cat = n_cat  # number of categories 
        self.cat_logit_lm = cat_logit_lm  # (n_features x n_categories) matrix representing the logits 
        
    @property
    def num_categories(self) -> int:
        """
        Returns the number of categorical labels.

        Returns:
            int: The number of categories.
        """
        return self.n_cat

    def sample_categories(self, features):
        """
        Samples categorical labels based on the given features.

        Args:
            features (TensorLike): The feature vectors for which to sample categories.

        Returns:
            TensorLike: The sampled categorical labels.
        """
        # Collect class logits for the samples
        logits = torch.matmul(features, self.cat_logit_lm)
        sampled_categories = Categorical(logits=logits).sample()
        return sampled_categories

    def sample(
        self,
        num_samples: int | None = None,
        comps: np.ndarray | None = None,
    ) -> TensorLike:
        """
        Samples from the Gaussian mixture model and assigns categorical labels.

        Args:
            num_samples (int, optional): The number of samples to generate. If None, a single sample is returned.
            comps (np.ndarray | None, optional): If provided, specifies which components to sample from for each sample.

        Returns:
            tuple[TensorLike, TensorLike]: A tuple containing the sampled data points and their corresponding categorical labels.
        """
        # overriding the num_samples argument if
        # the components are explicitly paxssed
        if comps is not None:
            num_samples = comps.shape[0]
        else:
            # only one sample by default
            if num_samples is None:
                num_samples = 1
            # Sample mixture components based on the weights 
            comps = np.random.choice(self.num_components, size=num_samples, p=self.weights)

        # converting components array to list
        if isinstance(comps, np.ndarray | torch.Tensor):
            comps = comps.tolist()

        samples = torch.stack([self.distributions[comp].sample() for comp in comps])
        sampled_categories = self.sample_categories(samples)
        return samples, sampled_categories
