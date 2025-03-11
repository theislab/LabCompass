from collections.abc import Callable, Sequence
from typing import Any

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
    _initialize_distributions: bool = True

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
        if self._initialize_distributions:
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
        if self.is_multi_attribute:
            mean = list(self.params.values())[0][0]["mean"]
        else:
            mean = self.params[0]["mean"]
        return mean.shape

    @property
    def is_multi_attribute(
        self,
    ) -> bool:
        """"""
        return isinstance(self.params, dict)

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
    _initialize_distributions: bool = True

    def __init__(
        self,
        params: Sequence[dict[str, TensorLike]],
        n_cat: int,
        cat_logit_lm: TensorLike,
        weights: Sequence[float] | None = None,
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
    def num_categories(
            self,
        ) -> int:
        """
        Returns the number of categorical labels.

        Returns:
            int: The number of categories.
        """
        return self.n_cat

    def sample_categories(
            self,
            features: TensorLike
        ) -> TensorLike:
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


class MultiAttributeAnnotatedGaussianMixtureModel(AnnotatedGaussianMixtureModel):
    """"""
    _initialize_distributions: bool = False

    def __init__(
        self,
        params: dict[str, Sequence[dict[str, TensorLike]]],
        n_cat: int,
        cat_logit_lm: TensorLike,
        weights: Sequence[float] | None = None,
    ) -> None:
        """"""
        super().__init__(
            params,
            n_cat,
            cat_logit_lm,
            weights=weights,
        )

    def sample_categories(
        self,
        features: TensorLike,
    ) -> dict[str, TensorLike]:
        """"""
        # computing the logits
        cov_logits = torch.matmul(features, self.cat_logit_lm)
        # sampling categories according to the logits
        return Categorical(logits=cov_logits).sample()

    def get_params(
        self,
        comps: tuple[int],
    ) -> Sequence[dict[str, TensorLike]]:
        """"""
        # retrieving the perturbation identifiers
        perturbation_ids = list(self.params.keys())
        # mapping the identifier to the integer index
        pert_ids = {idx: pert_id for idx, pert_id in enumerate(perturbation_ids)}
        # defining list to append the retrieved param
        params = []
        # iterating over the components of each perturbation
        for idx, comp in enumerate(comps):
            # this will give the identifier of the current component
            pert_id = pert_ids[idx]
            # this will retrieve the corresponding parameters
            pert_params = self.params[pert_id]
            # appending the current component params to the return list 
            params.append(pert_params[comp])
        return params

    def sample(
        self,
        num_samples: int | None = None,
        comps: np.ndarray | None = None,
    ) -> tuple[TensorLike, dict[str, TensorLike]]:
        """"""
        # overriding the num_samples argument if
        # the components are explicitly paxssed
        if comps is not None:
            # checking that all components have the same number of samples
            # by using the first one as reference
            reference_num_samples = len(list(comps.values())[0])
            for comp_cov, comp_ids in comps.items():
                num_samples = len(comp_ids)
                msg = f""
                assert num_samples == reference_num_samples, msg
        else:
            # only one sample by default
            if num_samples is None:
                num_samples = 1
            # Sample mixture components based on the weights 
            comps = { 
                comp_cov: np.random.choice(len(comp_params), size=num_samples, p=self.weights)
                    for comp_cov, comp_params in self.params.items()
            }

        # converting components array to list
        comps_copy = {}
        for comp_cov, comp_ids in comps.items():
            if isinstance(comps, np.ndarray | torch.Tensor):
                comp_ids = comp_ids.tolist()
            comps_copy[comp_cov] = comp_ids
        comps = comps_copy

        # zipping components together
        comps_zipped = list(zip(*list(comps.values())))

        # initializing store for distribution params
        states = []
        # iterating over the components of each observation
        for comp in comps_zipped:
            # retrieving the parameters for current components
            params = self.get_params(comp)
            # retrieving the mean for current components
            means = torch.stack([param["mean"] for param in params], dim=0)
            # computing the mean for current observations 
            # by summing the means of each perturbation feature
            mean = torch.sum(means, dim=0)

            # retrieving the mean for current components
            covs = torch.stack([param["cov"] for param in params], dim=0)
            # computing the mean for current observations 
            # by summing the means of each perturbation feature
            cov = torch.sum(covs, dim=0)

            # instantiating distribution for current observation
            states.append(
                MultivariateNormal(mean, cov).sample()
            )
        # concatenating the states
        states = torch.stack(states, dim=0)
        # sampling categories
        sampled_categories = self.sample_categories(states)
        return states, sampled_categories


class DoseResolvedAnnotatedGaussianMixtureModel(MultiAttributeAnnotatedGaussianMixtureModel):
    """"""
    _initialize_distributions: bool = False

    def __init__(
        self,
        params: dict[str, Sequence[dict[str, TensorLike]]] | Sequence[dict[str, TensorLike]],
        n_cat: int,
        cat_logit_lm: TensorLike,
        weights: Sequence[float] | None = None,
        dosage_prior: Callable[[Any], TensorLike] | None = None,
        interpolation_fn: Callable[[float, TensorLike, TensorLike], TensorLike] | None = None,
        control_mean: TensorLike | None = None
    ) -> None:
        """"""
        super().__init__(params, n_cat, cat_logit_lm, weights=weights)
        # setting additional attributes
        if dosage_prior is None:
            dosage_prior = torch.rand
        self.dosage_prior = dosage_prior

        if interpolation_fn is None:
            interpolation_fn = lambda dose, source, target: (1 - dose)*source + dose*target
        self.interpolation_fn = interpolation_fn
        
        if control_mean is None:
            control_mean = torch.zeros(self.dimensionality)
        self.control_mean = control_mean

        # handling parameters type in case is not multi-attribute 
        # to make it compatible with the methods of the parent class
        if not self.is_multi_attribute:
            self.params = {
                "pert": self.params,
            }

    def __interpolate_distributions(
        self,
        comps: dict[str, list[int]],
        dosages: dict[str, TensorLike],
    ) -> Sequence[MultivariateNormal]:
        """"""
        # converting dosages array to list
        dosages_copy = {}
        for pert_id, dosage in dosages.items():
            if isinstance(dosage, np.ndarray | torch.Tensor):
                dosage = dosage.tolist()
            dosages_copy[pert_id] = dosage
        dosages = dosages_copy

        # defining list of dose-resolved components
        dose_resolved_comps = []

        # zipping components together
        comps_zipped = list(zip(*list(comps.values())))

        # retrieving the perturbation identifiers
        perturbation_ids = list(self.params.keys())
        # mapping the identifier to the integer index
        pert_ids = {idx: pert_id for idx, pert_id in enumerate(perturbation_ids)}

        # iterating over the components of each observation
        for obs_id, comp in enumerate(comps_zipped):
            # retrieving the parameters for current components
            params = self.get_params(comp)

            # defining list to store interpolated means
            interpolated_means = []
            interpolated_covs = []

            # iterating over the perturbation covariates
            for covariate_idx, covariate_param in enumerate(params):
                # parsing params dictionary
                covariate_mean = covariate_param["mean"]
                covariate_cov = covariate_param["cov"]

                # retrieving the perturbation covariante
                pert_covariate = pert_ids[covariate_idx] 
                # retrieving corresponding dosage
                obs_dosage = dosages[pert_covariate][obs_id]

                # interpolating with control mean
                interpolated_mean = self.interpolation_fn(
                    obs_dosage,
                    self.control_mean,
                    covariate_mean,
                )

                # appending to interpolated mean
                interpolated_means.append(interpolated_mean)
                interpolated_covs.append(covariate_cov)

            # stacking the interpolated means and summing them
            interpolated_means = torch.stack(interpolated_means, dim=0)
            interpolated_mean = torch.sum(interpolated_means, dim=0)

            # stacking the interpolated covariances and summing them
            interpolated_covs = torch.stack(interpolated_covs, dim=0)
            interpolated_cov = torch.sum(interpolated_covs, dim=0)

            # updating components with new normal distribution
            dose_resolved_comps.append(
                MultivariateNormal(interpolated_mean, interpolated_cov)
            )
        return dose_resolved_comps

    def sample(
        self,
        dosages: np.ndarray | None = None,
        num_samples: int | None = None,
        comps: np.ndarray | None = None,
    ) -> tuple[TensorLike, TensorLike, TensorLike]:
        """"""

        # handling components type in case is not multi-attribute 
        # to make it compatible with the methods of the parent class
        if not isinstance(comps, dict):
            comps = {
                key: comps for key in self.params.keys()
            }

        # overriding the num_samples argument if
        # the components are explicitly paxssed
        if comps is not None:
            # checking that all components have the same number of samples
            # by using the first one as reference
            reference_num_samples = len(list(comps.values())[0])
            for comp_cov, comp_ids in comps.items():
                num_samples = len(comp_ids)
                msg = f""
                assert num_samples == reference_num_samples, msg
        else:
            # only one sample by default
            if num_samples is None:
                num_samples = 1
            # Sample mixture components based on the weights 
            comps = { 
                comp_cov: np.random.choice(len(comp_params), size=num_samples, p=self.weights)
                    for comp_cov, comp_params in self.params.items()
            }

        # converting components array to list
        comps_copy = {}
        for comp_cov, comp_ids in comps.items():
            if isinstance(comps, np.ndarray | torch.Tensor):
                comp_ids = comp_ids.tolist()
            comps_copy[comp_cov] = comp_ids
        comps = comps_copy
        
        # sampling from dosage prior if not explicitly passed
        if dosages is None:
            dosages = self.dosage_prior((num_samples, ))
        
        # handling dosages type in case is not multi-attribute 
        # to make it compatible with the methods of the parent class
        if not isinstance(dosages, dict):
            dosages = {
                key: dosages for key in self.params.keys()
            }
        
        # retrieving dose-resolved interpolated distributions
        distributions = self.__interpolate_distributions(comps, dosages)

        # sampling
        samples = torch.stack([distribution.sample() for distribution in distributions])
        sampled_categories = self.sample_categories(samples)
        return samples, sampled_categories, dosages
