import logging
from collections.abc import Sequence
from typing import Any, Literal

import anndata
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sc_exp_design.constants import DataFields
from sc_exp_design.data.data import AnnotatedPerturbationData
from sc_exp_design.types import TensorLike

logger = logging.getLogger(__name__)

__all__ = [
    "DataManager",
]


class DataManager:
    """
    Class for managing perturbation-related data in single-cell experiments.
    """
    def __init__(
        self,
        adata: anndata.AnnData | None = None,
        sample_rep: str | dict[str] | None = None,
        control_key: str | None = None,
        perturbations: str | Sequence[str] | None = None,
        perturbations_in_obsm: Sequence[str] | None = None, 
        perturbation_covariates: dict[str, str | Sequence[str]] | None = None,
        perturbation_reps: dict[str, str | Sequence[str]] | None = None,
        load_target_covariates: bool = False,
        target_covariates: dict[str, Literal["one_hot", "label", "identity"] | None] | None = None,
        target_covariates_in_obsm: dict[str, bool] | None = None,
        target_covariates_kwargs: dict[str, Any] | None = None,
        has_controls: bool = True

    ) -> None:
        """
        `self.perturbations`:
            the individual perturbations modeled. need to be a key in `adata.obs`

        `self.perturbation_reps`:
            the representation in self.perturbation_covariate_reps
            are the same over cell with the same perturbation
            basically the resulting data will have one
            entry for each unique perturbation considered
            for example, for a given drug this could be given
            by their chemical representation or any other feature that
            is constant across cells for any perturbation

        `self.perturbation_covariates`:
            these will be the covariates associated to a given perturbation
            on a cell level basis and that can vary over the cells for the
            same perturbations. The resulting data will have one entry for each cell.
            for example, for a given drug this could be given
            by the dosage or the time of the treatment or any other feature
            associated to the current perturbation that can vary across cells

        """
        self.adata = adata
        self.sample_rep = sample_rep
        self.control_key = control_key
        self.has_controls = has_controls
        
        # sanity check perturbations_in_obsm is iterable  
        if perturbations_in_obsm is not None:
            if isinstance(perturbations_in_obsm, str):
                perturbations_in_obsm = (perturbations_in_obsm,)

        # preparing the attributes
        if perturbations is not None:
            # handling type of perturbations argument
            if isinstance(perturbations, str):
                perturbations = (perturbations,)
            # iterating over the perturbations
            for perturbation in perturbations:
                # perturbation level covariates
                if perturbation_reps is not None:
                    if perturbation in perturbation_reps.keys():
                        rep = perturbation_reps[perturbation]
                        if (perturbations_in_obsm is not None) and (perturbation in perturbations_in_obsm):
                            msg = "When a perturbation is in .obsm, there should be only one representatio"
                            assert isinstance(rep, str), msg
                        # strings as an iterable
                        if isinstance(rep, str):
                            rep = (rep,)
                        perturbation_reps[perturbation] = rep
                    # skip perturbation if no representation found, warn
                    else:
                        msg = f"{perturbation} in `self.perturbation` has no representation associated to it, skipping."
                        logger.warning(msg)
                        continue
                # cell level covariates
                if perturbation_covariates is not None:
                    if perturbation in perturbation_covariates.keys():
                        covariates = perturbation_covariates[perturbation]
                        if isinstance(covariates, str):
                            covariates = (covariates,)
                    else:
                        covariates = ()
                    perturbation_covariates[perturbation] = covariates
                                        
        self.perturbations = perturbations
        self.perturbations_in_obsm = perturbations_in_obsm
        self.perturbation_covariates = perturbation_covariates
        self.perturbation_reps = perturbation_reps
        self.load_target_covariates = load_target_covariates

        # when we need some target representation for the conditions
        if self.load_target_covariates:
            msg = f"With {self.load_target_covariates=} you need to specify the target covariate reprs in `target_covariates`, `None` found"
            assert target_covariates is not None, msg
            if target_covariates_kwargs is None:
                target_covariates_kwargs = {}
                for target_covariate, target_covariate_rep in target_covariates.items():
                    target_covariates_kwargs[target_covariate] = {}

            if target_covariates_in_obsm is None:
                target_covariates_in_obsm = ()

        self.target_covariates = target_covariates
        self.target_covariates_in_obsm = target_covariates_in_obsm
        self.target_covariates_kwargs = target_covariates_kwargs

    @property
    def perturbations_with_rep(
        self,
    ) -> dict[str, Sequence[str]] | None:
        """
        Returns a dictionary with keys given by the modeled perturbation and values being the list
        of unique values that each perturbation can assume. This is needed to get a complete list
        of perturbations from which we can sample unique perturbation when using OT couplings.
        """
        # sanity check as we need to have initialized `self.adata` attribute
        msg = f""
        assert self.adata is not None, msg
        # no perturbation found
        if self.perturbations is None:
            return None
        # no representation found
        if self.perturbation_reps is None:
            return None
        # defining list of perturbations for which we have found the representation
        perturbations_with_rep = {}
        # iterating over each perturbation covariate
        for perturbation in self.perturbations:
            if perturbation not in self.perturbations_in_obsm:
                # This will contain a list with all the representation modalities for the current perturbation. 
                # It will be automatically constructed even when the perturbation does not have an associated representation
                # (check `self.__init__`), in which case it will be a 0-elements sequence.
                # Hence, we can check the length of this list to verify whether the perturbation has an associated representation or not.
                perturbation_covariate_rep = self.perturbation_reps[perturbation]
                # when we have at least one element, it means that we have found an associated representation
                # and we can append the perturbation label to the list of perturbations.
                if len(perturbation_covariate_rep) > 0:
                    # now we iterate over the different representation and verify that 
                    # they share the same keys (i.e.: the unique values of the current perturbation)
                    # using the first representation as reference
                    reference_keys = list(self.adata.uns[perturbation_covariate_rep[0]].keys())
                    for rep in perturbation_covariate_rep:
                        # retrieving the covariates and their representations
                        covariate_reps_keys = list(self.adata.uns[rep].keys())
                        # sanity check, we should have the same keys for each representation
                        # associated to the current perturbation
                        msg = "" # probably should do this check within the `self.__init__` method like the other ones
                        assert covariate_reps_keys == reference_keys, msg
                    # now we can append the dictionary that maps the current perturbation to its unique values.
                    perturbations_with_rep[perturbation] = reference_keys
            # If in obsm, add a string for perturbation rep
            else:
                # if in .obsm, key is perturbation and value is a str representing the associated representation 
                perturbations_with_rep[perturbation] = self.perturbation_reps[perturbation]  
        return perturbations_with_rep

    def __get_state_data(
        self,
        adata: anndata.AnnData,
    ) -> TensorLike:
        """
        :param adata: AnnData object containing single-cell data.
        :type adata: anndata.AnnData

        :return: The primary state representation of cells.
        :rtype: TensorLike

        :raises ValueError: If `sample_rep` is specified but not found in `adata.obsm`.
        """
        if self.sample_rep is None:
            state_data = adata.X
        else:
            if self.sample_rep not in adata.obsm.keys():
                msg = f"{self.sample_rep=} not found in `adata.obsm` (Keys found: {list(adata.obsm.keys())})"
                raise ValueError(msg)
            state_data = adata.obsm[self.sample_rep]
        return state_data

    def __get_perturbation_data(
        self,
        adata: anndata.AnnData,
    ) -> dict[str, TensorLike]:
        """
        :param adata: AnnData object containing single-cell data.
        :type adata: anndata.AnnData

        :return: Dictionary mapping perturbation features to tensor representations.
        :rtype: dict[str, TensorLike]

        :raises ValueError: If a specified perturbation or its representation is not found in `adata`.
        """
        # retrieving perturbation data
        perturbation_data = {}
        # iterating over each perturbation covariate
        for perturbation in self.perturbations:
            if (self.perturbations_in_obsm is not None) and (perturbation in self.perturbations_in_obsm):
                rep = self.perturbation_reps[perturbation][0]
                perturbation_data[f"{DataFields.CONDITION_REP}_{perturbation}_{rep}"] = adata.obsm[rep]
            else:
                # sanity check on the input AnnData
                if perturbation not in adata.obs.keys():
                    msg = f"{perturbation} not found in `adata.obs.keys()`"
                    raise ValueError(msg)
                
                # what perturbation was applied
                covariate_data = adata.obs[perturbation].values
                # optionally retrieving the representation of such covariate
                if self.perturbation_reps is not None:
                    perturbation_covariate_rep = self.perturbation_reps[perturbation]
                    for rep in perturbation_covariate_rep:
                        # sanity check on the input AnnData
                        if rep not in adata.uns.keys():
                            msg = f"{perturbation_covariate_rep} not found in `adata.uns.keys()`"
                            raise ValueError(msg)
                        # retrieving the covariates and their representations
                        covariate_reps_dict = adata.uns[rep]
                        # mapping each observation condition to their representation
                        covariate_reps = [covariate_reps_dict[covariate] for covariate in covariate_data]
                        covariate_reps = np.stack(covariate_reps, axis=0)
                        # storing the results
                        covariate_rep_key = f"{DataFields.CONDITION_REP}_{perturbation}_{rep}"
                        perturbation_data[covariate_rep_key] = covariate_reps
                        
                # loading perturbation covariates that are individual for each cell
                if self.perturbation_covariates is not None:
                    perturbation_covariates = self.perturbation_covariates[perturbation]
                    # iterating over each of such covariates associated to the current perturbation
                    # which should be stored in the corresponding obsm field of the AnnData object
                    for covariate in perturbation_covariates:
                        # sanity check on the input AnnData
                        if covariate not in adata.obsm.keys():
                            msg = f"{covariate=} not found in `self.adata.obsm.keys()`"
                            raise ValueError(msg)
                        # retrieving the covariate data
                        covariate_data = adata.obsm[covariate]
                        # storing the results
                        covariate_cov_key = f"{DataFields.CONDITION_COV}_{perturbation}_{covariate}"
                        perturbation_data[covariate_cov_key] = covariate_data
        return perturbation_data

    def __get_target_data(
        self,
        adata: anndata.AnnData,
    ) -> dict[str, TensorLike]:
        """
        :param adata: AnnData object containing single-cell data.
        :type adata: anndata.AnnData

        :return: Dictionary mapping perturbation target covariates to encoded representations.
        :rtype: dict[str, TensorLike]

        :raises AssertionError: If a required perturbation target covariate is missing in `adata.obs` or `adata.obsm`.
        :raises NotImplementedError: If an unsupported encoding type is requested.
        """
        # dictionary storing representations for perturbation target covariates
        out_dict = {}
        
        for target_covariate, target_covariate_rep in self.target_covariates.items():
            # if covariate is stored in adata.obsm we retrieve its representation directly
            if target_covariate in self.target_covariates_in_obsm:
                # sanity check
                msg = f"{target_covariate} not found in `adata.obsm.keys()`"
                assert target_covariate in adata.obsm.keys(), msg
                target_covariate_data = adata.obsm[target_covariate]
                out_dict[target_covariate] = target_covariate_data
            else:
                # sanity check
                msg = f"{target_covariate} not found in `adata.obs.columns`"
                assert target_covariate in adata.obs.columns, msg

                # retrieving the keywargs argument to get the target representation
                covariate_target_rep_kwargs = self.target_covariates_kwargs[target_covariate]

                # Collect the condition target covariate from the adata.obs 
                covariate_data = adata.obs[[target_covariate]].values

                if target_covariate_rep == "one_hot":
                    covariate_rep_encoder = OneHotEncoder(**covariate_target_rep_kwargs)
                    covariate_target_rep_data = covariate_rep_encoder.fit_transform(covariate_data).toarray()
                elif target_covariate_rep == "label":
                    covariate_rep_encoder = LabelEncoder()
                    if DataFields.TARGET_CATEGORIES in covariate_target_rep_kwargs.keys():
                        target_categories = covariate_target_rep_kwargs[DataFields.TARGET_CATEGORIES]
                        covariate_rep_encoder.fit(target_categories)
                        covariate_target_rep_data = covariate_rep_encoder.transform(covariate_data)
                    else:
                        covariate_target_rep_data = covariate_rep_encoder.fit_transform(covariate_data)
                elif target_covariate_rep == "identity":
                    covariate_target_rep_data = covariate_data
                else:
                    msg = f"{target_covariate_rep=} not currently supported (avaiable options are `['one_hot', 'label', 'identity']`)"
                    raise NotImplementedError(msg)

                # Retrun dictionary 
                out_dict[target_covariate] = covariate_target_rep_data
        return out_dict

    def get_data(
        self,
        adata: anndata.AnnData | None = None,
    ) -> AnnotatedPerturbationData:
        """
        :param adata: AnnData object containing single-cell data. If `None`, uses `self.adata`.
        :type adata: anndata.AnnData | None

        :return: A structured object containing all necessary training inputs.
        :rtype: AnnotatedPerturbationData

        :raises ValueError: If both `adata` and `self.adata` are `None`.
        """
        if adata is None and self.adata is None:
            msg = "Both `adata` and `self.adata` are None, you need to pass an `anndata.AnnData` object containing the data."
            raise ValueError(msg)
        elif adata is None:
            adata = self.adata
        # return cell features 
        state_data = self.__get_state_data(adata)
        perturbation_data = None
        # return perturbation data
        if self.perturbations is not None:
            perturbation_data = self.__get_perturbation_data(adata)
        # condition target representation
        target_data = None
        if self.load_target_covariates:
            target_data = self.__get_target_data(adata)
        
        return AnnotatedPerturbationData(adata,
                                         self.control_key, 
                                         state_data,
                                         perturbation_data,
                                         target_data,
                                         self.perturbations_with_rep, 
                                         self.has_controls,
                                         self.perturbations_in_obsm)
        