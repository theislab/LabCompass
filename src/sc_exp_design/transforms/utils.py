import logging
from typing import Any

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

from sc_exp_design.constants import DataFields

logger = logging.getLogger(__name__)

__all__ = [
    "one_hot_encode",
    "label_encode",
]

def one_hot_encode(
    data: np.ndarray,
    encoder_kwargs: dict[str, Any]
) -> np.ndarray:
    """"""
    # when we have only one dimension we reshape it to two
    if data.ndim == 1:
        msg = f"When using \"one_hot\" as target representation in `target_covariates`, you need to pass a 2-dimensional array, found {data.ndim=}. Reshaping the array."
        data = data.reshape(-1, 1)

    # we need two dimensions
    if data.ndim != 2:
        msg = f"When using \"one_hot\" as target representation in `target_covariates`, you need to pass a 2-dimensional array, found {data.ndim=}."
        raise ValueError(msg)

    # encoding the condition
    encoder = OneHotEncoder(**encoder_kwargs)
    return encoder.fit_transform(data).toarray()


def label_encode(
    data: np.ndarray,
    encoder_kwargs: dict[str, Any]
) -> np.ndarray:
    """"""

    # when we have two dimension the second should be a singleton
    if data.ndim == 2:
        if data.shape[1] != 1:
            msg = ""
            raise ValueError(msg)
        data = data.reshape(-1)

    # we require only one dimension when using label encoding
    if data.ndim != 1:
        msg = f""
        raise ValueError(msg)
    
    # retrieving encoder
    encoder = LabelEncoder()
    # when we specify the target categories of interest
    if DataFields.TARGET_CATEGORIES in encoder_kwargs.keys():
        target_categories = encoder_kwargs[DataFields.TARGET_CATEGORIES]
        encoder.fit(target_categories)
        data = encoder.transform(data)
    # inferring target categories from data
    else:
        data = encoder.fit_transform(data)
    return data
