import logging
import os

import cloudpickle

logger = logging.getLogger(__name__)

__all__ = ["BaseModel"]


class BaseModel:
    """"""

    def save(
        self,
        dump_dir: str,
        model_prefix: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """"""
        # construct file name
        if model_prefix is None:
            model_prefix = ""
        else:
            model_prefix = f"{model_prefix}_"
        file_name = f"{model_prefix}{self.__class__.__name__}.pkl"

        # defining path
        dump_path = os.path.join(dump_dir, file_name)

        # checking that the file exists
        if os.path.exists(dump_path):
            if not overwrite:
                msg = f"Cannot overwrite saved model at {dump_path} when {override=}."
                raise RuntimeError(msg)
            msg = f"Ovverriding model at {dump_path}."
            logger.warning(msg)

        # saving the model
        with open(dump_path, "wb") as fp:
            cloudpickle.dump(self, fp)

    @classmethod
    def load(
        cls,
        file_name: str,
    ) -> "BaseModel":
        """"""
        # loading model file
        with open(file_name, "rb") as fp:
            model = cloudpickle.load(fp)

        # veriying types
        if type(model) is not cls:
            msg = f"The loaded model is not of the correct type. Expected {cls}, found {type(model)}."
            raise TypeError(msg)
        
        return model
