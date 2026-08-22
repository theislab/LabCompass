import logging
import os

import cloudpickle

logger = logging.getLogger(__name__)

__all__ = ["BaseModel"]


class BaseModel:
    """Base class providing model persistence utilities based on :mod:`cloudpickle`."""

    def save(
        self,
        dump_dir: str,
        model_prefix: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """Serializes the model instance to disk using :mod:`cloudpickle`.

        :param dump_dir: Directory where the pickled model file is saved.
        :type dump_dir: class:`str`

        :param model_prefix: Optional prefix prepended to the saved file name, defaults to `None` in which case no prefix is added.
        :type model_prefix: class:`str | None`

        :param overwrite: Whether to overwrite the destination file if it already exists, defaults to `False`.
        :type overwrite: class:`bool`

        :raises RuntimeError: If a file already exists at the destination path and :param:`overwrite` is `False`.
        """
        # construct file name
        if model_prefix is None:
            model_prefix = ""
        else:
            model_prefix = f"{model_prefix}_"
        file_name = f"{model_prefix}{self.__class__.__name__}.pkl"

        # defining path
        dump_path = os.path.join(dump_dir, file_name)

        # checking that the fixle exists
        if os.path.exists(dump_path):
            if not overwrite:
                msg = f""
                raise RuntimeError(msg)
            msg = f""
            logger.warning(msg)

        # saving the model
        with open(dump_path, "wb") as fp:
            cloudpickle.dump(self, fp)

    @classmethod
    def load(
        cls,
        file_name: str,
    ) -> "FlowMatching":
        """Loads a pickled model instance from disk.

        :param file_name: Path to the pickled model file to load.
        :type file_name: class:`str`

        :return: The deserialized model instance.
        :rtype: class:`FlowMatching`

        :raises TypeError: If the deserialized object is not an instance of the class :method:`load` was called on.
        """
        # loading model file
        with open(file_name, "rb") as fp:
            model = cloudpickle.load(fp)
        
        # veriying types
        if type(model) is not cls:
            msg = f""
            raise TypeError(msg)
        
        return model
