import importlib
import importlib.abc
import importlib.util
import logging
import os
import sys

import cloudpickle

logger = logging.getLogger(__name__)

__all__ = ["BaseModel"]

_LEGACY_PACKAGE_NAME = "sc_exp_design"


class _LegacyPackageLoader(importlib.abc.Loader):
    """Redirects a legacy ``sc_exp_design.*`` import to its ``labcompass`` equivalent."""

    def create_module(self, spec):
        new_name = "labcompass" + spec.name[len(_LEGACY_PACKAGE_NAME) :]
        return importlib.import_module(new_name)

    def exec_module(self, module):
        pass


class _LegacyPackageFinder(importlib.abc.MetaPathFinder):
    """Finds imports of the old ``sc_exp_design`` package name, renamed to ``labcompass``."""

    def find_spec(self, fullname, path, target=None):
        if fullname == _LEGACY_PACKAGE_NAME or fullname.startswith(f"{_LEGACY_PACKAGE_NAME}."):
            return importlib.util.spec_from_loader(fullname, _LegacyPackageLoader())
        return None


def _install_legacy_module_alias() -> None:
    """Ensures old ``sc_exp_design``-pickled checkpoints can still be unpickled.

    ``cloudpickle`` stores each class's exact module path at save time, so
    checkpoints saved before the ``sc_exp_design`` -> ``labcompass`` rename
    reference a module that no longer exists. This installs an import
    redirect so those lookups resolve to the renamed ``labcompass`` modules.
    """
    if any(isinstance(finder, _LegacyPackageFinder) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _LegacyPackageFinder())


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
        _install_legacy_module_alias()
        with open(file_name, "rb") as fp:
            model = cloudpickle.load(fp)
        
        # veriying types
        if type(model) is not cls:
            msg = f""
            raise TypeError(msg)
        
        return model
