from importlib.metadata import version

from . import config, models, utils

__all__ = ["config", "models", "utils"]

__version__ = version("scExpDesign")
