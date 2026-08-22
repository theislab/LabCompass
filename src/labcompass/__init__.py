from importlib.metadata import version

from . import config, models, utils, inverse, sym

__all__ = ["config", "models", "utils"]

__version__ = version("LabCompass")
