from __future__ import annotations

__all__ = ["ConfigurableAction"]

from typing import Any

from lsst.pex.config.config import Config


class ConfigurableAction(Config):
    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError("This method should be overloaded in subclasses")
