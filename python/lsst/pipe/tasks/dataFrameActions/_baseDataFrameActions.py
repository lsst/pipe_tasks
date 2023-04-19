from __future__ import annotations

__all__ = ("DataFrameAction",)

from lsst.pex.config import Field, ListField
from typing import Iterable, Any, Mapping

from lsst.pex.config.configurableActions import ConfigurableAction


class DataFrameAction(ConfigurableAction):
    _actionCache: Mapping[int, Any]

    cache = Field(doc="Controls if the results of this action should be cached,"
                      " only works on frozen actions",
                  dtype=bool, default=False)
    cacheArgs = ListField(doc="If cache is True, this is a list of argument keys that will be used to "
                              "compute the cache key in addition to the DataFrameId",
                          dtype=str, optional=True)

    def __init_subclass__(cls, **kwargs) -> None:
        cls._actionCache = {}

        def call_wrapper(function):
            def inner_wrapper(self, dataFrame, **kwargs):
                dfId = id(dataFrame)
                extra = []
                for name in (self.cacheArgs or tuple()):
                    if name not in kwargs:
                        raise ValueError(f"{name} is not part of call signature and cant be used for "
                                         "caching")
                    extra.append(kwargs[name])
                extra.append(dfId)
                key = tuple(extra)
                if self.cache and self._frozen:
                    # look up to see if the value is in cache already
                    if result := self._actionCache.get(key):
                        return result
                result = function(self, dataFrame, **kwargs)
                if self.cache and self._frozen:
                    self._actionCache[key] = result
                return result
            return inner_wrapper
        cls.__call__ = call_wrapper(cls.__call__)
        super().__init_subclass__(**kwargs)

    def __call__(self, dataFrame, **kwargs) -> Iterable[Any]:
        """This method should return the result of an action performed on a
        dataframe
        """
        raise NotImplementedError("This method should be overloaded in a subclass")

    @property
    def columns(self) -> Iterable[str]:
        """This property should return an iterable of columns needed by this action
        """
        raise NotImplementedError("This method should be overloaded in a subclass")
