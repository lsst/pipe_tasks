from __future__ import annotations

__all__ = ("DataFrameAction",)

from typing import Iterable, Any

from ..configurableActions import ConfigurableAction


class DataFrameAction(ConfigurableAction):
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

    @classmethod
    def fromArgs(cls, **kwargs):
        """This supports directly creating an action from a script or notebook
        outside of a config hierarchy.
        """
        self = cls()
        for field in (self._fields.keys() & kwargs.keys()):
            setattr(self, field, kwargs[field])

        return self
