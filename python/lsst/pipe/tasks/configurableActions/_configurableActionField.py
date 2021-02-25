from __future__ import annotations

__all__ = ("ConfigurableActionField",)


from lsst.pex.config import ConfigField, FieldValidationError
from lsst.pex.config.config import _typeStr, _joinNamePath
from lsst.pex.config.callStack import getCallStack

from . import ConfigurableAction


class ConfigurableActionField(ConfigField):
    def __set__(self, instance, value, at=None, label="assignment"):
        if instance._frozen:
            raise FieldValidationError(self, instance,
                                       "Cannot modify a frozen Config")
        name = _joinNamePath(prefix=instance._name, name=self.name)

        if not issubclass(value, self.dtype) and not isinstance(value, self.dtype):
            msg = "Value %s is of incorrect type %s. Expected %s" % \
                (value, _typeStr(value), _typeStr(self.dtype))
            raise FieldValidationError(self, instance, msg)

        if at is None:
            at = getCallStack()

        oldValue = instance._storage.get(self.name, None)
        if oldValue is None:
            if issubclass(value, self.dtype):
                instance._storage[self.name] = self.dtype(__name=name, __at=at, __label=label)
            else:
                instance._storage[self.name] = self.dtype(__name=name, __at=at,
                                                          __label=label, **value._storage)
        else:
            if issubclass(value, self.dtype):
                value = value()
            oldValue.update(__at=at, __label=label, **value._storage)
        history = instance._history.setdefault(self.name, [])
        history.append(("config value set", at, label))

    def __init__(self, doc, dtype=ConfigurableAction, default=None, check=None, deprecated=None):
        super().__init__(doc=doc, dtype=dtype, default=default, check=check, deprecated=deprecated)
