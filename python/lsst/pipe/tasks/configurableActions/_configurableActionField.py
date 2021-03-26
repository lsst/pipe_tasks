# This file is part of pipe_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

__all__ = ("ConfigurableActionField",)

from lsst.pex.config import ConfigField, FieldValidationError
from lsst.pex.config.config import _typeStr, _joinNamePath
from lsst.pex.config.callStack import getCallStack

from . import ConfigurableAction


class ConfigurableActionField(ConfigField):
    """`ConfigurableActionField` is a subclass of `~lsst.pex.config.Field` that
    allows a single `ConfigurableAction` (or a subclass of thus) to be
    assigned to it. The `ConfigurableAction` is then accessed through this
    field for further configuration.

    Any configuration that is done prior to reasignment to a new
    `ConfigurableAction` is forgotten.
    """
    # These attributes are dynamically assigned when constructing the base
    # classes
    name: str

    def __set__(self, instance, value, at=None, label="assignment"):
        if instance._frozen:
            raise FieldValidationError(self, instance,
                                       "Cannot modify a frozen Config")
        name = _joinNamePath(prefix=instance._name, name=self.name)

        if not isinstance(value, self.dtype) and not issubclass(value, self.dtype):
            msg = f"Value {value} is of incorrect type {_typeStr(value)}. Expected {_typeStr(self.dtype)}"
            raise FieldValidationError(self, instance, msg)

        if at is None:
            at = getCallStack()

        if isinstance(value, self.dtype):
            instance._storage[self.name] = type(value)(__name=name, __at=at,
                                                       __label=label, **value._storage)
        else:
            instance._storage[self.name] = value(__name=name, __at=at, __label=label)
        history = instance._history.setdefault(self.name, [])
        history.append(("config value set", at, label))

    def __init__(self, doc, dtype=ConfigurableAction, default=None, check=None, deprecated=None):
        if not issubclass(dtype, ConfigurableAction):
            raise ValueError("dtype must be a subclass of ConfigurableAction")
        super().__init__(doc=doc, dtype=dtype, default=default, check=check, deprecated=deprecated)
