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

__all__ = ["ConfigurableAction", "ActionTypeVar"]

from typing import Any, TypeVar

from lsst.pex.config.config import Config


ActionTypeVar = TypeVar("ActionTypeVar", bound='ConfigurableAction')


class ConfigurableAction(Config):
    """A `ConfigurableAction` is an interface that extends a
    `lsst.pex.config.Config` class to include a `__call__` method.

    This interface is designed to create an action that can be used at
    runtime with state that is determined during the configuration stage. A
    single action thus may be assigned multiple times, each with different
    configurations.

    This allows state to be set and recorded at configuration time,
    making future reproduction of results easy.

    This class is intended to be an interface only, but because of various
    inheritance conflicts this class can not be implemented as an Abstract
    Base Class. Instead, the `__call__` method is purely virtual, meaning that
    it will raise a `NotImplementedError` when called. Subclasses that
    represent concrete actions must provide an override.
    """

    identity: str | None = None
    """If a configurable action is assigned to a `ConfigurableActionField`, or a
    `ConfigurableActionStructField` the name of the field will be bound to this
    variable when it is retrieved.
    """

    def __setattr__(self, attr, value, at=None, label="assignment"):
        if attr == 'identity':
            return object.__setattr__(self, attr, value)
        return super().__setattr__(attr, value, at, label)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("This method should be overloaded in subclasses")
