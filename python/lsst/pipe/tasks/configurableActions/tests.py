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

__all__ = (
    "ActionTest1",
    "ActionTest2",
    "ActionTest3",
    "TestConfig",
)

from lsst.pex.config import Config, Field
from ._configurableAction import ConfigurableAction
from ._configurableActionStructField import ConfigurableActionStructField
from ._configurableActionField import ConfigurableActionField


class ActionTest1(ConfigurableAction):
    var = Field(doc="test field", dtype=int, default=0)

    def __call__(self):
        return self.var

    def validate(self):
        assert self.var is not None


class ActionTest2(ConfigurableAction):
    var = Field(doc="test field", dtype=int, default=1)

    def __call__(self):
        return self.var

    def validate(self):
        assert self.var is not None


class ActionTest3(ConfigurableAction):
    var = Field(doc="test field", dtype=int, default=3)

    def __call__(self):
        return self.var

    def validate(self):
        assert self.var is not None


class TestConfig(Config):
    actions = ConfigurableActionStructField(doc="Actions to be tested", default=None)
    singleAction = ConfigurableActionField(doc="A configurable action", default=None)
