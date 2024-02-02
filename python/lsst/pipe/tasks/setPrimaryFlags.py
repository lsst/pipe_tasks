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

__all__ = ["SetPrimaryFlagsConfig", "SetPrimaryFlagsTask"]

from deprecated.sphinx import deprecated

from lsst.meas.algorithms import setPrimaryFlags
from lsst.pex.config import Field, ListField


# Remove this entire file on DM-42962.
@deprecated(reason="This class has been moved to meas_algorithms. Will be removed after v27.",
            version="v27.0", category=FutureWarning)
class SetPrimaryFlagsConfig(setPrimaryFlags.SetPrimaryFlagsConfig):
    nChildKeyName = Field(dtype=str, default="deprecated",
                          doc="Deprecated. This parameter is not used.",
                          deprecated="This parameter is not used. Will be removed after v27.")
    pseudoFilterList = ListField(
        dtype=str,
        default=['sky'],
        doc="Names of filters which should never be primary",
        deprecated="This class has been moved to meas_algorithms. Will be removed after v27."
    )


@deprecated(reason="This class has been moved to meas_algorithms. Will be removed after v27.",
            version="v27.0", category=FutureWarning)
class SetPrimaryFlagsTask(setPrimaryFlags.SetPrimaryFlagsTask):
    pass
