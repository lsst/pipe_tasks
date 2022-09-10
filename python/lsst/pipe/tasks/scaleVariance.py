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

__all__ = ["ScaleVarianceConfig", "ScaleVarianceTask"]

from deprecated.sphinx import deprecated

import lsst.meas.algorithms


class ScaleVarianceConfig(lsst.meas.algorithms.ScaleVarianceConfig):
    pass


@deprecated(
    reason="Please use lsst.meas.algorithms.ScaleVarianceTask instead. Will be removed after v24.",
    version="v24.0",
    category=FutureWarning,
)
class ScaleVarianceTask(lsst.meas.algorithms.ScaleVarianceTask):
    """Scale the variance in a MaskedImage

    This version of ``ScaleVarianceTask`` is deprecated, and the Task
    in ``lsst.meas.algorithms`` should be used instead.
    """
    _DefaultName = "scaleVarianceDeprecated"
