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

# TODO: Remove this entire file in DM-47521.

import warnings

try:
    from lsst.drp.tasks.make_direct_warp import MakeDirectWarpConfig, MakeDirectWarpTask  # noqa: F401
except ImportError as error:
    error.msg += ". Please import the warping tasks from drp_tasks package."
    raise error
finally:
    warnings.warn(
        "lsst.pipe.tasks.make_direct_warp is deprecated and will be removed after v29; "
        "Please use lsst.drp.tasks.make_direct_warp instead.",
        DeprecationWarning,
        stacklevel=2,
    )
