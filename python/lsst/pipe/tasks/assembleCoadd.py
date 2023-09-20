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

# TODO: Remove this entire file in DM-40826.

import warnings

try:
    from lsst.drp.tasks.assemble_coadd import *  # noqa: F401, F403
except ImportError as error:
    error.msg += ". Please import the coaddition tasks from drp_tasks package."
    raise error
finally:
    warnings.warn("lsst.pipe.tasks.assembleCoadd is deprecated and will be removed after v27; "
                  "Please use lsst.drp.tasks.assemble_coadd instead.",
                  DeprecationWarning,
                  stacklevel=2
                  )
