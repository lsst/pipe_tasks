# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.

import unittest

from lsst.utils.tests import ImportTestCase


class PipeTasksImportTestCase(ImportTestCase):
    """Test that every file can be imported.

    pipe_tasks does not import all the task code by default and not
    every file currently has a test.
    """

    PACKAGES = {
        "lsst.pipe.tasks",
        "lsst.pipe.tasks.script",
        "lsst.pipe.tasks.dataFrameActions",
    }

    SKIP_FILES = {
        "lsst.pipe.tasks": {
            "make_direct_warp.py",  # TODO: Remove in DM-47521.
            "make_psf_matched_warp.py",  # TODO: Remove in DM-47521.
        }
    }


if __name__ == "__main__":
    unittest.main()
