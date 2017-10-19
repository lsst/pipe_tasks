#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2017 AURA/LSST
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

from __future__ import division, print_function, absolute_import
import os
import unittest

import lsst.afw.image as afwImage
import lsst.utils.tests
from lsst.utils import getPackageDir
from lsst.obs.base import ExposureIdInfo
from lsst.log import Log
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask


class PsfFlagTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        # Load sample input from disk
        expPath = os.path.join(getPackageDir("pipe_tasks"), "tests", "data", "v695833-e0-c000-a00.sci.fits")
        self.exposure = afwImage.ExposureF(expPath)
        # set log level so that warnings do not display
        Log.getLogger("characterizeImage").setLevel(Log.ERROR)

    def tearDown(self):
        del self.exposure

    def testFlags(self):
        # Test that all of the flags are defined and there is no reservation by default
        # also test that used sources are a subset of candidate sources
        task = CharacterizeImageTask()
        results = task.characterize(self.exposure)
        used = 0
        reserved = 0
        for source in results.sourceCat:
            if source.get("calib_psfUsed"):
                used += 1
                self.assertTrue(source.get("calib_psfCandidate"))
            if source.get("calib_psf_reserved"):
                reserved += 1
        self.assertGreater(used, 0)
        self.assertEqual(reserved, 0)

class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
