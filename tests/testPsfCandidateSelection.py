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
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig

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
        # test that all of the flags are defined and there is no reservation by default
        # also test that used sources are a subset of candidate sources
        task = CharacterizeImageTask()
        results = task.characterize(self.exposure)
        used = 0
        reserved = 0
        for source in results.sourceCat:
            if source.get("calib_psfUsed"):
                used += 1
                self.assertTrue(source.get("calib_psfCandidate"))
            if source.get("calib_psfReserved"):
                reserved += 1
        self.assertGreater(used, 0)
        self.assertEqual(reserved, 0)

    def testReserveFraction(self):
        # test that a fraction of the possible candidates can be reserved
        # and that different expIds can produce a different reserveLists
        task = CharacterizeImageTask()
        # set the reserve fraction, and see if the right proportion are reserved.
        task.measurePsf.config.reserveFraction = .3
        exposureIdInfo = ExposureIdInfo(12345, expBits=16)
        results = task.characterize(self.exposure, exposureIdInfo=exposureIdInfo)
        candidates = 0
        reservedSources1 = []
        for source in results.sourceCat:
            if source.get("calib_psfCandidate"):
                candidates += 1
            if source.get("calib_psfReserved"):
                reservedSources1.append(source.getId())
        reserved = len(reservedSources1)
        self.assertEqual(reserved, int(.3 * (candidates + reserved)))

        # try again with a different id, and see if the list is different
        # but the number of reserved sources is the same
        exposureIdInfo = ExposureIdInfo(6789, expBits=16)
        results = task.characterize(self.exposure, exposureIdInfo=exposureIdInfo)
        reservedSources2 = []
        for source in results.sourceCat:
            if source.get("calib_psfReserved"):
                reservedSources2.append(source.getId())
        # Length of reserveList should be the same, but specific sources may differ
        self.assertEqual(len(reservedSources1), len(reservedSources2))
        self.assertNotEqual(reservedSources1, reservedSources2)

    def testReserveSeedReproducible(self):
        # test that the same seed twice will produce the same reserve set
        task = CharacterizeImageTask()
        task.measurePsf.config.reserveFraction = .3
        results = task.characterize(self.exposure)
        reservedSources1 = []
        for source in results.sourceCat:
            if source.get("calib_psfReserved"):
                reservedSources1.append(source.getId())

        # try again with the same seed (the default)
        results = task.characterize(self.exposure)
        reservedSources2 = []
        for source in results.sourceCat:
            if source.get("calib_psfReserved"):
                reservedSources2.append(source.getId())
        # reserveLists should be the same
        self.assertEqual(reservedSources1, reservedSources2)

class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
