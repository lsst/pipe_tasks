#
# LSST Data Management System
# Copyright 2008-2017 AURA/LSST.
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
# see <https://www.lsstcorp.org/LegalNotices/>.

import os

import unittest
from scipy.spatial import distance

import lsst.afw.image as afwImage
import lsst.utils.tests
from lsst.pipe.tasks.quickFrameMeasurement import QuickFrameMeasurement, QuickFrameMeasurementConfig


class QuickFrameMeasurementTaskTestCase(lsst.utils.tests.TestCase):
    try:
        afwDataDir = lsst.utils.getPackageDir('afwdata')
    except Exception:
        afwDataDir = None

    truthValues = {"postISRCCD_2020021800073-KPNO_406_828nm~EMPTY-det000.fits.fz": (2496, 1105),
                   }

    TOLERANCE = 5  # number of pixels total distance it's acceptable to miss by

    @unittest.skipUnless(afwDataDir, "afwdata not available")
    def setUp(self):
        self.directConfig = QuickFrameMeasurementConfig()
        self.directTask = QuickFrameMeasurement(config=self.directConfig)

        self.dispersedConfig = QuickFrameMeasurementConfig()
        self.dispersedConfig.imageIsDispersed = True
        self.dispersedTask = QuickFrameMeasurement(config=self.dispersedConfig)

    @unittest.skipUnless(afwDataDir, "afwdata not available")
    def testDirectCentroiding(self):
        task = self.directTask
        filenames = ["postISRCCD_2020021800073-KPNO_406_828nm~EMPTY-det000.fits.fz"]

        for filename in filenames:
            fullName = os.path.join(self.afwDataDir, "LATISS/postISRCCD", filename)
            trueCentroid = self.truthValues[filename]

            exp = afwImage.ExposureF(fullName)
            result = task.run(exp)
            foundCentroid = result.brightestObjCentroid

            dist = distance.euclidean(foundCentroid, trueCentroid)
            self.assertLess(dist, self.TOLERANCE)

    @unittest.skipUnless(afwDataDir, "afwdata not available")
    def testDispersedCentroiding(self):

        pass


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
