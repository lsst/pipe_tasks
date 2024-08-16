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

import os

import unittest
import numpy as np

import lsst.afw.image as afwImage
import lsst.utils
import lsst.utils.tests
from lsst.pipe.tasks.quickFrameMeasurement import QuickFrameMeasurementTask, QuickFrameMeasurementTaskConfig


class QuickFrameMeasurementTaskTestCase(lsst.utils.tests.TestCase):
    try:
        afwDataDir = lsst.utils.getPackageDir('afwdata')
    except Exception:
        afwDataDir = None

    truthValuesDirect = {
        "postISRCCD_2020021800073-KPNO_406_828nm~EMPTY-det000.fits.fz": (2496.296, 1104.171),
        "postISRCCD_2020021800027-KPNO_406_828nm~EMPTY-det000.fits.fz": (1540.807, 1421.422),
        "postISRCCD_2020021800224-EMPTY~EMPTY-det000.fits.fz": (1865.814, 2273.021),
    }
    truthValuesDispersed = {
        "postISRCCD_2020021700249-EMPTY~ronchi90lpmm-det000.fits.fz": (2531.088, 2287.149),
        "postISRCCD_2020031500119-EMPTY~ronchi90lpmm-det000.fits.fz": (2112.865, 2181.377),
    }

    CENTROID_TOLERANCE = 2  # number of pixels total distance it's acceptable to miss by
    COM_TOLERANCE = 8  # number of pixels the center of mass is allowed to be off by

    @unittest.skipUnless(afwDataDir, "afwdata not available")
    def setUp(self):
        self.directConfig = QuickFrameMeasurementTaskConfig()
        self.directTask = QuickFrameMeasurementTask(config=self.directConfig)

        # support for handling dispersed images seperately in future via config
        self.dispersedConfig = QuickFrameMeasurementTaskConfig()
        self.dispersedTask = QuickFrameMeasurementTask(config=self.dispersedConfig)

    @unittest.skipUnless(afwDataDir, "afwdata not available")
    def testDirectCentroiding(self):
        task = self.directTask
        filenames = self.truthValuesDirect.keys()

        for filename in filenames:
            fullName = os.path.join(self.afwDataDir, "LATISS/postISRCCD", filename)
            trueCentroid = self.truthValuesDirect[filename]

            exp = afwImage.ExposureF(fullName)
            result = task.run(exp)
            foundCentroid = result.brightestObjCentroid

            dist = np.linalg.norm(np.asarray(foundCentroid) - np.asarray(trueCentroid))
            distCoM = np.linalg.norm(np.asarray(result.brightestObjCentroidCofM) - np.asarray(trueCentroid))

            self.assertLess(dist, self.CENTROID_TOLERANCE)
            self.assertLess(distCoM, self.COM_TOLERANCE)

            # offset size shouldn't really matter, just make it >> PSF, and make
            # sure the value isn't off-chip or right by the edge for any of the
            # test images
            offset = 300
            wrongCentroid = (foundCentroid[0]+offset, foundCentroid[1]+offset)
            with self.assertRaises(ValueError):
                task.checkResult(exp, wrongCentroid, -1, self.directConfig.centroidPixelPercentile)

    @unittest.skipUnless(afwDataDir, "afwdata not available")
    def testDispersedCentroiding(self):
        task = self.dispersedTask
        filenames = self.truthValuesDispersed.keys()

        for filename in filenames:
            fullName = os.path.join(self.afwDataDir, "LATISS/postISRCCD", filename)
            trueCentroid = self.truthValuesDispersed[filename]

            exp = afwImage.ExposureF(fullName)
            result = task.run(exp)
            foundCentroid = result.brightestObjCentroid

            dist = np.linalg.norm(np.asarray(foundCentroid) - np.asarray(trueCentroid))
            self.assertLess(dist, self.CENTROID_TOLERANCE)

            # this doesn't matter much as CoM centroid is for donuts
            # and we don't intend to do dispersed donuts, but good to catch
            # big regressions. Can be removed later if needed
            distCoM = np.linalg.norm(np.asarray(result.brightestObjCentroidCofM) - np.asarray(trueCentroid))
            self.assertLess(distCoM, self.COM_TOLERANCE)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
