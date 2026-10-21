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

"""Tests for RepairTask cosmic ray finding."""
import unittest

import numpy as np

import lsst.afw.detection as afwDet
import lsst.afw.image as afwImage
import lsst.geom
import lsst.utils.tests
from lsst.pipe.tasks.repair import RepairTask


class RepairTaskTestCase(lsst.utils.tests.TestCase):
    """Test RepairTask.cosmicRay with and without a local background model."""

    def setUp(self):
        self.sky = 100.0
        self.gain = 2.0
        self.crValue = 5000.0
        self.crPosition = (40, 60)  # x, y
        rng = np.random.default_rng(12345)
        size = 200
        exposure = afwImage.ExposureF(lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                                                      lsst.geom.Extent2I(size, size)))
        image = exposure.image.array
        image[:] = rng.normal(self.sky, np.sqrt(self.sky/self.gain), size=(size, size))
        x, y = self.crPosition
        image[y, x] = self.crValue
        exposure.variance.array[:] = image/self.gain
        exposure.setPsf(afwDet.GaussianPsf(11, 11, 1.5))
        self.exposure = exposure
        self.originalMedian = np.median(image)

    def _checkRepaired(self, exposure):
        x, y = self.crPosition
        crBit = exposure.mask.getPlaneBitMask("CR")
        self.assertEqual(exposure.mask.array[y, x] & crBit, crBit)
        self.assertLess(exposure.image.array[y, x], 10*np.sqrt(self.sky/self.gain) + self.sky)
        # The background must not be changed by the repair.
        self.assertFloatsAlmostEqual(np.median(exposure.image.array), self.originalMedian, rtol=0.02)

    def testConstantBackground(self):
        """With the default binSize (larger than the image) a single median
        background is used."""
        config = RepairTask.ConfigClass()
        config.doInterpolate = False
        task = RepairTask(config=config)
        task.run(self.exposure)
        self._checkRepaired(self.exposure)

    def testLocalBackground(self):
        """With binSize smaller than the image a background model is
        subtracted before finding cosmic rays and added back afterwards."""
        config = RepairTask.ConfigClass()
        config.doInterpolate = False
        config.cosmicray.background.binSize = 64
        task = RepairTask(config=config)
        task.run(self.exposure)
        self._checkRepaired(self.exposure)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
