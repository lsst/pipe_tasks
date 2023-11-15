#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
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
import unittest

import numpy as np

import lsst.utils.tests
import lsst.afw.image as afwImage
from lsst.coadd.utils import setCoaddEdgeBits
import lsst.meas.base.tests
from lsst.pipe.tasks.snapCombine import SnapCombineTask


def simple_add(exp0, exp1, bad_pixel_mask):
    """Add two exposures, avoiding bad pixels
    """
    result = exp0.Factory(exp0, True)
    weights = afwImage.ImageF(exp0.getDimensions())

    good0 = np.bitwise_and(exp0.mask.array, bad_pixel_mask) == 0
    good1 = np.bitwise_and(exp1.mask.array, bad_pixel_mask) == 0

    result.image.array[:, :] = np.where(good0, exp0.image.array, 0) + np.where(good1, exp1.image.array, 0)
    result.variance.array[:, :] = np.where(good0, exp0.variance.array, 0) + \
        np.where(good1, exp1.variance.array, 0)
    result.mask.array[:, :] = np.bitwise_or(np.where(good0, exp0.mask.array, 0),
                                            np.where(good1, exp1.mask.array, 0))
    weights.array[:, :] = np.where(good0, 1, 0) + np.where(good1, 1, 0)

    result.maskedImage /= weights
    result.maskedImage *= 2  # want addition, not mean, where both pixels are valid

    setCoaddEdgeBits(result.mask, weights)

    return result


class SnapCombineTestCase(lsst.utils.tests.TestCase):
    """A test case for SnapCombineTask."""
    def setUp(self):
        # Different x/y dimensions so they're easy to distinguish in a plot,
        # and non-zero minimum, to help catch xy0 errors.
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(5, 4), lsst.geom.Point2I(205, 184))

        center = lsst.geom.Point2D(50, 60)
        # Two TestDatasets, as that is where VisitInfo is created.
        dataset = lsst.meas.base.tests.TestDataset(bbox, visitId=12345)
        dataset.addSource(10000, center)
        schema = dataset.makeMinimalSchema()
        self.snap0, _ = dataset.realize(noise=1001, schema=schema)
        self.snap0.mask.array[10, 10] |= afwImage.Mask.getPlaneBitMask("BAD")
        dataset = lsst.meas.base.tests.TestDataset(bbox, visitId=123456)
        dataset.addSource(10000, center)
        self.snap1, _ = dataset.realize(noise=1002, schema=schema)

    def testAddition(self):
        """Test addition with bad pixels
        """
        config = SnapCombineTask.ConfigClass()
        config.bad_mask_planes = ("BAD", "SAT", "NO_DATA", "CR")
        bad_pixel_mask = afwImage.MaskX.getPlaneBitMask(config.bad_mask_planes)
        task = SnapCombineTask(config=config)

        result = task.run(self.snap0, self.snap1).exposure

        expect = simple_add(self.snap0, self.snap1, bad_pixel_mask)
        self.assertMaskedImagesAlmostEqual(result.maskedImage, expect.maskedImage)
        self._check_visitInfo(result.visitInfo)

    def testAdditionAllGood(self):
        """Test the case where all pixels are valid
        """
        config = SnapCombineTask.ConfigClass()
        task = SnapCombineTask(config=config)

        result = task.run(self.snap0, self.snap1).exposure

        expect = self.snap0.maskedImage.Factory(self.snap0.maskedImage, True)
        expect += self.snap1.maskedImage
        self.assertMaskedImagesAlmostEqual(result.maskedImage, expect)




class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
