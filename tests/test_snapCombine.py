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
from lsst.pipe.tasks.snapCombine import SnapCombineTask

np.random.seed(1)


def makeRandomExposure(width, height, imMean, varMean, maxMask):
    """Make a random exposure with Poisson distribution for image and variance

    @param[in] width image width (pixels)
    @param[in] height image height (pixels)
    @param[in] imMean mean of image plane
    @param[in] varMean mean of variance plane
    @param[in] maxMask maximum mask value; values will be uniformly chosen in [0, maxMask]
    """
    exp = afwImage.ExposureF(width, height)
    exp.image.array[:, :] = np.random.poisson(imMean, size=exp.image.array.shape)
    exp.variance.array[:, :] = np.random.poisson(varMean, size=exp.variance.array.shape)
    exp.mask.array[:, :] = np.random.randint(0, maxMask + 1, size=exp.mask.array.shape)

    return exp


def simpleAdd(exp0, exp1, badPixelMask):
    """Add two exposures, avoiding bad pixels
    """
    expRes = exp0.Factory(exp0, True)
    weightMap = afwImage.ImageF(exp0.getDimensions())

    good0 = np.bitwise_and(exp0.mask.array, badPixelMask) == 0
    good1 = np.bitwise_and(exp1.mask.array, badPixelMask) == 0

    expRes.image.array[:, :] = np.where(good0, exp0.image.array, 0) + np.where(good1, exp1.image.array, 0)
    expRes.variance.array[:, :] = np.where(good0, exp0.variance.array, 0) + \
        np.where(good1, exp1.variance.array, 0)
    expRes.mask.array[:, :] = np.bitwise_or(np.where(good0, exp0.mask.array, 0),
                                            np.where(good1, exp1.mask.array, 0))
    weightMap.array[:, :] = np.where(good0, 1, 0) + np.where(good1, 1, 0)

    expRes.maskedImage /= weightMap
    expRes.maskedImage *= 2  # want addition, not mean, where both pixels are valid

    setCoaddEdgeBits(expRes.mask, weightMap)

    return expRes


class SnapCombineTestCase(lsst.utils.tests.TestCase):

    """A test case for SnapCombineTask."""

    def testAddition(self):
        """Test addition with bad pixels
        """
        config = SnapCombineTask.ConfigClass()
        config.doRepair = False
        config.doDiffIm = False
        config.badMaskPlanes = ("BAD", "SAT", "NO_DATA", "CR")
        badPixelMask = afwImage.MaskX.getPlaneBitMask(config.badMaskPlanes)
        task = SnapCombineTask(config=config)

        snap0 = makeRandomExposure(25, 25, 10000, 5000, badPixelMask)
        snap1 = makeRandomExposure(25, 25, 10000, 5000, badPixelMask)
        resExp = task.run(snap0, snap1).exposure
        resMi = resExp.getMaskedImage()

        predExp = simpleAdd(snap0, snap1, badPixelMask)
        predMi = predExp.getMaskedImage()
        self.assertMaskedImagesAlmostEqual(resMi, predMi)

    def testAdditionAllGood(self):
        """Test the case where all pixels are valid
        """
        config = SnapCombineTask.ConfigClass()
        config.doRepair = False
        config.doDiffIm = False
        task = SnapCombineTask(config=config)

        snap0 = makeRandomExposure(25, 25, 10000, 5000, 0)
        snap1 = makeRandomExposure(25, 25, 10000, 5000, 0)
        resExp = task.run(snap0, snap1).exposure
        resMi = resExp.getMaskedImage()

        predMi = snap0.getMaskedImage().Factory(snap0.getMaskedImage(), True)
        predMi += snap1.getMaskedImage()
        self.assertMaskedImagesAlmostEqual(resMi, predMi)

    def testMetadata(self):
        """Test more advanced metadata handling
        """
        config = SnapCombineTask.ConfigClass()
        config.doRepair = False
        config.doDiffIm = False
        # the MISS<N> keys are missing from metadata<N>
        # and so cannot be summed or averaged
        # MISS0 keys will be copied without alterate
        # MISS1 keys will be missing from the result
        config.averageKeys = ("AVG0", "AVG1", "MISS0AVG", "MISS1AVG")
        config.sumKeys = ("SUM0", "SUM1", "MISS0SUM", "MISS1SUM")
        task = SnapCombineTask(config=config)

        snap0 = makeRandomExposure(5, 5, 10000, 5000, 0)
        snap1 = makeRandomExposure(5, 5, 10000, 5000, 0)

        metadata0 = snap0.getMetadata()
        metadata1 = snap1.getMetadata()
        metadata0.set("NUM0", 45.2)
        metadata0.set("ASTR", "this is a string")
        metadata0.set("AVG0", 10.5)
        metadata1.set("AVG0", 9.5)
        metadata0.set("AVG1", -0.7)
        metadata1.set("AVG1", 0.2)
        metadata0.set("MISS0AVG", 2.3523)
        metadata1.set("MISS1AVG", 8.23)
        metadata0.set("SUM0", 1.23)
        metadata1.set("SUM0", 4.56)
        metadata0.set("SUM1", 9814)
        metadata1.set("SUM1", 3)
        metadata0.set("MISS0SUM", 75.4)
        metadata1.set("MISS1SUM", -234.3)

        allKeys = set(metadata0.names()) | set(metadata1.names())
        miss0Keys = set(key for key in allKeys if key.startswith("MISS0"))
        miss1Keys = set(key for key in allKeys if key.startswith("MISS1"))
        missKeys = miss0Keys | miss1Keys
        avgKeys = set(config.averageKeys) - missKeys  # keys that will be averaged
        sumKeys = set(config.sumKeys) - missKeys  # keys that will be summed
        sameKeys = allKeys - (avgKeys | sumKeys | miss1Keys)  # keys that will be the same

        resExp = task.run(snap0, snap1).exposure
        resMetadata = resExp.getMetadata()

        for key in sameKeys:
            self.assertEqual(resMetadata.getScalar(key), metadata0.getScalar(key))
        for key in avgKeys:
            self.assertAlmostEqual(resMetadata.getScalar(key),
                                   (metadata0.getScalar(key) + metadata1.getScalar(key)) / 2.0)
        for key in sumKeys:
            self.assertAlmostEqual(resMetadata.getScalar(key),
                                   metadata0.getScalar(key) + metadata1.getScalar(key))
        for key in miss1Keys:
            self.assertFalse(resMetadata.exists(key))


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
