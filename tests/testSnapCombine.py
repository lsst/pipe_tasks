#!/usr/bin/env python

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

import numpy

import lsst.utils.tests as utilsTests
import lsst.afw.image as afwImage
import lsst.afw.image.testUtils as afwTestUtils
import lsst.afw.coord as afwCoord
import lsst.afw.geom as afwGeom
from lsst.coadd.utils import setCoaddEdgeBits
from lsst.pipe.tasks.snapCombine import SnapCombineTask

def makeRandomExposure(width, height, imMean, varMean, maxMask):
    """Make a random exposure with Poisson distribution for image and variance
    
    @param[in] width image width (pixels)
    @param[in] height image height (pixels)
    @param[in] imMean mean of image plane
    @param[in] varMean mean of variance plane
    @param[in] maxMask maximum mask value; values will be uniformly chosen in [0, maxMask]
    """
    exp = afwImage.ExposureF(width, height)
    mi = exp.getMaskedImage()
    imArr, maskArr, varArr = mi.getArrays()
    imArr[:,:] = numpy.random.poisson(imMean, size=imArr.shape)
    varArr[:,:] = numpy.random.poisson(varMean, size=varArr.shape)
    maskArr[:,:] = numpy.random.random_integers(0, maxMask, size=maskArr.shape)
    
    return exp

def simpleAdd(exp0, exp1, badPixelMask):
    """Add two exposures, avoiding bad pixels
    """
    imArr0, maskArr0, varArr0 = exp0.getMaskedImage().getArrays()
    imArr1, maskArr1, varArr1 = exp1.getMaskedImage().getArrays()
    expRes = exp0.Factory(exp0, True)
    miRes = expRes.getMaskedImage()
    imArrRes, maskArrRes, varArrRes = miRes.getArrays()
    
    weightMap = afwImage.ImageF(exp0.getDimensions())
    weightArr = weightMap.getArray()

    good0 = numpy.bitwise_and(maskArr0, badPixelMask) == 0
    good1 = numpy.bitwise_and(maskArr1, badPixelMask) == 0

    imArrRes[:,:]  = numpy.where(good0,  imArr0, 0) + numpy.where(good1,  imArr1, 0)
    varArrRes[:,:] = numpy.where(good0, varArr0, 0) + numpy.where(good1, varArr1, 0)
    maskArrRes[:,:] = numpy.bitwise_or(numpy.where(good0, maskArr0, 0), numpy.where(good1, maskArr1, 0))
    weightArr[:,:] = numpy.where(good0, 1, 0) + numpy.where(good1, 1, 0)
    
    miRes /= weightMap
    miRes *= 2 # want addition, not mean, where both pixels are valid

    setCoaddEdgeBits(miRes.getMask(), weightMap)    
    
    return expRes
    

class SnapCombineTestCase(unittest.TestCase):
    """A test case for SnapCombineTask."""
    def testAddition(self):
        """Test addition with bad pixels
        """
        config = SnapCombineTask.ConfigClass()
        config.doRepair = False
        config.doDiffIm = False
        config.badMaskPlanes = ("BAD", "SAT", "NO_DATA", "CR")
        badPixelMask = afwImage.MaskU.getPlaneBitMask(config.badMaskPlanes)
        task = SnapCombineTask(config=config)

        snap0 = makeRandomExposure(25, 25, 10000, 5000, badPixelMask)
        snap1 = makeRandomExposure(25, 25, 10000, 5000, badPixelMask)
        resExp = task.run(snap0, snap1).exposure
        resMi = resExp.getMaskedImage()
        
        predExp = simpleAdd(snap0, snap1, badPixelMask)
        predMi = predExp.getMaskedImage()
        errMsg = afwTestUtils.maskedImagesDiffer(resMi.getArrays(), predMi.getArrays())
        if errMsg:
            self.fail(errMsg)
    
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
        errMsg = afwTestUtils.maskedImagesDiffer(resMi.getArrays(), predMi.getArrays())
        if errMsg:
            self.fail(errMsg)
    
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
        avgKeys = set(config.averageKeys) - missKeys # keys that will be averaged
        sumKeys = set(config.sumKeys) - missKeys # keys that will be summed
        sameKeys = allKeys - (avgKeys | sumKeys | miss1Keys) # keys that will be the same
        
        resExp = task.run(snap0, snap1).exposure
        resMetadata = resExp.getMetadata()
        
        for key in sameKeys:
            self.assertEqual(resMetadata.get(key), metadata0.get(key))
        for key in avgKeys:
            self.assertAlmostEqual(resMetadata.get(key), ((metadata0.get(key) + metadata1.get(key)) / 2.0))
        for key in sumKeys:
            self.assertAlmostEqual(resMetadata.get(key), (metadata0.get(key) + metadata1.get(key)))
        for key in miss1Keys:
            self.assertFalse(resMetadata.exists(key))

def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(SnapCombineTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
