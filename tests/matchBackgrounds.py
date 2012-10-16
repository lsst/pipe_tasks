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
import lsst.utils.tests as utilsTests
import lsst.pipe.tasks as pipeTasks
import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import numpy
from lsst.pipe.tasks.matchBackgrounds import MatchBackgroundsTask

class MatchBackgroundsTestCase(unittest.TestCase):
    """Background Matching"""

    def setUp(self):
        #Make a few test images here
        #1) with chip gap
        self.chipGap = afwImage.ExposureF(600,600)
        im = self.chipGap.getMaskedImage().getImage()
        afwMath.randomGaussianImage(im,afwMath.Random(1))
        im += 10
        im.getArray()[:,200:250] = numpy.nan
        self.chipGap.getMaskedImage().getVariance().set(1.0)
        
        #2) with low coverage
        self.lowCover = afwImage.ExposureF(600,600)
        im = self.lowCover.getMaskedImage().getImage()
        afwMath.randomGaussianImage(im,afwMath.Random(2))
        im += 20
        self.lowCover.getMaskedImage().getImage().getArray()[:,200:] = numpy.nan
        self.lowCover.getMaskedImage().getVariance().set(1.0)

        #3 with wrong size
        self.wrongSize = afwImage.ExposureF(500,500)
        self.wrongSize.getMaskedImage().getImage().set(1.0)
        self.wrongSize.getMaskedImage().getVariance().set(1.0)

        #make two matchBackgrounds Objects
        self.matcher =  pipeTasks.matchBackgrounds.MatchBackgroundsTask()
        self.matcher.config.useDetectionBackground = False      

        self.sctrl = afwMath.StatisticsControl()
        self.sctrl.setNanSafe(True)
        self.sctrl.setAndMask(afwImage.MaskU.getPlaneBitMask(["EDGE", "DETECTED", "DETECTED_NEGATIVE","SAT","BAD","INTRP","CR"]))
        
    def tearDown(self):
        self.wrongSize = None
        self.lowCover = None
        self.chipGap = None
        self.matcher = None
        self.sctrl = None
        

    #TEST WORK AROUND
    #1) Should throw ValueError if Order > # of grid points
    #2) Should throw ValueError if binsize is > size of image
    def testConfig(self):
        print "testing Config"
        #binsize must be <= dimensions
        self.matcher.config.chebBinSize = 1000
        self.assertRaises(ValueError,self.matcher.matchBackgrounds,self.chipGap, self.lowCover)
        #Order must be > npoints
        self.matcher.config.chebBinSize = 256
        self.matcher.config.backgroundOrder = 8
        self.assertRaises(ValueError,self.matcher.matchBackgrounds,self.chipGap, self.lowCover)

    #3)  Should throw RuntimeError if dimensions don't match
    def testInputParams(self):
        print "testing input dimensions"
        self.matcher.config.chebBinSize = 128
        self.matcher.config.backgroundOrder = 4
        self.assertRaises(RuntimeError,self.matcher.matchBackgrounds,self.chipGap, self.wrongSize)
        
    def testAccuracy(self):
        print "testing basic accuracy"
        testExp = afwImage.ExposureF(self.chipGap, True)
        im = testExp.getMaskedImage().getImage()
        #afwMath.randomGaussianImage(im,afwMath.Random(3))
        
        im += 20
        self.matcher.config.chebBinSize = 128
        self.matcher.config.backgroundOrder = 4
        resultModel, resultExp = self.matcher.matchBackgrounds(self.chipGap, testExp)     
        stats = afwMath.makeStatistics(resultExp.getMaskedImage(), afwMath.MEAN |afwMath.VARIANCE,self.sctrl)
        mean, _ = stats.getResult(afwMath.MEAN)
        var, _ = stats.getResult(afwMath.VARIANCE)
        self.assertAlmostEqual(mean, 10.,delta = var)
        self.assertLess(var,2)

        
        #5) Should be able to handle nans
        #     a) in chip gaps
        #     b) in most of the image (image covers small % of patch)        
    def testNansThrowExpection(self):
        print "testing low coverage  THROW_EXCEPTION"
        #Test 3 options if too much of the image is missing
        self.matcher.config.chebBinSize = 256
        self.matcher.config.backgroundOrder = 4
        self.matcher.config.undersampleStyle =  "THROW_EXCEPTION"
        self.assertRaises(ValueError,self.matcher.matchBackgrounds,self.chipGap, self.lowCover)

    def testNansIncreaseSample(self):
        print "testing low coverage INCREASE_NXNYSAMPLE"
        self.matcher.config.chebBinSize = 256
        self.matcher.config.backgroundOrder = 4        
        self.matcher.config.undersampleStyle = "INCREASE_NXNYSAMPLE"
        resultModel, resultExp = self.matcher.matchBackgrounds(self.chipGap, self.lowCover)
        stats = afwMath.makeStatistics(resultExp.getMaskedImage(), afwMath.MEAN |afwMath.VARIANCE,self.sctrl)
        mean, _ = stats.getResult(afwMath.MEAN)
        var, _ = stats.getResult(afwMath.VARIANCE)
        self.assertAlmostEqual(mean, 10.,delta = var)
        self.assertLess(var,2)

    def testNansReduceInterpOrder(self):
        print "testing low coverage REDUCE_INTERP_ORDER"
        self.matcher.config.chebBinSize = 256
        self.matcher.config.backgroundOrder = 4        
        self.matcher.config.undersampleStyle =  "REDUCE_INTERP_ORDER"
        resultModel, resultExp = self.matcher.matchBackgrounds(self.chipGap, self.lowCover)
        stats = afwMath.makeStatistics(resultExp.getMaskedImage(), afwMath.MEAN |afwMath.VARIANCE,self.sctrl)
        mean, _ = stats.getResult(afwMath.MEAN)
        var, _ = stats.getResult(afwMath.VARIANCE)
        self.assertAlmostEqual(mean, 10.,delta = var)
        self.assertLess(var,2)
        

    def testMasks(self):
        print "testing mask stats"
        testExpRef = afwImage.ExposureF(self.chipGap, True)
        testExp = afwImage.ExposureF(self.chipGap, True)
        im   = testExp.getMaskedImage().getImage()
        im += 10
        mask = testExp.getMaskedImage().getMask()
        satbit = mask.getPlaneBitMask('SAT')
        for i in range(0,200,20):
            mask.set(5,i,satbit)
            im.set(5,i, 65000)
        self.matcher.config.chebBinSize = 128
        self.matcher.config.backgroundOrder = 4    
        resultModel, resultExp = self.matcher.matchBackgrounds(testExpRef, testExp)
        stats = afwMath.makeStatistics(resultExp.getMaskedImage(), afwMath.MEAN |afwMath.VARIANCE,self.sctrl)
        mean, _ = stats.getResult(afwMath.MEAN)
        var, _ = stats.getResult(afwMath.VARIANCE)
        self.assertAlmostEqual(mean, 10.,delta = var)
        self.assertLess(var,2)
 
    #More tests:
        #4) MSE (mean((refExp - matchedExp)**2)) < Variance of diffIm * 1.05
        #6) Should be able to handle situation when most of image is masked 
        #7) What should it do if the two images are the same?


        #TEST BACKGROUND CLASS (FUTURE)
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""

    utilsTests.init()

    suites = []
    suites += unittest.makeSuite(MatchBackgroundsTestCase)
    suites += unittest.makeSuite(utilsTests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(shouldExit = False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
