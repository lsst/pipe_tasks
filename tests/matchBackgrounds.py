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
        #1) full coverage (plain vanilla image) has mean = 50counts
        self.vanilla = afwImage.ExposureF(600,600)
        im = self.vanilla.getMaskedImage().getImage()
        afwMath.randomGaussianImage(im,afwMath.Random(1))
        im += 50
        self.vanilla.getMaskedImage().getVariance().set(1.0)
        
        #2) has chip gap and mean = 10 counts
        self.chipGap = afwImage.ExposureF(600,600)
        im = self.chipGap.getMaskedImage().getImage()
        afwMath.randomGaussianImage(im,afwMath.Random(2))
        im += 10
        im.getArray()[:,200:300] = numpy.nan #simulate 100pix chip gap
        self.chipGap.getMaskedImage().getVariance().set(1.0)
        
        #3) has low coverage and mean = 20 counts
        self.lowCover = afwImage.ExposureF(600,600)
        im = self.lowCover.getMaskedImage().getImage()
        afwMath.randomGaussianImage(im,afwMath.Random(3))
        im += 20
        self.lowCover.getMaskedImage().getImage().getArray()[:,200:] = numpy.nan 
        self.lowCover.getMaskedImage().getVariance().set(1.0)

        #make a matchBackgrounds object
        self.matcher = pipeTasks.matchBackgrounds.MatchBackgroundsTask()
        self.matcher.config.usePolynomial = True
        self.matcher.binSize = 64
        self.matcher.debugDataIdString = 'Test Visit'

        self.sctrl = afwMath.StatisticsControl()
        self.sctrl.setNanSafe(True)
        self.sctrl.setAndMask(afwImage.MaskU.getPlaneBitMask(["NO_DATA", "DETECTED",
                                                              "DETECTED_NEGATIVE","SAT",
                                                              "BAD","INTRP","CR"]))
        
    def tearDown(self):
        self.vanilla = None
        self.lowCover = None
        self.chipGap = None
        self.matcher = None
        self.sctrl = None
        
    def checkAccuracy(self, refExp, sciExp):
        """Check for accurate background matching in the matchBackgrounds Method.
        
           To be called by tests expecting successful matching.
        """
        struct = self.matcher.matchBackgrounds(refExp, sciExp)
        resultExp = sciExp
        MSE = struct.matchedMSE
        diffImVar = struct.diffImVar
        RMS = struct.fitRMS
        
        resultStats = afwMath.makeStatistics(resultExp.getMaskedImage(),
                                             afwMath.MEAN | afwMath.VARIANCE,
                                             self.sctrl)
        resultMean, _ = resultStats.getResult(afwMath.MEAN)
        resultVar, _ = resultStats.getResult(afwMath.VARIANCE)
        refStats = afwMath.makeStatistics(refExp.getMaskedImage(), afwMath.MEAN | afwMath.VARIANCE,self.sctrl)
        refMean, _ = refStats.getResult(afwMath.MEAN)
        #print "refMean %.03f, resultMean %.03f, resultVar %.03f"%(refMean, resultMean, resultVar)
        self.assertAlmostEqual(refMean, resultMean, delta = resultVar) #very loose test.      
        #If MSE is within 1% of the variance of the difference image:  SUCCESS
        self.assertLess(MSE, diffImVar * 1.01)
    
                      
    #-=-=-=-=-=-=Test Polynomial Fit (Approximate class)-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def testConfig(self):
        """Test checks on the configuration.
        
            1) Should throw ValueError if binsize is > size of image
            2) Need Order + 1 points to fit. Should throw ValueError if Order > # of grid points
        """
        for size in range(601,1000,100):
            self.matcher.config.binSize = size
            self.assertRaises(ValueError,self.matcher.matchBackgrounds,self.chipGap, self.vanilla)
            
        #for image 600x600 and binsize 256 = 3x3 grid for fitting.  order 3,4,5...should fail
        self.matcher.config.binSize = 256
        for order in range(3,8):
            self.matcher.config.order = order
            self.assertRaises(ValueError,self.matcher.matchBackgrounds,self.chipGap, self.vanilla)
    
        for size, order in [(600,0), (300,1), (200,2), (100,5)]:
            self.matcher.config.binSize = size
            self.matcher.config.order = order
            self.checkAccuracy(self.chipGap, self.vanilla)
        
    def testInputParams(self):
        """Test throws RuntimeError when dimensions don't match."""
        self.matcher.config.binSize = 128
        self.matcher.config.order = 2
        #make image with wronge size
        wrongSize = afwImage.ExposureF(500,500)
        wrongSize.getMaskedImage().getImage().set(1.0)
        wrongSize.getMaskedImage().getVariance().set(1.0)
        self.assertRaises(RuntimeError,self.matcher.matchBackgrounds,self.chipGap, wrongSize)

    def testVanillaApproximate(self):
        """Test basic matching scenario with .Approximate."""
        self.matcher.config.binSize = 128
        self.matcher.config.order = 4
        self.checkAccuracy(self.chipGap, self.vanilla)

    def testRampApproximate(self):
        """Test basic matching of a linear gradient with Approximate."""
        self.matcher.config.binSize = 64
        testExp = afwImage.ExposureF(self.vanilla, True)
        testIm = testExp.getMaskedImage().getImage()
        afwMath.randomGaussianImage(testIm,afwMath.Random(5))
        nx, ny = testExp.getDimensions()
        dzdx, dzdy, z0 = 1, 2, 0.0
        for x in range(nx):
            for y in range(ny):
                z = testIm.get(x, y)
                testIm.set(x, y, z + dzdx * x + dzdy * y + z0)          
        self.checkAccuracy(testExp, self.vanilla)
        
    def testLowCoverThrowExpectionApproximate(self):
        """Test low coverage with .config.undersampleStyle = THROW_EXCEPTION.
        Confirm throws ValueError with Approximate.
        """
        self.matcher.config.binSize = 64
        self.matcher.config.order = 8
        self.matcher.config.undersampleStyle =  "THROW_EXCEPTION"
        self.assertRaises(ValueError,self.matcher.matchBackgrounds,self.chipGap, self.lowCover)


    def testLowCoverIncreaseSampleApproximate(self):
        """Test low coverage with .config.undersampleStyle = INCREASE_NXNYSAMPLE.
        Confirm successful matching with .Approximate.
        """
        self.matcher.config.binSize = 128
        self.matcher.config.order = 4        
        self.matcher.config.undersampleStyle = "INCREASE_NXNYSAMPLE"
        self.checkAccuracy(self.chipGap, self.lowCover)

    def testLowCoverReduceInterpOrderApproximate(self):
        """Test low coverage with .config.undersampleStyle = REDUCE_INTERP_ORDER.
        Confirm successful matching with .Approximate.
        """
        self.matcher.config.binSize = 64
        self.matcher.config.order = 8
        self.matcher.config.undersampleStyle =  "REDUCE_INTERP_ORDER"
        self.checkAccuracy(self.chipGap, self.lowCover)
        
    def testMasksApproximate(self):
        """Test that masks are ignored in matching backgrounds: .Approximate."""
        testExp = afwImage.ExposureF(self.chipGap, True)
        im   = testExp.getMaskedImage().getImage()
        im += 10
        mask = testExp.getMaskedImage().getMask()
        satbit = mask.getPlaneBitMask('SAT')
        for i in range(0,200,20):
            mask.set(5,i,satbit)
            im.set(5,i, 65000)
        self.matcher.config.binSize = 128
        self.matcher.config.order = 4
        self.checkAccuracy(self.chipGap, testExp)

    def testWeightsByInvError(self):
        """Test that bins with high std.dev. and low count are weighted less in fit."""
        testExp = afwImage.ExposureF(self.chipGap, True)
        testIm = testExp.getMaskedImage().getImage()
        self.matcher.config.binSize = 60
        self.matcher.config.order = 4
        for x in  range(0,50):
            for y  in range(0,50):
                if numpy.random.rand(1)[0] < 0.6:
                    testIm.set(x,y,numpy.nan)
                else:
                    testIm.set(x, y, numpy.random.rand(1)[0]*1000)
                    
        struct = self.matcher.matchBackgrounds(self.vanilla, testExp)
        resultExp = testExp
        resultArr = resultExp.getMaskedImage().getImage().getArray()[60:,60:]
        resultMean = numpy.mean(resultArr[numpy.where(~numpy.isnan(resultArr))])
        resultVar = numpy.std(resultArr[numpy.where(~numpy.isnan(resultArr))])**2
        self.assertAlmostEqual(50, resultMean, delta = resultVar) #very loose test.        
                   
    def testSameImageApproximate(self):
        """Test able to match identical images:  .Approximate."""
        vanillaTwin = afwImage.ExposureF(self.vanilla, True)
        self.matcher.config.binSize = 128
        self.matcher.config.order = 4                
        self.checkAccuracy(self.vanilla, vanillaTwin)


    #-=-=-=-=-=-=-=-=-=Background Interp (Splines) -=-=-=-=-=-=-=-=-
    def testVanillaBackground(self):
        """Test basic matching scenario with .Background."""
        self.matcher.config.usePolynomial = False
        self.matcher.config.binSize = 128
        self.checkAccuracy(self.chipGap, self.vanilla)

    def testRampBackground(self):
        """Test basic matching of a linear gradient with .Background."""
        self.matcher.config.usePolynomial = False
        self.matcher.config.binSize = 64
        testExp = afwImage.ExposureF(self.vanilla, True)
        testIm = testExp.getMaskedImage().getImage()
        afwMath.randomGaussianImage(testIm,afwMath.Random(5))
        nx, ny = testExp.getDimensions()
        dzdx, dzdy, z0 = 1, 2, 0.0
        for x in range(nx):
            for y in range(ny):
                z = testIm.get(x, y)
                testIm.set(x, y, z + dzdx * x + dzdy * y + z0)
        self.checkAccuracy(testExp, self.vanilla)
        
    def testUndersampleBackgroundPasses(self):
        """Test undersample style (REDUCE_INTERP_ORDER): .Background.
        
        INCREASE_NXNYSAMPLE no longer supported by .Background because .Backgrounds's are
        defined by their nx and ny grid. 
        """
        self.matcher.config.usePolynomial = False
        self.matcher.config.binSize = 256
        self.matcher.config.undersampleStyle = "REDUCE_INTERP_ORDER"
        self.checkAccuracy(self.chipGap, self.vanilla)
        
        self.matcher.config.undersampleStyle = "INCREASE_NXNYSAMPLE"
        self.assertRaises(RuntimeError, self.matcher.matchBackgrounds,self.chipGap, self.vanilla)
        
    def testSameImageBackground(self):
        """Test able to match identical images with .Background."""
        self.matcher.config.usePolynomial = False
        self.matcher.config.binSize = 256
        self.checkAccuracy(self.vanilla, self.vanilla)

    def testMasksBackground(self):
        """Test masks ignored in matching backgrounds with .Background."""
        self.matcher.config.usePolynomial = False
        self.matcher.config.binSize = 256
        testExp = afwImage.ExposureF(self.chipGap, True)
        im   = testExp.getMaskedImage().getImage()
        im += 10
        mask = testExp.getMaskedImage().getMask()
        satbit = mask.getPlaneBitMask('SAT')
        for i in range(0,200,20):
            mask.set(5,i,satbit)
            im.set(5,i, 65000)
        self.checkAccuracy(self.chipGap, testExp)

    def testChipGapHorizontalBackground(self): 
        """ Test able to match image with horizontal chip gap (row of nans) with .Background"""
        self.matcher.config.usePolynomial = False
        self.matcher.config.binSize = 64
        chipGapHorizontal = afwImage.ExposureF(600,600)
        im = chipGapHorizontal.getMaskedImage().getImage()
        afwMath.randomGaussianImage(im,afwMath.Random(8))
        im += 10
        im.getArray()[200:300,:] = numpy.nan #simulate 100pix chip gap horizontal
        chipGapHorizontal.getMaskedImage().getVariance().set(1.0)
        self.checkAccuracy(self.vanilla, chipGapHorizontal)

    def testChipGapVerticalBackground(self):
        """ Test able to match images with vertical chip gaps (column of nans) wider than bin size"""
        self.matcher.config.usePolynomial = False
        self.matcher.config.binSize = 64
        self.checkAccuracy(self.chipGap, self.vanilla)

    def testLowCoverBackground(self):
        """ Test able to match images that do not cover the whole patch"""
        self.matcher.config.usePolynomial = False
        self.matcher.config.binSize = 64
        self.checkAccuracy(self.vanilla, self.lowCover)


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
