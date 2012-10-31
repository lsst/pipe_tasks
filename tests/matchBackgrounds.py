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
        #1) full coverage (plain vanilla image) w/ mean = 50counts
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
        self.lowCover.getMaskedImage().getImage().getArray()[:,200:] = numpy.nan #simulate low image coverage:
        self.lowCover.getMaskedImage().getVariance().set(1.0)

        #make a matchBackgrounds object
        self.matcher = pipeTasks.matchBackgrounds.MatchBackgroundsTask()
        self.matcher.config.usePolynomial = True
        self.matcher.binSize = 64
        self.matcher.debugVisitString = 'Test Visit'

        self.sctrl = afwMath.StatisticsControl()
        self.sctrl.setNanSafe(True)
        self.sctrl.setAndMask(afwImage.MaskU.getPlaneBitMask(["EDGE", "DETECTED", "DETECTED_NEGATIVE","SAT","BAD","INTRP","CR"]))
        
    def tearDown(self):
        self.vanilla = None
        self.lowCover = None
        self.chipGap = None
        self.matcher = None
        self.sctrl = None
        
    def checkAccuracy(self, refExp, sciExp):
        """Checks for accurate background matching in the matchBackgrounds Method
           to be called by tests expecting successful matching
        """
        struct = self.matcher.matchBackgrounds(refExp, sciExp)
        resultExp = struct.matchedExposure
        MSE = struct.matchedMSE
        diffImVar = struct.diffImVar
        RMS = struct.fitRMS
        
        resultStats = afwMath.makeStatistics(resultExp.getMaskedImage(), afwMath.MEAN | afwMath.VARIANCE,self.sctrl)
        resultMean, _ = resultStats.getResult(afwMath.MEAN)
        resultVar, _ = resultStats.getResult(afwMath.VARIANCE)
        refStats = afwMath.makeStatistics(refExp.getMaskedImage(), afwMath.MEAN | afwMath.VARIANCE,self.sctrl)
        refMean, _ = refStats.getResult(afwMath.MEAN)
        #print "refMean %.03f, resultMean %.03f, resultVar %.03f"%(refMean, resultMean, resultVar)
        self.assertAlmostEqual(refMean, resultMean, delta = resultVar) #very loose test. 
        #print "MSE %.05f, diffImVar %.05f"%(MSE, diffImVar)
        
        #MSE within 2% of the variance of the difference image = SUCCESS
        self.assertLess(MSE, diffImVar * 1.02)
    
                      
    #-=-=-=-=-=-=Test Polynomial Fit (Approximate class)-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def testConfig(self):
        """ 
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
        """Tests input dimensions. Throws RuntimeError if dimensions don't match"""
        self.matcher.config.binSize = 128
        self.matcher.config.order = 2
        #make image with wronge size
        wrongSize = afwImage.ExposureF(500,500)
        wrongSize.getMaskedImage().getImage().set(1.0)
        wrongSize.getMaskedImage().getVariance().set(1.0)
        self.assertRaises(RuntimeError,self.matcher.matchBackgrounds,self.chipGap, wrongSize)

    def testVanillaApproximate(self):
        """Tests basic matching"""
        self.matcher.config.binSize = 128
        self.matcher.config.order = 4
        self.checkAccuracy(self.chipGap, self.vanilla)

    def testRampApproximate(self):
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
        """low coverage with .config.undersampleStyle = THROW_EXCEPTION should throw ValueError"""
        self.matcher.config.binSize = 64
        self.matcher.config.order = 8
        self.matcher.config.undersampleStyle =  "THROW_EXCEPTION"
        self.assertRaises(ValueError,self.matcher.matchBackgrounds,self.chipGap, self.lowCover)


    def testLowCoverIncreaseSampleApproximate(self):
        """low coverage with .config.undersampleStyle = INCREASE_NXNYSAMPLE should succeed"""
        self.matcher.config.binSize = 128
        self.matcher.config.order = 4        
        self.matcher.config.undersampleStyle = "INCREASE_NXNYSAMPLE"
        self.checkAccuracy(self.chipGap, self.lowCover)

    def testLowCoverReduceInterpOrderApproximate(self):
        """low coverage with .config.undersampleStyle = REDUCE_INTERP_ORDER should succeed"""
        self.matcher.config.binSize = 64
        self.matcher.config.order = 8        
        self.matcher.config.undersampleStyle =  "REDUCE_INTERP_ORDER"
        self.checkAccuracy(self.chipGap, self.lowCover)

    def testMasksApproximate(self):
        """Masks should be ignored in matching backgrounds"""
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

    def testSameImageApproximate(self):
        """What should it do if the two images are the same? Succeeds"""
        vanillaTwin = afwImage.ExposureF(self.vanilla, True)
        self.matcher.config.binSize = 128
        self.matcher.config.order = 4                
        self.checkAccuracy(self.vanilla, vanillaTwin)


    #-=-=-=-=-=-=-=-=-=Background Interp (Splines) -=-=-=-=-=-=-=-=-
    def testVanillaBackground(self): 
        self.matcher.config.usePolynomial = False
        self.matcher.config.binSize = 128
        self.checkAccuracy(self.chipGap, self.vanilla)

    def testRampBackground(self):
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
        self.matcher.config.usePolynomial = False
        self.matcher.config.binSize = 256
        self.matcher.config.undersampleStyle = "REDUCE_INTERP_ORDER"
        self.checkAccuracy(self.chipGap, self.vanilla)
     # "INCREASE_NXNYSAMPLE" DEPRECIATED
     #   self.matcher.config.undersampleStyle = "INCREASE_NXNYSAMPLE"
     #   self.checkAccuracy(self.chipGap, self.vanilla)
        
    def testSameImageBackground(self):
        """What should it do if the two images are the same? Succeeds"""
        self.matcher.config.usePolynomial = False
        self.matcher.config.binSize = 256
        self.checkAccuracy(self.vanilla, self.vanilla)

    def testMasksBackground(self):
        """Masks should be ignored in matching backgrounds"""
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

    #Demonstrate Known Failures
    def testLowCoverBackground(self): #Known Failure
        """ Background class throws RuntimeError for images that do not cover the whole patch"""
        self.matcher.config.usePolynomial = False
        self.matcher.config.binSize = 64
        self.assertRaises(RuntimeError, self.matcher.matchBackgrounds, self.vanilla, self.lowCover)
        
    def testChipGapBackground(self): #Known Failure
        """ Background class throws RuntimeError for chip gaps wider than bin size"""
        self.matcher.config.usePolynomial = False
        self.matcher.config.binSize = 64
        self.assertRaises(RuntimeError, self.matcher.matchBackgrounds,self.chipGap, self.vanilla)

        

   # "INCREASE_NXNYSAMPLE" DEPRECIATETED
   #       
   # def testUndersampleBackgroundFails(self):  #Known Failure
   #     """ Background class throws RuntimeError for chip gaps wider than bin size"""
   #     self.matcher.config.usePolynomial = False
   #     self.matcher.config.binSize = 256
   #     self.matcher.config.undersampleStyle = "INCREASE_NXNYSAMPLE"
   #     #widen chip gap
   #     testExp = afwImage.ExposureF(self.chipGap, True)
   #     testExp.getMaskedImage().getImage().getArray()[:,200:400] = numpy.nan 
   #     self.assertRaises(RuntimeError, self.matcher.matchBackgrounds,testExp, self.vanilla)
   #     self.matcher.config.undersampleStyle = "REDUCE_INTERP_ORDER"
   #     self.assertRaises(RuntimeError, self.matcher.matchBackgrounds,testExp, self.vanilla)
   #     #Message: Failed to initialise spline for type cspline, length 0

        
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
