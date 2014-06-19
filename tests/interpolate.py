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

"""
Tests for bad pixel interpolation task

Run with:
   python interpolate.py
or
   python
   >>> import interpolate; interpolate.run()
"""

import os
import unittest
import numpy as np
import lsst.utils.tests as tests
import lsst.afw.image as afwImage
import lsst.afw.display.ds9 as ds9
import lsst.meas.algorithms as measAlg
from lsst.pipe.tasks.interpImage import InterpImageTask

try:
    type(display)
except NameError:
    display = False

class interpolationTestCase(unittest.TestCase):
    """A test case for interpolation"""
    def setUp(self):
        self.FWHM = 5
        self.psf = measAlg.DoubleGaussianPsf(15, 15, self.FWHM/(2*np.sqrt(2*np.log(2))))

    def tearDown(self):
        del self.psf

    def testEdge(self):
        """Test that we can interpolate to the edge"""

        for useFallbackValueAtEdge in (True, False):
            config = InterpImageTask.ConfigClass()
            config.useFallbackValueAtEdge = useFallbackValueAtEdge
            interpTask = InterpImageTask(config)

            mi = afwImage.MaskedImageF(80, 30)
            ima = mi.getImage().getArray()
            #
            # We'll set the BAD bit in pixels we wish to interpolate over
            #
            pixelPlane = "BAD"
            badBit = afwImage.MaskU.getPlaneBitMask(pixelPlane)
            #
            # Set bad columns near left and right sides of image
            #
            nBadCol = 10
            mi.set((0, 0x0, 0))

            np.random.seed(666)
            ima[:] = np.random.uniform(-1, 1, ima.shape)
            #
            mi[0:nBadCol, :] = (10, badBit, 0) # Bad left edge
            mi[-nBadCol:, :] = (10, badBit, 0) # With another bad set of columns next to bad left edge
            mi[nBadCol+1:nBadCol+4, 0:10] = (100, badBit, 0) # Bad right edge
            mi[-nBadCol-4:-nBadCol-1, 0:10] = (100, badBit, 0) # more bad of columns next to bad right edge

            if display:
                ds9.mtv(mi, frame=0)
            #
            # Time to interpolate
            #
            interpTask.run(mi, pixelPlane, self.psf, fallbackValue=0)

            if display:
                ds9.mtv(mi, frame=1)

            self.assertGreater(np.min(ima), -2)
            self.assertGreater(2, np.max(ima))

            val0 = np.mean(mi.getImage()[1, :].getArray())
            if useFallbackValueAtEdge:
                self.assertEqual(val0, 0)
            else:
                self.assertNotEqual(val0, 0)
                
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def suite():
    """Returns a suite containing all the test cases in this module."""
    tests.init()

    suites = []
    suites += unittest.makeSuite(interpolationTestCase)
    suites += unittest.makeSuite(tests.MemoryTestCase)
    return unittest.TestSuite(suites)

def run(exit = False):
    """Run the tests"""
    tests.run(suite(), exit)

if __name__ == "__main__":
    run(True)
