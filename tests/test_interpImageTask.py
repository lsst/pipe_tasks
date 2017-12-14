#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008-2015 AURA/LSST.
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
#

"""
Tests for bad pixel interpolation task

Run with:
   python testInterpImageTask.py
or
   python
   >>> import testInterpImageTask; testInterpImageTask.run()
"""
from __future__ import division, print_function, absolute_import

import unittest

import numpy as np

import lsst.utils.tests
import lsst.afw.image as afwImage
import lsst.afw.display.ds9 as ds9
import lsst.ip.isr as ipIsr
import lsst.pex.config as pexConfig
from lsst.pipe.tasks.interpImage import InterpImageTask

try:
    type(display)
except NameError:
    display = False


class interpolationTestCase(unittest.TestCase):
    """A test case for interpolation"""

    def setUp(self):
        self.FWHM = 5

    def testEdge(self):
        """Test that we can interpolate to the edge"""

        mi = afwImage.MaskedImageF(80, 30)
        ima = mi.getImage().getArray()
        #
        # We'll set the BAD bit in pixels we wish to interpolate over
        #
        pixelPlane = "BAD"
        badBit = afwImage.Mask.getPlaneBitMask(pixelPlane)
        #
        # Set bad columns near left and right sides of image
        #
        nBadCol = 10
        mi.set((0, 0x0, 0))

        np.random.seed(666)
        ima[:] = np.random.uniform(-1, 1, ima.shape)

        mi[0:nBadCol, :] = (10, badBit, 0)  # Bad left edge
        mi[-nBadCol:, :] = (10, badBit, 0)  # With another bad set of columns next to bad left edge
        mi[nBadCol+1:nBadCol+4, 0:10] = (100, badBit, 0)  # Bad right edge
        mi[-nBadCol-4:-nBadCol-1, 0:10] = (100, badBit, 0)  # more bad of columns next to bad right edge

        defectList = ipIsr.getDefectListFromMask(mi, pixelPlane)

        if display:
            ds9.mtv(mi, frame=0)

        def validateInterp(miInterp, useFallbackValueAtEdge, fallbackValue):
            imaInterp = miInterp.getImage().getArray()
            if display:
                ds9.mtv(miInterp, frame=1)
            self.assertGreater(np.min(imaInterp), min(-2, 2*fallbackValue))
            self.assertGreater(max(2, 2*fallbackValue), np.max(imaInterp))
            val0 = np.mean(miInterp.getImage()[1, :].getArray(), dtype=float)
            if useFallbackValueAtEdge:
                self.assertAlmostEqual(val0, fallbackValue, 6)
            else:
                self.assertNotEqual(val0, 0)

        for useFallbackValueAtEdge in (False, True):
            miInterp = mi.clone()
            config = InterpImageTask.ConfigClass()
            config.useFallbackValueAtEdge = useFallbackValueAtEdge
            interpTask = InterpImageTask(config)

            if useFallbackValueAtEdge:
                config.fallbackUserValue = -1.0
                # choiceField fallbackValueType cannot be None if useFallbackValueAtEdge is True
                config.fallbackValueType = None
                self.assertRaises(NotImplementedError, interpTask._setFallbackValue, miInterp)
                # make sure an invalid fallbackValueType raises a pexConfig.FieldValidationError
                with self.assertRaises(pexConfig.FieldValidationError):
                    config.fallbackValueType = "NOTUSED"
                # make sure ValueError is raised if both a planeName and defects list are provided
                self.assertRaises(ValueError, interpTask.run, miInterp, defects=defectList,
                                  planeName=pixelPlane, fwhmPixels=self.FWHM)

                for fallbackValueType in ("USER", "MEAN", "MEDIAN", "MEANCLIP"):
                    for negativeFallbackAllowed in (True, False):
                        config.negativeFallbackAllowed = negativeFallbackAllowed
                        config.fallbackValueType = fallbackValueType
                        # Should raise if negative not allowed, but USER supplied negative value
                        if not negativeFallbackAllowed and fallbackValueType == "USER":
                            self.assertRaises(ValueError, config.validate)

                        interpTask = InterpImageTask(config)
                        fallbackValue = interpTask._setFallbackValue(mi)
                        #
                        # Time to interpolate
                        #
                        miInterp = mi.clone()
                        interpTask.run(miInterp, planeName=pixelPlane, fwhmPixels=self.FWHM)
                        validateInterp(miInterp, useFallbackValueAtEdge, fallbackValue)
                        miInterp = mi.clone()
                        interpTask.run(miInterp, defects=defectList)
                        validateInterp(miInterp, useFallbackValueAtEdge, fallbackValue)
            else:
                #
                # Time to interpolate
                #
                miInterp = mi.clone()
                interpTask.run(miInterp, planeName=pixelPlane, fwhmPixels=self.FWHM)
                validateInterp(miInterp, useFallbackValueAtEdge, 0)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
