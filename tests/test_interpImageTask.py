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

"""
Tests for bad pixel interpolation task

Run with:
   python test_interpImageTask.py
or
   pytest test_interpImageTask.py
"""
import unittest

import numpy as np

import lsst.utils.tests
import lsst.geom
import lsst.afw.image as afwImage
import lsst.pex.config as pexConfig
import lsst.ip.isr as ipIsr
from lsst.pipe.tasks.interpImage import InterpImageTask

try:
    display
except NameError:
    display = False
else:
    import lsst.afw.display as afwDisplay
    afwDisplay.setDefaultMaskTransparency(75)


class InterpolationTestCase(lsst.utils.tests.TestCase):
    """A test case for interpolation.
    """

    def setUp(self):
        self.FWHM = 5

    def testEdge(self):
        """Test that we can interpolate to the edge.
        """
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

        mi[0:nBadCol, :, afwImage.LOCAL] = (10, badBit, 0)  # Bad left edge
        # With another bad set of columns next to bad left edge
        mi[-nBadCol:, :, afwImage.LOCAL] = (10, badBit, 0)
        mi[nBadCol + 1:nBadCol + 4, 0:10, afwImage.LOCAL] = (100, badBit, 0)  # Bad right edge
        # more bad of columns next to bad right edge
        mi[-nBadCol - 4:-nBadCol - 1, 0:10, afwImage.LOCAL] = (100, badBit, 0)

        defectList = ipIsr.Defects.fromMask(mi, pixelPlane)

        if display:
            afwDisplay.Display(frame=0).mtv(mi, title=self._testMethodName + ": image")

        def validateInterp(miInterp, useFallbackValueAtEdge, fallbackValue):
            imaInterp = miInterp.getImage().getArray()
            if display:
                afwDisplay.Display(frame=1).mtv(miInterp, title=self._testMethodName + ": interp image")
            self.assertGreater(np.min(imaInterp), min(-2, 2*fallbackValue))
            self.assertGreater(max(2, 2*fallbackValue), np.max(imaInterp))
            val0 = np.mean(miInterp.image[1:2, :, afwImage.LOCAL].array, dtype=float)
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
                            continue

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

    def testTranspose(self):
        """Test transposition before interpolation

        Interpolate over a bad row (not a bad column).
        """
        box = lsst.geom.Box2I(lsst.geom.Point2I(12345, 6789), lsst.geom.Extent2I(123, 45))
        value = 123.45
        bad = "BAD"
        image = afwImage.MaskedImageF(box)
        image.image.set(value)
        image.mask.set(0)

        badRow = box.getHeight()//2
        image.image.array[badRow] = 10*value
        image.mask.array[badRow] = image.mask.getPlaneBitMask(bad)

        config = InterpImageTask.ConfigClass()
        config.transpose = True
        task = InterpImageTask(config)
        task.run(image, planeName=bad, fwhmPixels=self.FWHM)
        self.assertFloatsEqual(image.image.array, value)


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
