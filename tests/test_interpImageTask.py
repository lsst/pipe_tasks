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
from lsst.pipe.tasks.interpImage import CloughTocher2DInterpolateTask, InterpImageTask

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


class CloughTocher2DInterpolateTestCase(lsst.utils.tests.TestCase):
    """Test the CloughTocher2DInterpolateTask."""

    def setUp(self):
        super().setUp()

        self.maskedimage = afwImage.MaskedImageF(100, 121)
        for x in range(100):
            for y in range(121):
                self.maskedimage[x, y] = (3 * y + x * 5, 0, 1.0)

        # Clone the maskedimage so we can compare it after running the task.
        self.reference = self.maskedimage.clone()

        # Set some central pixels as SAT
        sliceX, sliceY = slice(30, 35), slice(40, 45)
        self.maskedimage.mask[sliceX, sliceY] = afwImage.Mask.getPlaneBitMask("SAT")
        self.maskedimage.image[sliceX, sliceY] = np.nan
        # Put nans here to make sure interp is done ok

        # Set an entire column as BAD
        self.maskedimage.mask[54:55, :] = afwImage.Mask.getPlaneBitMask("BAD")
        self.maskedimage.image[54:55, :] = np.nan

        # Set an entire row as BAD
        self.maskedimage.mask[:, 110:111] = afwImage.Mask.getPlaneBitMask("BAD")
        self.maskedimage.image[:, 110:111] = np.nan

        # Set a diagonal set of pixels as CR
        for i in range(74, 78):
            self.maskedimage.mask[i, i] = afwImage.Mask.getPlaneBitMask("CR")
            self.maskedimage.image[i, i] = np.nan

        # Set one of the edges as EDGE
        self.maskedimage.mask[0:1, :] = afwImage.Mask.getPlaneBitMask("EDGE")
        self.maskedimage.image[0:1, :] = np.nan

        # Set a smaller streak at the edge
        self.maskedimage.mask[25:28, 0:1] = afwImage.Mask.getPlaneBitMask("EDGE")
        self.maskedimage.image[25:28, 0:1] = np.nan

        # Update the reference image's mask alone, so we can compare them after
        # running the task.
        self.reference.mask.array[:, :] = self.maskedimage.mask.array

        # Create a noise image
        self.noise = self.maskedimage.clone()
        np.random.seed(12345)
        self.noise.image.array[:, :] = np.random.normal(size=self.noise.image.array.shape)

    @lsst.utils.tests.methodParameters(n_runs=(1, 2))
    def test_interpolation(self, n_runs: int):
        """Test that the interpolation is done correctly.

        Parameters
        ----------
        n_runs : `int`
            Number of times to run the task. Running the task more than once
            should have no effect.
        """
        config = CloughTocher2DInterpolateTask.ConfigClass()
        config.badMaskPlanes = (
            "BAD",
            "SAT",
            "CR",
            "EDGE",
        )
        config.fillValue = 0.5
        task = CloughTocher2DInterpolateTask(config)
        for n in range(n_runs):
            task.run(self.maskedimage)

        # Assert that the mask and the variance planes remain unchanged.
        self.assertImagesEqual(self.maskedimage.variance, self.reference.variance)
        self.assertMasksEqual(self.maskedimage.mask, self.reference.mask)

        # Check that the long streak of bad pixels have been replaced with the
        # fillValue, but not the short streak.
        np.testing.assert_array_equal(self.maskedimage.image[0:1, :].array, config.fillValue)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(self.maskedimage.image[25:28, 0:1].array, config.fillValue)

        # Check that interpolated pixels are close to the reference (original),
        # and that none of them is still NaN.
        self.assertTrue(np.isfinite(self.maskedimage.image.array).all())
        self.assertImagesAlmostEqual(
            self.maskedimage.image[1:, :], self.reference.image[1:, :], rtol=1e-05, atol=1e-08
        )

    @lsst.utils.tests.methodParametersProduct(pass_badpix=(True, False), pass_goodpix=(True, False))
    def test_interpolation_with_noise(self, pass_badpix: bool = True, pass_goodpix: bool = True):
        """Test that we can reuse the badpix and goodpix.

        Parameters
        ----------
        pass_badpix : `bool`
            Whether to pass the badpix to the task?
        pass_goodpix : `bool`
            Whether to pass the goodpix to the task?
        """

        config = CloughTocher2DInterpolateTask.ConfigClass()
        config.badMaskPlanes = (
            "BAD",
            "SAT",
            "CR",
            "EDGE",
        )
        task = CloughTocher2DInterpolateTask(config)

        badpix, goodpix = task.run(self.noise)
        task.run(
            self.maskedimage,
            badpix=(badpix if pass_badpix else None),
            goodpix=(goodpix if pass_goodpix else None),
        )

        # Check that the long streak of bad pixels by the edge have been
        # replaced with fillValue, but not the short streak.
        np.testing.assert_array_equal(self.maskedimage.image[0:1, :].array, config.fillValue)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(self.maskedimage.image[25:28, 0:1].array, config.fillValue)

        # Check that interpolated pixels are close to the reference (original),
        # and that none of them is still NaN.
        self.assertTrue(np.isfinite(self.maskedimage.image.array).all())
        self.assertImagesAlmostEqual(
            self.maskedimage.image[1:, :], self.reference.image[1:, :], rtol=1e-05, atol=1e-08
        )


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
