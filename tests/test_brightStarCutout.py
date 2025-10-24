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

import unittest

import lsst.afw.cameraGeom.testUtils
import lsst.afw.image
import lsst.utils.tests
import numpy as np
from lsst.afw.image import ImageD, ImageF, MaskedImageF
from lsst.afw.math import FixedKernel
from lsst.geom import Point2I
from lsst.meas.algorithms import KernelPsf
from lsst.pipe.tasks.brightStarSubtraction import BrightStarCutoutConfig, BrightStarCutoutTask


class BrightStarCutoutTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        # Fit values
        self.scale = 2.34e5
        self.pedestal = 3210.1
        self.xGradient = 5.432
        self.yGradient = 10.987

        # Create a pedestal + 2D plane
        xCoords = np.linspace(-50, 50, 101)
        yCoords = np.linspace(-50, 50, 101)
        xPlane, yPlane = np.meshgrid(xCoords, yCoords)
        pedestal = np.ones_like(xPlane) * self.pedestal

        # Create a pseudo-PSF
        dist_from_center = np.sqrt(xPlane**2 + yPlane**2)
        psfArray = np.exp(-dist_from_center / 5)
        psfArray /= np.sum(psfArray)
        fixedKernel = FixedKernel(ImageD(psfArray))
        psf = KernelPsf(fixedKernel)
        self.psf = psf.computeKernelImage(psf.getAveragePosition())

        # Bring everything together to construct a stamp masked image
        stampArray = psfArray * self.scale + pedestal + xPlane * self.xGradient + yPlane * self.yGradient
        stampIm = ImageF((stampArray).astype(np.float32))
        stampVa = ImageF(stampIm.getBBox(), 654.321)
        self.stampMI = MaskedImageF(image=stampIm, variance=stampVa)
        self.stampMI.setXY0(Point2I(-50, -50))

        # Ensure that all mask planes required by the task are in-place;
        # new mask plane entries will be created as necessary
        badMaskPlanes = [
            "BAD",
            "CR",
            "CROSSTALK",
            "EDGE",
            "NO_DATA",
            "SAT",
            "SUSPECT",
            "UNMASKEDNAN",
            "NEIGHBOR",
        ]
        _ = [self.stampMI.mask.addMaskPlane(mask) for mask in badMaskPlanes]

    def test_fitPsf(self):
        """Test the PSF fitting method."""
        brightStarCutoutConfig = BrightStarCutoutConfig()
        brightStarCutoutTask = BrightStarCutoutTask(config=brightStarCutoutConfig)
        fitPsfResults = brightStarCutoutTask._fitPsf(
            self.stampMI,
            self.psf,
        )
        assert abs(fitPsfResults["scale"] - self.scale) / self.scale < 1e-6
        assert abs(fitPsfResults["pedestal"] - self.pedestal) / self.pedestal < 1e-6
        assert abs(fitPsfResults["xGradient"] - self.xGradient) / self.xGradient < 1e-6
        assert abs(fitPsfResults["yGradient"] - self.yGradient) / self.yGradient < 1e-6


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
