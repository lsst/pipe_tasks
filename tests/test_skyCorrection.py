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

import lsst.utils.tests
import numpy as np
from lsst.afw.image import ExposureF
from lsst.afw.math import BackgroundMI
from lsst.obs.subaru import HyperSuprimeCam
from lsst.pipe.tasks.skyCorrection import SkyCorrectionConfig, SkyCorrectionTask


class SkyCorrectionTestCase(lsst.utils.tests.TestCase):

    def setUp(self):
        hsc = HyperSuprimeCam()
        self.camera = hsc.getCamera()
        bbox = self.camera[0].getBBox()

        self.skyCorrectionConfig = SkyCorrectionConfig()
        self.skyCorrectionConfig.bgModel1.xSize = 8192 * 0.015
        self.skyCorrectionConfig.bgModel1.ySize = 8192 * 0.015
        self.skyCorrectionConfig.bgModel1.pixelSize = 0.015
        self.skyCorrectionConfig.bgModel2.xSize = 256 * 0.015
        self.skyCorrectionConfig.bgModel2.ySize = 256 * 0.015
        self.skyCorrectionConfig.bgModel2.pixelSize = 0.015

        def _createGaussian(flux, fwhm, size):
            """Create a 2D Gaussian image.

            Parameters
            ----------
            flux : `float`
                Total flux of the Gaussian.
            fwhm : `float`
                Full width at half maximum of the Gaussian.
            size : `int`
                Size of the square image in pixels.
            """
            sigma = fwhm / 2.355
            x0, y0 = size // 2, size // 2  # Center of the Gaussian

            # Create a 2D grid of (x, y) coordinates
            y, x = np.mgrid[0:size, 0:size]
            gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))

            # Normalize to get the total flux of 100
            gaussian *= flux / gaussian.sum()
            return gaussian

        # Generate calexp/calexpBackground/sky for the central 6 HSC detectors
        self.calExps = []
        self.calBkgs = []
        self.skyFrames = []
        for detector in [41, 42, 49, 50, 57, 58]:
            rng = np.random.default_rng(detector)

            # Science image
            calexp = ExposureF(bbox)
            calexp.setDetector(self.camera[detector])
            # Add a sky frame signature to a subregion of the image
            calexp.image.array[:, (5 * 32) : (10 * 32)] += 5
            calexp.image.array[(15 * 32) : (20 * 32)] += 10
            # Add random noise
            calexp.image.array += 100 * rng.standard_normal((bbox.getDimensions().y, bbox.getDimensions().x))
            # Add some fake Gaussian sources to the image
            gaussian_size = 100
            for i in range(10):
                x = round(rng.uniform(gaussian_size // 2, bbox.getDimensions().x - gaussian_size // 2))
                y = round(rng.uniform(gaussian_size // 2, bbox.getDimensions().y - gaussian_size // 2))
                calexp.image.array[
                    y - gaussian_size // 2 : y + gaussian_size // 2,
                    x - gaussian_size // 2 : x + gaussian_size // 2,
                ] += _createGaussian(1e6, 10, gaussian_size)
            self.calExps.append(calexp)

            # Background image
            backgroundImage = ExposureF(bbox)
            backgroundImage.setDetector(self.camera[detector])
            backgroundImage.image.array += 3000
            background = BackgroundMI(bbox, backgroundImage.getMaskedImage())
            calexpBackground = lsst.afw.math.BackgroundList(
                (
                    background,
                    lsst.afw.math.Interpolate.AKIMA_SPLINE,
                    lsst.afw.math.UndersampleStyle.REDUCE_INTERP_ORDER,
                    lsst.afw.math.ApproximateControl.CHEBYSHEV,
                    6,
                    6,
                    1,
                )
            )
            self.calBkgs.append(calexpBackground)

            # Sky frame
            sky = ExposureF(64, 131)
            sky.setDetector(self.camera[detector])
            header = sky.getMetadata()
            header.set("BOX.MINX", bbox.getMinX())
            header.set("BOX.MINY", bbox.getMinY())
            header.set("BOX.MAXX", bbox.getMaxX())
            header.set("BOX.MAXY", bbox.getMaxY())
            header.set("ALGORITHM", "NATURAL_SPLINE")
            sky.image.array[:, 5:10] += 5
            sky.image.array[15:20, :] += 10
            sky.maskedImage.set(0.0, 0x0, 1.0)
            self.skyFrames.append(sky)

    def tearDown(self):
        del self.camera
        del self.calExps
        del self.calBkgs
        del self.skyFrames

    def testSkyCorrection(self):
        """TEMP."""

        # breakpoint()
        skyCorrectionTask = SkyCorrectionTask(config=self.skyCorrectionConfig)
        results = skyCorrectionTask.run(self.calExps, self.calBkgs, self.skyFrames, self.camera)
        breakpoint()


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
