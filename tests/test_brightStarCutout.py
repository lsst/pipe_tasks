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
        self.x_gradient = 5.432
        self.y_gradient = 10.987
        self.curvature_x = 0.1
        self.curvature_y = -0.2
        self.curvature_xy = 1e-2

        # Create a pedestal + 2D plane
        x_coords = np.linspace(-50, 50, 101)
        y_coords = np.linspace(-50, 50, 101)
        x_plane, y_plane = np.meshgrid(x_coords, y_coords)
        pedestal = np.ones_like(x_plane) * self.pedestal

        # Create a pseudo-PSF
        dist_from_center = np.sqrt(x_plane ** 2 + y_plane ** 2)
        psf_array = np.exp(-dist_from_center / 5)
        psf_array /= np.sum(psf_array)
        fixed_kernel = FixedKernel(ImageD(psf_array))
        psf = KernelPsf(fixed_kernel)
        self.psf = psf.computeKernelImage(psf.getAveragePosition())

        # Bring everything together to construct a stamp masked image
        stamp_array = (
            psf_array * self.scale + pedestal + x_plane * self.x_gradient + y_plane * self.y_gradient
        )
        stamp_array += (
            x_plane**2 * self.curvature_x
            + y_plane**2 * self.curvature_y
            + x_plane * y_plane * self.curvature_xy
        )
        stampIm = ImageF((stamp_array).astype(np.float32))
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
        fit_psf_results = brightStarCutoutTask._fit_psf(
            self.stampMI,
            self.psf,
        )

        assert abs(fit_psf_results["scale"] - self.scale) / self.scale < 1e-3
        assert abs(fit_psf_results["pedestal"] - self.pedestal) / self.pedestal < 1e-3
        assert abs(fit_psf_results["x_gradient"] - self.x_gradient) / self.x_gradient < 1e-3
        assert abs(fit_psf_results["y_gradient"] - self.y_gradient) / self.y_gradient < 1e-3
        assert abs(fit_psf_results["curvature_x"] - self.curvature_x) / self.curvature_x < 1e-3
        assert abs(fit_psf_results["curvature_y"] - self.curvature_y) / self.curvature_y < 1e-3
        assert abs(fit_psf_results["curvature_xy"] - self.curvature_xy) / self.curvature_xy < 1e-3


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
