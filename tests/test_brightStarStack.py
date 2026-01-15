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

import lsst.afw.image
import lsst.utils.tests
import numpy as np
from lsst.afw.image import ImageF, MaskedImageF
from lsst.pipe.tasks.brightStarSubtraction import BrightStarStackConfig, BrightStarStackTask


# Mock class to simulate the DeferredDatasetHandle (DDH) behavior
class MockHandle:
    def __init__(self, content):
        self.content = content

    def get(self):
        return self.content


# Mock class to simulate a single BrightStarStamp
class MockStamp:
    def __init__(self, stamp_im, mag, fit_params, stats):
        self.stamp_im = stamp_im
        self.ref_mag = mag

        # Unpack fit parameters
        self.scale = fit_params.get("scale", 1.0)
        self.pedestal = fit_params.get("pedestal", 0.0)
        self.gradient_x = fit_params.get("gradient_x", 0.0)
        self.gradient_y = fit_params.get("gradient_y", 0.0)
        self.curvature_x = fit_params.get("curvature_x", 0.0)
        self.curvature_y = fit_params.get("curvature_y", 0.0)
        self.curvature_xy = fit_params.get("curvature_xy", 0.0)

        # Unpack statistics for filtering
        self.global_reduced_chi_squared = stats.get("global_chi2", 1.0)
        self.psf_reduced_chi_squared = stats.get("psf_chi2", 1.0)
        self.bright_global_reduced_chi_squared = stats.get("bright_global_chi2", 1.0)
        self.psf_bright_reduced_chi_squared = stats.get("psf_bright_chi2", 1.0)
        self.bright_star_threshold = stats.get("brights_threshold", 100.0)
        self.focal_plane_radius = stats.get("fp_radius", 100.0)


class BrightStarStackTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        # Define fit values
        self.scale = 10.0
        self.pedestal = 50.0
        self.x_gradient = 0.5
        self.y_gradient = -0.5
        self.curvature_x = 0.01
        self.curvature_y = 0.01
        self.curvature_xy = 0.005

        self.fit_params = {
            "scale": self.scale,
            "pedestal": self.pedestal,
            "gradient_x": self.x_gradient,
            "gradient_y": self.y_gradient,
            "curvature_x": self.curvature_x,
            "curvature_y": self.curvature_y,
            "curvature_xy": self.curvature_xy,
        }

        # Create the "Clean" PSF (a simple Gaussian)
        self.dim = 51
        x_coords = np.linspace(-25, 25, self.dim)
        y_coords = np.linspace(-25, 25, self.dim)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        sigma = 5.0
        dist_sq = x_grid**2 + y_grid**2
        self.clean_array = np.exp(-dist_sq / (2 * sigma**2))

        # Create the "star" Image (What the task receives)
        # Apply scaling
        star_array = self.clean_array * self.scale

        # Add background terms
        x_indices, y_indices = np.meshgrid(np.arange(self.dim), np.arange(self.dim))

        star_array += self.pedestal
        star_array += x_indices * self.x_gradient
        star_array += y_indices * self.y_gradient
        star_array += (x_indices**2) * self.curvature_x
        star_array += (y_indices**2) * self.curvature_y
        star_array += (x_indices * y_indices) * self.curvature_xy

        # Create MaskedImage
        stampIm = ImageF(star_array.astype(np.float32))
        stampVa = ImageF(stampIm.getBBox(), 1.0)
        self.stampMI = MaskedImageF(image=stampIm, variance=stampVa)

        # Initialize the mask planes required
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

    def test_applyStampFit(self):
        """Test that _applyStampFit correctly removes background and normalizes."""
        config = BrightStarStackConfig()
        task = BrightStarStackTask(config=config)

        # Create a mock stamp
        stamp_mi_copy = self.stampMI.clone()
        mock_stamp = MockStamp(stamp_mi_copy, mag=10.0, fit_params=self.fit_params, stats={})

        # Run the method
        task._applyStampFit(mock_stamp)

        # The result should be the clean array (normalized to scale 1.0)
        result_array = mock_stamp.stamp_im.image.array

        # Allow for small floating point discrepancies
        np.testing.assert_allclose(result_array, self.clean_array, atol=1e-5)

    def test_run(self):
        """Test the full run method: filtering, binning, and stacking."""
        config = BrightStarStackConfig()
        # Set config to ensure our test stamps are included
        config.magnitude_bins = [11, 9]
        config.subset_stamp_number = [1]
        config.stack_type = "MEDIAN"

        task = BrightStarStackTask(config=config)

        valid_stats = {"global_chi2": 1.0, "psf_chi2": 1.0, "fp_radius": 100.0}
        invalid_stats = {"global_chi2": 1e9, "psf_chi2": 1e9, "fp_radius": 100.0}

        stamp1 = MockStamp(self.stampMI.clone(), mag=10.0, fit_params=self.fit_params, stats=valid_stats)
        stamp2 = MockStamp(self.stampMI.clone(), mag=10.0, fit_params=self.fit_params, stats=valid_stats)

        # This stamp should be ignored
        bad_stamp = MockStamp(self.stampMI.clone(), mag=10.0, fit_params=self.fit_params, stats=invalid_stats)

        # Create mock input structure
        # brightStarStamps is a list of handles
        input_stamps = [MockHandle([stamp1, bad_stamp]), MockHandle([stamp2])]

        result = task.run(brightStarStamps=input_stamps)

        # Verify output exists
        self.assertIsNotNone(result.extendedPsf)

        # Verify output dimensions match input
        self.assertEqual(result.extendedPsf.getDimensions(), self.stampMI.getDimensions())

        # Verify the calculation
        # Since we stacked identical "clean" stamps (after fit application),
        # the result should match self.clean_array
        result_array = result.extendedPsf.image.array
        np.testing.assert_allclose(result_array, self.clean_array, atol=1e-5)

    def test_filtering_logic(self):
        """Test that stamps outside focal plane radius or thresholds are skipped."""
        config = BrightStarStackConfig()
        config.min_focal_plane_radius = 50.0
        config.max_focal_plane_radius = 150.0
        config.global_reduced_chi_squared_threshold = 5.0
        config.magnitude_bins = [15, 11, 9]
        config.subset_stamp_number = [100, 1]

        task = BrightStarStackTask(config=config)

        good_stats = {"global_chi2": 1.0, "psf_chi2": 1.0, "fp_radius": 100.0}
        bad_radius_low = {"global_chi2": 1.0, "psf_chi2": 1.0, "fp_radius": 10.0}
        bad_radius_high = {"global_chi2": 1.0, "psf_chi2": 1.0, "fp_radius": 2000.0}
        bad_chi2 = {"global_chi2": 100.0, "psf_chi2": 1.0, "fp_radius": 100.0}

        stamps = [
            MockStamp(self.stampMI.clone(), 10.0, self.fit_params, good_stats),
            MockStamp(self.stampMI.clone(), 10.0, self.fit_params, bad_radius_low),
            MockStamp(self.stampMI.clone(), 10.0, self.fit_params, bad_radius_high),
            MockStamp(self.stampMI.clone(), 10.0, self.fit_params, bad_chi2),
        ]

        input_stamps = [MockHandle(stamps)]
        task.run(brightStarStamps=input_stamps)

        bin_key = "9"  # Based on config magnitude_bins=[11, 9], the lower bound is 9

        self.assertEqual(task.metadata["psf_star_count"]["all"], 4)

        self.assertEqual(task.metadata["psf_star_count"][bin_key], 2)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
