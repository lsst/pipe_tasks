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
from astropy.table import Table
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS
from lsst.afw.cameraGeom.testUtils import CameraWrapper
from lsst.afw.geom import makeCdMatrix, makeSkyWcs
from lsst.afw.image import ExposureF, ImageD, ImageF, MaskedImageF, makePhotoCalibFromCalibZeroPoint
from lsst.afw.math import FixedKernel
from lsst.geom import Box2I, Extent2I, Point2D, Point2I, SpherePoint, arcseconds, degrees
from lsst.meas.algorithms import KernelPsf
from lsst.pipe.tasks.brightStarSubtraction import (
    BrightStarCutoutConfig,
    BrightStarCutoutTask,
    BrightStarStackConfig,
    BrightStarStackTask,
)


class BrightStarSubtractionTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)

        # Background coefficients
        sigma = 60.0  # noise
        pedestal = 3210.1
        coef_x = 1e-2
        coef_y = 2e-2
        coef_x2 = 1e-5
        coef_xy = 2e-5
        coef_y2 = 3e-5

        # Make an input exposure
        wcs = makeSkyWcs(
            crpix=Point2D(0, 0),
            crval=SpherePoint(0, 0, degrees),
            cdMatrix=makeCdMatrix(scale=0.2 * arcseconds, flipX=True),
        )
        self.exposure = ExposureF(1001, 1001, wcs)
        self.exposure.setPhotoCalib(makePhotoCalibFromCalibZeroPoint(10 ** (0.4 * 30), 1.0))
        ny, nx = self.exposure.image.array.shape
        grid_y, grid_x = np.mgrid[(-ny + 1) // 2 : ny // 2 + 1, (-nx + 1) // 2 : nx // 2 + 1]
        self.exposure.image.array[:] += rng.normal(scale=sigma, size=self.exposure.image.array.shape)
        self.exposure.image.array += pedestal
        self.exposure.image.array += coef_x * grid_x
        self.exposure.image.array += coef_y * grid_y
        self.exposure.image.array += coef_x2 * grid_x * grid_x
        self.exposure.image.array += coef_xy * grid_x * grid_y
        self.exposure.image.array += coef_y2 * grid_y * grid_y
        camera = CameraWrapper().camera
        detector = camera[10]
        self.exposure.setDetector(detector)
        for mask_plane in [
            "BAD",
            "CR",
            "CROSSTALK",
            "EDGE",
            "NO_DATA",
            "SAT",
            "SUSPECT",
            "UNMASKEDNAN",
            "NEIGHBOR",
        ]:
            _ = self.exposure.mask.addMaskPlane(mask_plane)
        self.exposure.variance.array.fill(1.0)

        # Make a bright stars table
        corners = self.exposure.wcs.pixelToSky([Point2D(x) for x in self.exposure.getBBox().getCorners()])
        corner_ras = [corner.getRa().asDegrees() for corner in corners]
        corner_decs = [corner.getDec().asDegrees() for corner in corners]
        num_stars = 10
        ras = rng.uniform(np.min(corner_ras), np.max(corner_ras), num_stars)
        decs = rng.uniform(np.min(corner_decs), np.max(corner_decs), num_stars)
        sky_coords = [SpherePoint(ra, dec, degrees) for ra, dec in zip(ras, decs)]
        pixel_coords = self.exposure.wcs.skyToPixel(sky_coords)
        pixel_x = [coord.getX() for coord in pixel_coords]
        pixel_y = [coord.getY() for coord in pixel_coords]
        mags = rng.uniform(10, 20, num_stars)
        fluxes = [self.exposure.photoCalib.magnitudeToInstFlux(mag) for mag in mags]
        mm_coords = detector.transform(pixel_coords, PIXELS, FOCAL_PLANE)
        mm_coords_x = np.array([mm_coord.x for mm_coord in mm_coords])
        mm_coords_y = np.array([mm_coord.y for mm_coord in mm_coords])
        radius_mm = np.sqrt(mm_coords_x**2 + mm_coords_y**2)
        theta_radians = np.arctan2(mm_coords_y, mm_coords_x)
        self.bright_stars = Table(
            {
                "id": np.arange(num_stars),
                "coord_ra": ras,
                "coord_dec": decs,
                "phot_g_mean_flux": fluxes,
                "mag": mags,
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "radius_mm": radius_mm,
                "theta_radians": theta_radians,
            }
        )

        # Make a synthetic star
        stamp_radius = 25
        grid_y, grid_x = np.mgrid[-stamp_radius : stamp_radius + 1, -stamp_radius : stamp_radius + 1]
        dist_from_center = np.sqrt(grid_x**2 + grid_y**2)
        sigma = 1.5
        psf_array = np.exp(-(dist_from_center**2) / (2 * sigma**2))
        psf_array /= np.sum(psf_array)
        fixed_kernel = FixedKernel(ImageD(psf_array))
        kernel_psf = KernelPsf(fixed_kernel)
        self.exposure.setPsf(kernel_psf)
        psf = kernel_psf.computeKernelImage(kernel_psf.getAveragePosition())

        # Add synthetic stars to the exposure
        footprints = ImageF(self.exposure.getBBox())
        for bright_star_id, bright_star in enumerate(self.bright_stars):
            bbox_star = Box2I(
                Point2I(bright_star["pixel_x"], bright_star["pixel_y"]), Extent2I(1, 1)
            ).dilatedBy(stamp_radius)
            bbox_star_clipped = bbox_star.clippedTo(self.exposure.getBBox())
            bright_star_im = MaskedImageF(bbox_star)
            bright_star_im.image.array = bright_star["phot_g_mean_flux"] * psf.getArray()
            bright_star_im = bright_star_im[bbox_star_clipped]
            detection_threshold = self.exposure.getPhotoCalib().magnitudeToInstFlux(25)
            detected = bright_star_im.image.array > detection_threshold
            footprints[bbox_star_clipped].array = detected * (bright_star_id + 1)
            _ = bright_star_im.mask.addMaskPlane("DETECTED")
            bright_star_im.mask.array[detected] |= bright_star_im.mask.getPlaneBitMask("DETECTED")
            bright_star_im.variance.array.fill(1.0)
            self.exposure.maskedImage[bbox_star_clipped] += bright_star_im
        self.footprints = footprints.array

        # Run the bright star cutout task
        brightStarCutoutConfig = BrightStarCutoutConfig()
        brightStarCutoutTask = BrightStarCutoutTask(config=brightStarCutoutConfig)
        self.bright_star_stamps = brightStarCutoutTask._get_bright_star_stamps(
            input_exposure=self.exposure,
            input_background=None,
            footprints=self.footprints,
            bright_stars=self.bright_stars,
            visit=12345,
        )
        assert self.bright_star_stamps is not None

        # Run the bright star stack task
        brightStarStackConfig = BrightStarStackConfig()
        brightStarStackTask = BrightStarStackTask(config=brightStarStackConfig)
        bss_results = brightStarStackTask.run(bright_star_stamps=self.bright_star_stamps)
        assert bss_results is not None
        self.extended_psf = bss_results.extended_psf
        self.moffat_results = bss_results.moffat_results

    def test_brightStarCutout(self):
        """Test BrightStarCutoutTask."""
        assert self.bright_star_stamps is not None
        self.assertAlmostEqual(self.bright_star_stamps.metadata["FOCAL_PLANE_RADIUS_MIN"], 5.22408977, 7)
        self.assertAlmostEqual(self.bright_star_stamps.metadata["FOCAL_PLANE_RADIUS_MAX"], 14.6045832, 7)
        self.assertEqual(len(self.bright_star_stamps), len(self.bright_stars))
        self.assertEqual(self.bright_star_stamps[0].visit, 12345)
        self.assertEqual(self.bright_star_stamps[0].detector, 10)

        for bright_star_stamp, bright_star_row in zip(self.bright_star_stamps, self.bright_stars):
            self.assertEqual(bright_star_stamp.ref_id, bright_star_row["id"])
            self.assertEqual(bright_star_stamp.ref_mag, bright_star_row["mag"])
            assert bright_star_stamp.position is not None
            self.assertEqual(bright_star_stamp.position.getX(), bright_star_row["pixel_x"])
            self.assertEqual(bright_star_stamp.position.getY(), bright_star_row["pixel_y"])
            self.assertEqual(bright_star_stamp.focal_plane_radius, bright_star_row["radius_mm"])
            assert bright_star_stamp.focal_plane_angle is not None
            focal_plane_angle = bright_star_stamp.focal_plane_angle.asRadians()
            self.assertEqual(focal_plane_angle, bright_star_row["theta_radians"])

    def test_brightStarStack(self):
        """Test BrightStarStackTask."""
        assert self.extended_psf is not None
        assert self.moffat_results is not None
        self.assertAlmostEqual(np.sum(self.extended_psf.image.array), 0.8233416, places=5)
        self.assertAlmostEqual(np.sum(self.extended_psf.variance.array), 0.007561891, places=5)
        self.assertAlmostEqual(self.moffat_results["MOFFAT_AMPLITUDE"], 0.078900464260488, places=5)
        self.assertAlmostEqual(self.moffat_results["MOFFAT_X_0"], -0.68834523633912, places=5)
        self.assertAlmostEqual(self.moffat_results["MOFFAT_Y_0"], -0.069005412739451, places=5)
        self.assertAlmostEqual(self.moffat_results["MOFFAT_GAMMA"], 8.0966823485900, places=5)
        self.assertAlmostEqual(self.moffat_results["MOFFAT_ALPHA"], 16.048683662812, places=5)
        self.assertAlmostEqual(self.moffat_results["MOFFAT_CHI2"], 107652.97393353, places=5)
        self.assertAlmostEqual(self.moffat_results["MOFFAT_REDUCED_CHI2"], 1.7088858647141, places=5)


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
