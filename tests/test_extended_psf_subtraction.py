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
from typing import cast

import astropy.units as u
import lsst.utils.tests
import numpy as np
from astropy.table import Table
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS
from lsst.afw.cameraGeom.testUtils import CameraWrapper
from lsst.afw.geom import makeCdMatrix, makeSkyWcs
from lsst.afw.image import (
    ExposureF,
    ImageD,
    ImageF,
    MaskedImageF,
    VisitInfo,
    makePhotoCalibFromCalibZeroPoint,
)
from lsst.afw.math import FixedKernel
from lsst.geom import Box2I, Extent2I, Point2D, Point2I, SpherePoint, arcseconds, degrees
from lsst.images import Image
from lsst.meas.algorithms import KernelPsf
from lsst.pipe.tasks.extended_psf import ExtendedPsfCutoutConfig, ExtendedPsfCutoutTask


class ExtendedPsfSubtractionTestCase(lsst.utils.tests.TestCase):
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
        self.exposure.info.setVisitInfo(VisitInfo(id=12345))
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

        # Make a table of extended PSF candidate stars
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
        self.extended_psf_candidate_table = Table(
            {
                "id": np.arange(num_stars),
                "coord_ra": ras,
                "coord_dec": decs,
                "phot_g_mean_flux": fluxes,
                "mag": mags,
                "pixel_x": pixel_x,
                "pixel_y": pixel_y,
                "radius_mm": radius_mm,
                "angle_radians": theta_radians,
            }
        )

        # Make a synthetic star
        cutout_radius = 25
        grid_y, grid_x = np.mgrid[-cutout_radius : cutout_radius + 1, -cutout_radius : cutout_radius + 1]
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
        for candidate_id, candidate in enumerate(self.extended_psf_candidate_table):
            bbox_star = Box2I(Point2I(candidate["pixel_x"], candidate["pixel_y"]), Extent2I(1, 1)).dilatedBy(
                cutout_radius
            )
            bbox_star_clipped = bbox_star.clippedTo(self.exposure.getBBox())
            candidate_im = MaskedImageF(bbox_star)
            candidate_im.image.array = candidate["phot_g_mean_flux"] * psf.getArray()
            candidate_im = candidate_im[bbox_star_clipped]
            detection_threshold = self.exposure.getPhotoCalib().magnitudeToInstFlux(25)
            detected = candidate_im.image.array > detection_threshold
            footprints[bbox_star_clipped].array = detected * (candidate_id + 1)
            _ = candidate_im.mask.addMaskPlane("DETECTED")
            candidate_im.mask.array[detected] |= candidate_im.mask.getPlaneBitMask("DETECTED")
            candidate_im.variance.array.fill(1.0)
            self.exposure.maskedImage[bbox_star_clipped] += candidate_im
        self.footprints = footprints.array

        # Run the cutout task
        extendedPsfCutoutConfig = ExtendedPsfCutoutConfig()
        extendedPsfCutoutTask = ExtendedPsfCutoutTask(config=extendedPsfCutoutConfig)
        self.extended_psf_candidates = extendedPsfCutoutTask._get_extended_psf_candidates(
            input_exposure=self.exposure,
            input_background=None,
            footprints=self.footprints,
            extended_psf_candidate_table=self.extended_psf_candidate_table,
        )

    def test_extendedPsfCutout(self):
        """Test ExtendedPsfCutoutTask."""
        assert self.extended_psf_candidates is not None
        self.assertAlmostEqual(
            float(self.extended_psf_candidates.metadata["FOCAL_PLANE_RADIUS_MM_MIN"]), 5.22408977, 7
        )
        self.assertAlmostEqual(
            float(self.extended_psf_candidates.metadata["FOCAL_PLANE_RADIUS_MM_MAX"]), 14.6045832, 7
        )
        self.assertEqual(len(self.extended_psf_candidates), len(self.extended_psf_candidate_table))
        self.assertEqual(self.extended_psf_candidates[0].star_info.visit, 12345)
        self.assertEqual(self.extended_psf_candidates[0].star_info.detector, 10)

        for candidate, candidate_entry in zip(
            self.extended_psf_candidates, self.extended_psf_candidate_table
        ):
            self.assertEqual(candidate.star_info.ref_id, candidate_entry["id"])
            self.assertEqual(candidate.star_info.ref_mag, candidate_entry["mag"])
            self.assertEqual(candidate.star_info.position_x, candidate_entry["pixel_x"])
            self.assertEqual(candidate.star_info.position_y, candidate_entry["pixel_y"])
            self.assertIsInstance(candidate.psf_kernel_image, Image)
            self.assertEqual(candidate.psf_kernel_image.array.ndim, 2)
            self.assertGreater(candidate.psf_kernel_image.array.shape[0], 0)
            self.assertGreater(candidate.psf_kernel_image.array.shape[1], 0)
            self.assertTrue(np.isfinite(candidate.psf_kernel_image.array).all())
            self.assertGreater(candidate.psf_kernel_image.array.sum(), 0.0)
            focal_plane_radius = cast(u.Quantity, candidate.star_info.focal_plane_radius)
            focal_plane_angle = cast(u.Quantity, candidate.star_info.focal_plane_angle)
            self.assertEqual(focal_plane_radius.to_value(u.mm), candidate_entry["radius_mm"])
            self.assertEqual(focal_plane_angle.to_value(u.rad), candidate_entry["angle_radians"])


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
