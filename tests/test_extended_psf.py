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

import logging
import unittest
from typing import cast

import astropy.units as u
import lsst.images.tests
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
from lsst.afw.math import (
    REDUCE_INTERP_ORDER,
    ApproximateControl,
    BackgroundControl,
    BackgroundList,
    FixedKernel,
    Interpolate,
    makeBackground,
)
from lsst.geom import Box2I, Extent2I, Point2D, Point2I, SpherePoint, arcseconds, degrees
from lsst.images import Box, Image, Mask, MaskedImage, MaskSchema
from lsst.meas.algorithms import KernelPsf
from lsst.pipe.tasks.extended_psf import (
    ExtendedPsfCandidate,
    ExtendedPsfCandidateInfo,
    ExtendedPsfCandidates,
    ExtendedPsfCutoutConfig,
    ExtendedPsfCutoutTask,
    ExtendedPsfFit,
    ExtendedPsfImage,
    ExtendedPsfImageInfo,
    ExtendedPsfMoffatFit,
    ExtendedPsfStackConfig,
    ExtendedPsfStackTask,
    ExtendedPsfSubtractConfig,
    ExtendedPsfSubtractTask,
)


class ExtendedPsfCandidatesTestCase(lsst.utils.tests.TestCase):
    """Test ExtendedPsfCandidate and ExtendedPsfCandidates data structures."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.Generator(np.random.MT19937(seed=5))
        cutout_size = (25, 25)

        # Generate simulated stars
        extended_psf_candidates = []
        cls.star_infos = []
        mask_schema = MaskSchema([])

        for i in range(3):
            candidate_masked_image = MaskedImage(
                image=Image(rng.random(cutout_size).astype(np.float32)),
                mask=Mask(0, schema=mask_schema, shape=cutout_size),
                variance=Image(rng.random(cutout_size).astype(np.float32)),
            )

            star_info = ExtendedPsfCandidateInfo(
                visit=100 + i,
                detector=200 + i,
                ref_id=1000 + i,
                ref_mag=10.0 + i,
                position_x=float(rng.random()),
                position_y=float(rng.random()),
                focal_plane_radius=float(rng.random()) * u.mm,
                focal_plane_angle=float(rng.random()) * u.rad,
            )
            cls.star_infos.append(star_info)

            # Build a normalized 2-D Gaussian kernel image directly.
            yy, xx = np.mgrid[: cutout_size[0], : cutout_size[1]]
            cy = (cutout_size[0] - 1) / 2.0
            cx = (cutout_size[1] - 1) / 2.0
            sigma = 1.5
            kernel_array = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))
            kernel_array /= np.sum(kernel_array)
            psf_kernel_image = Image(kernel_array.astype(np.float64))

            extended_psf_candidates.append(
                ExtendedPsfCandidate(
                    image=candidate_masked_image.image,
                    mask=candidate_masked_image.mask,
                    variance=candidate_masked_image.variance,
                    psf_kernel_image=psf_kernel_image,
                    star_info=star_info,
                )
            )

        cls.global_metadata = {"CANDIDATES_TEST_KEY": "CANDIDATES_TEST_VALUE"}
        cls.extended_psf_candidates = ExtendedPsfCandidates(extended_psf_candidates, cls.global_metadata)

    @classmethod
    def tearDownClass(cls):
        del cls.extended_psf_candidates
        del cls.star_infos
        del cls.global_metadata

    def test_fits_roundtrip(self):
        """Test that ExtendedPsfCandidates can be serialized/deserialized."""

        with lsst.images.tests.RoundtripFits(
            self, self.extended_psf_candidates, storage_class="ExtendedPsfCandidates"
        ) as roundtrip:
            pass
        extended_psf_candidates = roundtrip.result

        global_metadata = extended_psf_candidates.metadata
        self.assertEqual(self.global_metadata["CANDIDATES_TEST_KEY"], global_metadata["CANDIDATES_TEST_KEY"])
        self.assertEqual(len(self.extended_psf_candidates), len(extended_psf_candidates))
        for input_info, input_candidate, output_candidate in zip(
            self.star_infos, self.extended_psf_candidates, extended_psf_candidates
        ):
            self.assertEqual(input_candidate.metadata, {})
            self.assertEqual(output_candidate.metadata, {})

            output_info = output_candidate.star_info
            self.assertEqual(input_info.visit, output_info.visit)
            self.assertEqual(input_info.detector, output_info.detector)
            self.assertEqual(input_info.ref_id, output_info.ref_id)
            self.assertAlmostEqual(input_info.ref_mag, output_info.ref_mag, places=10)
            self.assertAlmostEqual(input_info.position_x, output_info.position_x, places=10)
            self.assertAlmostEqual(input_info.position_y, output_info.position_y, places=10)

            self.assertIsNotNone(input_info.focal_plane_radius)
            self.assertIsNotNone(output_info.focal_plane_radius)
            input_radius = cast(u.Quantity, input_info.focal_plane_radius)
            output_radius = cast(u.Quantity, output_info.focal_plane_radius)
            self.assertAlmostEqual(input_radius.to_value(u.mm), output_radius.to_value(u.mm), places=10)

            self.assertIsNotNone(input_info.focal_plane_angle)
            self.assertIsNotNone(output_info.focal_plane_angle)
            input_angle = cast(u.Quantity, input_info.focal_plane_angle)
            output_angle = cast(u.Quantity, output_info.focal_plane_angle)
            self.assertAlmostEqual(input_angle.to_value(u.rad), output_angle.to_value(u.rad), places=10)

            np.testing.assert_allclose(
                input_candidate.image.array,
                output_candidate.image.array,
                rtol=0.0,
                atol=1e-10,
            )
            np.testing.assert_array_equal(
                input_candidate.mask.array,
                output_candidate.mask.array,
            )
            np.testing.assert_allclose(
                input_candidate.variance.array, output_candidate.variance.array, rtol=0.0, atol=1e-10
            )
            np.testing.assert_allclose(
                input_candidate.psf_kernel_image.array,
                output_candidate.psf_kernel_image.array,
                rtol=0.0,
                atol=1e-10,
            )

    def test_ref_id_map(self):
        """Test that ref_id_map maps reference catalog IDs to candidates."""
        ref_id_map = self.extended_psf_candidates.ref_id_map
        self.assertEqual(len(ref_id_map), len(self.star_infos))
        for i, star_info in enumerate(self.star_infos):
            self.assertIn(star_info.ref_id, ref_id_map)
            self.assertIs(ref_id_map[star_info.ref_id], self.extended_psf_candidates[i])

    def test_slice_preserves_subcollection_and_metadata(self):
        """Test that slicing candidates returns a sub-collection."""
        sub = self.extended_psf_candidates[1:3]  # length=2
        self.assertIsInstance(sub, ExtendedPsfCandidates)
        self.assertEqual(len(sub), 2)
        self.assertEqual(sub.metadata, self.global_metadata)
        self.assertIs(sub[0], self.extended_psf_candidates[1])
        self.assertIs(sub[1], self.extended_psf_candidates[2])

    def test_spatial_box_slicing(self):
        """Test that candidates support spatial slicing with a Box."""
        candidate = self.extended_psf_candidates[0]
        subbox = Box.from_shape((10, 10), start=(5, 5))
        sub = candidate[subbox]
        self.assertIsInstance(sub, ExtendedPsfCandidate)
        self.assertEqual(sub.bbox, subbox)
        self.assertIs(sub.star_info, candidate.star_info)
        np.testing.assert_array_equal(sub.image.array, candidate.image[subbox].array)
        np.testing.assert_array_equal(sub.mask.array, candidate.mask[subbox].array)

    def test_copy(self):
        """Test that copy() produces an independent deep copy."""
        candidate = self.extended_psf_candidates[0]
        copied = candidate.copy()
        self.assertIsInstance(copied, ExtendedPsfCandidate)
        np.testing.assert_array_equal(copied.image.array, candidate.image.array)
        np.testing.assert_array_equal(copied.variance.array, candidate.variance.array)
        # Modifying the copy must not affect the original.
        copied.image.array[:] = 0.0
        self.assertFalse(np.all(candidate.image.array == 0.0))


class ExtendedPsfImageTestCase(lsst.utils.tests.TestCase):
    """Test ExtendedPsfImage data model, properties, and operations."""

    def setUp(self):
        image = Image(
            np.arange(30, dtype=np.float32).reshape(5, 6),
            yx0=(10, -3),
            unit=u.nJy,
        )
        variance = Image(
            np.full((5, 6), 2.5, dtype=np.float32),
            bbox=image.bbox,
            unit=u.nJy**2,
        )
        info = ExtendedPsfImageInfo(
            n_stars=17,
        )
        fit = ExtendedPsfMoffatFit(
            chi2=12.3,
            reduced_chi2=1.23,
            dof=10,
            amplitude=0.8,
            x_0=-0.2,
            y_0=0.4,
            gamma=2.3,
            alpha=4.5,
        )
        self.extended_psf_image = ExtendedPsfImage(
            image=image,
            variance=variance,
            info=info,
            fit=fit,
            metadata={"EPSF_TEST_KEY": "EPSF_TEST VALUE"},
        )

    def tearDown(self):
        del self.extended_psf_image

    def test_fits_roundtrip(self):
        """Test that ExtendedPsfImage can be serialized/deserialized."""
        with lsst.images.tests.RoundtripFits(
            self,
            self.extended_psf_image,
            storage_class="ExtendedPsfImage",
        ) as roundtrip:
            subbox = Box.from_shape((3, 3), start=(11, -1))
            subimage = roundtrip.get(bbox=subbox)
            expected_subimage = self.extended_psf_image[subbox]

            np.testing.assert_allclose(
                subimage.image.array,
                expected_subimage.image.array,
                rtol=0.0,
                atol=1e-8,
            )
            np.testing.assert_allclose(
                subimage.variance.array,
                expected_subimage.variance.array,
                rtol=0.0,
                atol=1e-8,
            )
            self.assertEqual(subimage.info, self.extended_psf_image.info)
            self.assertEqual(subimage.fit, self.extended_psf_image.fit)

        roundtripped = roundtrip.result
        np.testing.assert_allclose(
            roundtripped.image.array,
            self.extended_psf_image.image.array,
            rtol=0.0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            roundtripped.variance.array,
            self.extended_psf_image.variance.array,
            rtol=0.0,
            atol=1e-8,
        )
        self.assertEqual(roundtripped.info, self.extended_psf_image.info)
        self.assertEqual(roundtripped.fit, self.extended_psf_image.fit)
        self.assertEqual(roundtripped.metadata["EPSF_TEST_KEY"], "EPSF_TEST VALUE")

    def test_unit_sky_projection_bbox_properties(self):
        """Test ExtendedPsfImage properties: unit, sky_projection, and bbox."""
        self.assertEqual(self.extended_psf_image.unit, u.nJy)
        self.assertIsNone(self.extended_psf_image.sky_projection)
        self.assertEqual(self.extended_psf_image.bbox, self.extended_psf_image.image.bbox)

    def test_copy_independence(self):
        """Test that copy() produces an independent deep copy."""
        copied = self.extended_psf_image.copy()
        np.testing.assert_array_equal(copied.image.array, self.extended_psf_image.image.array)
        np.testing.assert_array_equal(copied.variance.array, self.extended_psf_image.variance.array)
        self.assertEqual(copied.info, self.extended_psf_image.info)
        self.assertEqual(copied.fit, self.extended_psf_image.fit)
        # Modifying the copy must not affect the original.
        copied.image.array[:] = 0.0
        self.assertFalse(np.all(self.extended_psf_image.image.array == 0.0))

    def test_setitem_subregion_assignment(self):
        """Test __setitem__ writes image and variance into a subregion."""
        target = ExtendedPsfImage(Image(np.zeros((5, 6), dtype=np.float32), yx0=(10, -3), unit=u.nJy))
        subbox = Box.from_shape((3, 3), start=(11, -1))
        target[subbox] = self.extended_psf_image[subbox]
        np.testing.assert_array_equal(
            target[subbox].image.array,
            self.extended_psf_image[subbox].image.array,
        )
        np.testing.assert_array_equal(
            target[subbox].variance.array,
            self.extended_psf_image[subbox].variance.array,
        )

    def test_constructor_invalid_inputs(self):
        """Test that ValueError is raised for invalid constructor inputs."""
        # Mismatched bboxes
        image = Image(np.ones((5, 6), dtype=np.float32))
        variance = Image(np.ones((4, 6), dtype=np.float32))
        with self.assertRaises(ValueError):
            ExtendedPsfImage(image=image, variance=variance)

        # Variance has wrong units (nJy instead of nJy**2).
        image = Image(np.ones((5, 6), dtype=np.float32), unit=u.nJy)
        variance_wrong_units = Image(np.ones((5, 6), dtype=np.float32), unit=u.nJy)
        with self.assertRaises(ValueError):
            ExtendedPsfImage(image=image, variance=variance_wrong_units)

        # Image has no units but variance does.
        image_no_units = Image(np.ones((5, 6), dtype=np.float32))
        variance_with_units = Image(np.ones((5, 6), dtype=np.float32), unit=u.nJy**2)
        with self.assertRaises(ValueError):
            ExtendedPsfImage(image=image_no_units, variance=variance_with_units)


class ExtendedPsfFitTestCase(lsst.utils.tests.TestCase):
    """Test ExtendedPsfFit."""

    def test_construction(self):
        """Test construction with and without optional fields."""
        # With all fields
        fit = ExtendedPsfFit(chi2=10.5, dof=10, reduced_chi2=1.05)
        self.assertEqual(fit.chi2, 10.5)
        self.assertEqual(fit.dof, 10)
        self.assertEqual(fit.reduced_chi2, 1.05)

        # With only the required field
        fit_minimal = ExtendedPsfFit(chi2=10.5)
        self.assertEqual(fit_minimal.chi2, 10.5)
        self.assertIsNone(fit_minimal.dof)
        self.assertIsNone(fit_minimal.reduced_chi2)


class ExtendedPsfSubtractionTestCase(lsst.utils.tests.TestCase):
    """Integration tests for all extended PSF subtraction pipeline tasks."""

    @classmethod
    def setUpClass(cls):
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
        cls.exposure = ExposureF(1001, 1001, wcs)
        cls.exposure.setPhotoCalib(makePhotoCalibFromCalibZeroPoint(10 ** (0.4 * 30), 1.0))
        ny, nx = cls.exposure.image.array.shape
        grid_y, grid_x = np.mgrid[(-ny + 1) // 2 : ny // 2 + 1, (-nx + 1) // 2 : nx // 2 + 1]
        cls.exposure.image.array[:] += rng.normal(scale=sigma, size=cls.exposure.image.array.shape)
        cls.background_array = np.zeros_like(cls.exposure.image.array)
        cls.background_array += pedestal
        cls.background_array += coef_x * grid_x
        cls.background_array += coef_y * grid_y
        cls.background_array += coef_x2 * grid_x * grid_x
        cls.background_array += coef_xy * grid_x * grid_y
        cls.background_array += coef_y2 * grid_y * grid_y
        cls.exposure.image.array += cls.background_array
        cls.exposure.info.setVisitInfo(VisitInfo(id=12345))
        camera = CameraWrapper().camera
        detector = camera[10]
        cls.exposure.setDetector(detector)
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
            _ = cls.exposure.mask.addMaskPlane(mask_plane)
        cls.exposure.variance.array.fill(1.0)

        # Make a background list
        background_image = ImageF(cls.exposure.getBBox())
        background_image.array[:] = cls.background_array
        background_control = BackgroundControl(Interpolate.NATURAL_SPLINE)
        background_control.setNxSample(background_image.getWidth())
        background_control.setNySample(background_image.getHeight())
        background_model = makeBackground(background_image, background_control)
        cls.background = BackgroundList()
        cls.background.append(
            (
                background_model,
                Interpolate.NATURAL_SPLINE,
                REDUCE_INTERP_ORDER,
                ApproximateControl.UNKNOWN,
                0,
                0,
                False,
            )
        )

        # Make a table of extended PSF candidate stars
        corners = cls.exposure.wcs.pixelToSky([Point2D(x) for x in cls.exposure.getBBox().getCorners()])
        corner_ras = [corner.getRa().asDegrees() for corner in corners]
        corner_decs = [corner.getDec().asDegrees() for corner in corners]
        num_stars = 10
        ras = rng.uniform(np.min(corner_ras), np.max(corner_ras), num_stars)
        decs = rng.uniform(np.min(corner_decs), np.max(corner_decs), num_stars)
        sky_coords = [SpherePoint(ra, dec, degrees) for ra, dec in zip(ras, decs)]
        pixel_coords = cls.exposure.wcs.skyToPixel(sky_coords)
        pixel_x = [coord.getX() for coord in pixel_coords]
        pixel_y = [coord.getY() for coord in pixel_coords]
        mags = rng.uniform(10, 20, num_stars)
        fluxes = [cls.exposure.photoCalib.magnitudeToInstFlux(mag) for mag in mags]
        mm_coords = detector.transform(pixel_coords, PIXELS, FOCAL_PLANE)
        mm_coords_x = np.array([mm_coord.x for mm_coord in mm_coords])
        mm_coords_y = np.array([mm_coord.y for mm_coord in mm_coords])
        radius_mm = np.sqrt(mm_coords_x**2 + mm_coords_y**2)
        theta_radians = np.arctan2(mm_coords_y, mm_coords_x)
        cls.extended_psf_candidate_table = Table(
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
        cls.exposure.setPsf(kernel_psf)
        psf = kernel_psf.computeKernelImage(kernel_psf.getAveragePosition())

        # Add synthetic stars to the exposure
        footprints = ImageF(cls.exposure.getBBox())
        cls.injected_stars_array = np.zeros_like(cls.exposure.image.array)
        for candidate_id, candidate in enumerate(cls.extended_psf_candidate_table):
            bbox_star = Box2I(Point2I(candidate["pixel_x"], candidate["pixel_y"]), Extent2I(1, 1)).dilatedBy(
                cutout_radius
            )
            bbox_star_clipped = bbox_star.clippedTo(cls.exposure.getBBox())
            candidate_im = MaskedImageF(bbox_star)
            candidate_im.image.array = candidate["phot_g_mean_flux"] * psf.getArray()
            candidate_im = candidate_im[bbox_star_clipped]
            detection_threshold = cls.exposure.getPhotoCalib().magnitudeToInstFlux(25)
            detected = candidate_im.image.array > detection_threshold
            footprints[bbox_star_clipped].array = detected * (candidate_id + 1)
            _ = candidate_im.mask.addMaskPlane("DETECTED")
            candidate_im.mask.array[detected] |= candidate_im.mask.getPlaneBitMask("DETECTED")
            candidate_im.variance.array.fill(1.0)
            y_slice = slice(bbox_star_clipped.getMinY(), bbox_star_clipped.getMaxY() + 1)
            x_slice = slice(bbox_star_clipped.getMinX(), bbox_star_clipped.getMaxX() + 1)
            cls.injected_stars_array[y_slice, x_slice] += candidate_im.image.array
            cls.exposure.maskedImage[bbox_star_clipped] += candidate_im
        cls.footprints = footprints.array

        # Run the cutout task
        extendedPsfCutoutConfig = ExtendedPsfCutoutConfig()
        extendedPsfCutoutTask = ExtendedPsfCutoutTask(config=extendedPsfCutoutConfig)
        cls.extended_psf_candidates = extendedPsfCutoutTask._get_extended_psf_candidates(
            input_exposure=cls.exposure,
            input_background=None,
            footprints=cls.footprints,
            extended_psf_candidate_table=cls.extended_psf_candidate_table,
        )

        # Run the stack task
        extendedPsfStackConfig = ExtendedPsfStackConfig()
        extendedPsfStackTask = ExtendedPsfStackTask(config=extendedPsfStackConfig)
        stack_result = extendedPsfStackTask.run(extended_psf_candidates=cls.extended_psf_candidates)
        cls.extended_psf = stack_result.extended_psf if stack_result is not None else None

    @classmethod
    def tearDownClass(cls):
        del cls.exposure
        del cls.background
        del cls.background_array
        del cls.injected_stars_array
        del cls.extended_psf_candidate_table
        del cls.footprints
        del cls.extended_psf_candidates
        del cls.extended_psf

    def test_cutout_task_candidate_extraction(self):
        """Test ExtendedPsfCutoutTask candidate extraction."""
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

    def test_stack_task_moffat_fitting(self):
        """Test Moffat fitting."""
        assert self.extended_psf is not None
        self.assertAlmostEqual(np.sum(self.extended_psf.image.array), 0.8233417, places=2)
        self.assertAlmostEqual(np.sum(self.extended_psf.variance.array), 0.0075618913, places=4)
        fit = self.extended_psf.fit
        self.assertAlmostEqual(fit.chi2, 107652.97393353, delta=5)
        self.assertEqual(fit.dof, 62996)
        self.assertAlmostEqual(fit.reduced_chi2, 1.7088858647141, places=2)
        self.assertAlmostEqual(fit.amplitude, 0.078900464260488, places=2)
        self.assertAlmostEqual(fit.x_0, -0.68834523633912, places=2)
        self.assertAlmostEqual(fit.y_0, -0.069005412739451, places=2)
        self.assertAlmostEqual(fit.gamma, 8.0966823485900, places=2)
        self.assertAlmostEqual(fit.alpha, 16.048683662812, places=2)

    def test_stack_task_no_candidates(self):
        """Test that None returned when no candidates pass the radius check."""
        config = ExtendedPsfStackConfig()
        config.min_focal_plane_radius = 1e6  # Excludes all candidates.
        task = ExtendedPsfStackTask(config=config)
        with self.assertLogs(level=logging.INFO):
            result = task.run(extended_psf_candidates=self.extended_psf_candidates)
        self.assertIsNone(result)

    def test_subtract_task(self):
        """Test subtraction task on all synthetic stars.

        This test validates subtraction against known injected-star flux
        while exercising default background restore and re-estimation.
        """

        assert self.extended_psf is not None

        config = ExtendedPsfSubtractConfig()
        config.bad_mask_planes = []
        task = ExtendedPsfSubtractTask(config=config)

        preliminary_visit_image = self.exposure.clone()
        preliminary_visit_image.image.array -= self.background_array
        restored_input_image = self.exposure.image.array.copy()

        star_table = self.extended_psf_candidate_table

        with unittest.mock.patch.object(task, "_get_subtraction_star_table", return_value=star_table):
            result = task.run(
                preliminary_visit_image=preliminary_visit_image,
                preliminary_visit_image_background=self.background,
                extended_psf=self.extended_psf,
                ref_obj_loader=object(),
            )

        output_exposure = result.preliminary_visit_image_extended_psf_subtracted
        output_background = result.preliminary_visit_image_extended_psf_subtracted_background
        self.assertIsNotNone(output_background)

        restored_output_image = output_exposure.image.array + output_background.getImage().array
        removed_image = restored_input_image - restored_output_image
        expected_flux = float(np.sum(self.injected_stars_array))
        removed_flux = float(np.sum(removed_image))
        self.assertFloatsAlmostEqual(removed_flux, expected_flux, rtol=0.25)

        metadata = output_exposure.getMetadata()
        n_stars = len(star_table)
        self.assertEqual(metadata.getScalar("EPSFSUB_ATTEMPTED"), n_stars)
        self.assertEqual(metadata.getScalar("EPSFSUB_SUBTRACTED"), n_stars)


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
