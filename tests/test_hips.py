# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""Test HIPS code."""
import unittest
import cv2
import numpy as np
import hpgeom as hpg

import lsst.utils.tests
import lsst.pipe.base
import lsst.afw.image
import lsst.skymap
import lsst.geom

from lsst.pipe.tasks.hips import (
    HighResolutionHipsTask,
    HighResolutionHipsConfig,
    HighResolutionHipsConnections
)


class HipsTestCase(unittest.TestCase):
    def test_hips_single(self):
        """Test creating a single HIPS image."""
        np.random.seed(12345)

        config = HighResolutionHipsConfig()

        skymap = self._make_skymap()

        tract = 9597
        patch = 50
        tract_info = skymap[tract]
        patch_info = tract_info[patch]

        exposure = self._make_noise_exposure(patch_info)
        handles = [lsst.pipe.base.InMemoryDatasetHandle(exposure)]

        center = patch_info.wcs.pixelToSky(patch_info.inner_bbox.getCenter())
        pixel = self._get_pixel(2**config.hips_order, center)

        hips_task = HighResolutionHipsTask(config=config)
        output = hips_task.run([pixel], handles)

        # Check that all the pixels are filled.
        npix = (np.isfinite(output.hips_exposures[pixel].image.array.ravel()).sum())
        self.assertEqual(npix, output.hips_exposures[pixel].image.array.size)

        # Check that metadata is correct
        self.assertEqual(output.hips_exposures[pixel].getPhotoCalib(), exposure.getPhotoCalib())
        self.assertEqual(output.hips_exposures[pixel].getFilter(), exposure.getFilter())

    def test_hips_double(self):
        """Test creating a HIPS image from two neighboring patches."""
        np.random.seed(12345)

        config = HighResolutionHipsConfig()

        skymap = self._make_skymap()

        tract = 9597
        patches = [50, 51]
        tract_info = skymap[tract]

        handles = []
        centers = []
        for patch in patches:
            patch_info = tract_info[patch]
            exposure = self._make_noise_exposure(patch_info)
            handles.append(lsst.pipe.base.InMemoryDatasetHandle(exposure))
            centers.append(patch_info.wcs.pixelToSky(patch_info.inner_bbox.getCenter()))

        center = lsst.geom.SpherePoint(
            (centers[0].getRa().asDegrees() + centers[1].getRa().asDegrees())/2.*lsst.geom.degrees,
            (centers[0].getDec().asDegrees() + centers[1].getDec().asDegrees())/2.*lsst.geom.degrees
        )
        pixel = self._get_pixel(2**config.hips_order, center)

        # Just transform one, make sure it falls off the edge.
        hips_task = HighResolutionHipsTask(config=config)
        output = hips_task.run([pixel], [handles[0]])

        # Check that not all the pixels are filled.
        npix = (np.isfinite(output.hips_exposures[pixel].image.array.ravel()).sum())
        self.assertLess(npix, output.hips_exposures[pixel].image.array.size)

        # Transform both.
        hips_task = HighResolutionHipsTask(config=config)
        output = hips_task.run([pixel], handles)

        # Check that all the pixels are filled.
        npix = (np.isfinite(output.hips_exposures[pixel].image.array.ravel()).sum())
        self.assertEqual(npix, output.hips_exposures[pixel].image.array.size)

    def test_hips_none(self):
        """Test making a HIPS image with no overlapping inputs."""
        np.random.seed(12345)

        config = HighResolutionHipsConfig()

        skymap = self._make_skymap()

        tract = 9597
        patch = 50
        tract_info = skymap[tract]
        patch_info = tract_info[patch]

        exposure = self._make_noise_exposure(patch_info)
        handles = [lsst.pipe.base.InMemoryDatasetHandle(exposure)]

        pixel = 0

        hips_task = HighResolutionHipsTask(config=config)
        output = hips_task.run([pixel], handles)

        # Check that there is no returned image
        self.assertEqual(len(output.hips_exposures), 0)

    def test_hips_connections(self):
        """Test that the HIPS connections validate properly."""
        config = HighResolutionHipsConfig()

        # Test that the connections validate
        _ = HighResolutionHipsConnections(config=config)

        # Test that changing hips_order will break things because of the
        # dimensions mismatch.
        config.hips_order = 5
        with self.assertRaises(ValueError):
            _ = HighResolutionHipsConnections(config=config)

        # I'd like to change the dimensions but I don't know how to do that.

    def _make_noise_exposure(self, patch_info):
        """Make a simple noise exposure.

        Parameters
        ----------
        patch_info : `lsst.skymap.PatchInfo`
            Patch info to use to make the exposure.

        Returns
        -------
        exposure : `lsst.afw.image.ExposureF`
            Noise exposure.
        """
        exposure = lsst.afw.image.ExposureF(patch_info.outer_bbox)
        exposure.image.array[:, :] = np.random.normal(scale=1.0, size=exposure.image.array.shape)
        exposure.setWcs(patch_info.wcs)
        exposure.setPhotoCalib(lsst.afw.image.PhotoCalib(calibrationMean=1.0))
        exposure.setFilter(lsst.afw.image.FilterLabel(band='i'))

        return exposure

    def _make_skymap(self):
        """Make a testing skymap.

        Returns
        -------
        skymap : `lsst.skymap.RingsSkyMap`
        """

        # Generate a skymap
        skymap_config = lsst.skymap.ringsSkyMap.RingsSkyMapConfig()
        skymap_config.numRings = 120
        skymap_config.projection = "TAN"
        skymap_config.tractOverlap = 1.0/60
        skymap_config.pixelScale = 0.168
        return lsst.skymap.ringsSkyMap.RingsSkyMap(skymap_config)

    def _get_pixel(self, nside, sphpoint):
        """Get the pixel value from a spherepoint.

        Parameters
        ----------
        nside : `int`
            Healpix nside
        sphpoint : `lsst.geom.SpherePoint`
            Point to compute pixel value.

        Returns
        -------
        pixel : `int`
            Healpix pixel (nest ordering)
        """
        pixel = hpg.angle_to_pixel(
            nside,
            sphpoint.getRa().asDegrees(),
            sphpoint.getDec().asDegrees(),
        )
        return pixel


if __name__ == "__main__":
    unittest.main()


class ColorHipsTestCase(unittest.TestCase):
    def test_color_hips_config_builds(self):
        """Regression: ``GenerateColorHipsConfig.setDefaults()`` configured
        functor fields that the stellaRGB port removed (``Q``, per-action
        ``do_local_contrast``), so constructing it raised AttributeError."""
        from lsst.pipe.tasks.hips import GenerateColorHipsConfig

        config = GenerateColorHipsConfig()
        scene = config.rgbGenerator.scene
        self.assertEqual((scene.input_whitepoint.x, scene.input_whitepoint.y), (0.28, 0.28))
        self.assertEqual(config.rgbGenerator.luminance_config.stretch, 250)

    def test_high_order_hips_rejects_hdr(self):
        """HDR (is_hdr) ColorImages would be silently clipped to [0, 1] by the
        HiPS tiling; the task must refuse them outright instead."""
        from lsst.pipe.base import InMemoryDatasetHandle
        from lsst.images import ColorImage
        from lsst.geom import Box2I
        from lsst.pipe.tasks.rgb2hips._high_order_hips import HighOrderHipsTask

        config = HighOrderHipsTask.ConfigClass()
        config.hips_base_uri = "/tmp/hips_hdr_guard_test"
        config.color_ordering = "gri"
        task = HighOrderHipsTask(config=config)
        image = ColorImage(np.zeros((4, 4, 3), dtype=np.float32), metadata={"is_hdr": True})
        handle = InMemoryDatasetHandle(inMemoryDataset=image)
        with self.assertRaises(ValueError):
            task._assemble_sub_region({1: [(handle, None, Box2I())]}, 0)


class ColorHipsWarpBackendTestCase(unittest.TestCase):
    """Equivalence of the "afw" (afwMath Warper, reference) and "cv2"
    (precomputed-map cv2.remap) warp backends of HighOrderHipsTask."""

    def _make_config(self, backend, base_uri):
        from lsst.pipe.tasks.rgb2hips._high_order_hips import HighOrderHipsTask

        config = HighOrderHipsTask.ConfigClass()
        config.hips_base_uri = base_uri
        config.color_ordering = "gri"
        config.warp_backend = backend
        return config

    def _make_discrete_skymap(self):
        """Same two-tract discrete skymap used by test_meta_mosaic (patches
        of 640 px on a 400 px pitch at (ra, dec) ~ (10, -30))."""
        config = lsst.skymap.DiscreteSkyMap.ConfigClass()
        tract_builder = config.tractBuilder["cells"]
        tract_builder.cellInnerDimensions = [100, 100]
        tract_builder.cellBorder = 20
        tract_builder.numCellsPerPatchInner = 4
        tract_builder.numCellsInPatchBorder = 1
        config.tractBuilder.name = "cells"
        config.raList = [10.0, 10.05]
        config.decList = [-30.0, -30.0]
        config.radiusList = [0.15, 0.15]
        return lsst.skymap.DiscreteSkyMap(config)

    def _make_inputs(self, skymap):
        """Two constant-color patch images from the same tract (adjacent,
        overlapping), as run() receives them: (ndarray, SkyWcs, Box2I)."""
        inputs = []
        for patch in (0, 1):
            info = skymap[0][patch]
            bbox = info.getOuterBBox()
            array = np.full((bbox.getHeight(), bbox.getWidth(), 3), 0.5, dtype=np.float32)
            inputs.append((array, info.getWcs(), bbox))
        return inputs

    def test_warp_backend_choicefield_defaults_to_afw(self):
        config = self._make_config("afw", "/tmp/hips_backend_default_test")
        self.assertEqual(config.warp_backend, "afw")
        allowed = {k for k in type(config).warp_backend.allowed if k is not None}
        self.assertEqual(allowed, {"afw", "cv2"})

    def test_high_order_hips_cv2_matches_afw(self):
        """Both backends must produce the same persisted order-8 tile (the
        returned 256x256 output_hpx) within 5e-3 on all pixels not near a
        warped footprint edge.

        The cv2 backend uses Lanczos4 while the afw config uses lanczos5, so
        the thin ring around each footprint (different NaN-kernel bands and
        kernel truncation) plus its smear through the 16x Lanczos downsampling
        to 256 is excluded; the interior constant plateau must match, and it
        is the majority of the tile.
        """
        import tempfile
        from unittest import mock

        from lsst.afw.geom import makeHpxWcs
        from lsst.geom import Box2I, Extent2I, Point2I
        from lsst.pipe.tasks.prettyPictureMaker._utils import warp_maps
        from lsst.pipe.tasks.rgb2hips._high_order_hips import HighOrderHipsTask

        skymap = self._make_discrete_skymap()
        inputs = self._make_inputs(skymap)

        # The healpix order-8 tile containing the input patches. The task's
        # makeHpxWcs(8, id, 12, False) takes (order, pixel, shift_order,
        # validate_pixel_id) -- sphgeom/afw use NESTED ordering throughout.
        info = skymap[0][0]
        center = info.getWcs().pixelToSky(info.getOuterBBox().getCenter())
        healpix_id = hpg.angle_to_pixel(
            256, center.getRa().asDegrees(), center.getDec().asDegrees(), nest=True
        )

        outputs = {}
        with tempfile.TemporaryDirectory() as tmp:
            for backend in ("afw", "cv2"):
                config = self._make_config(backend, tmp)
                task = HighOrderHipsTask(config=config)
                with mock.patch(
                    "lsst.pipe.tasks.rgb2hips._high_order_hips._write_hips_image"
                ):
                    outputs[backend] = task.run(list(inputs), healpix_id).output_hpx

        a, b = outputs["afw"], outputs["cv2"]
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))

        # Build the exclusion mask: ring around each footprint boundary,
        # dilated for the 16x block + Lanczos smear of the final resize.
        target_wcs = makeHpxWcs(8, healpix_id, 12, False)
        exp_bbox = Box2I(corner=Point2I(0, 0), dimensions=Extent2I(2**12, 2**12))
        bad = np.zeros((4096, 4096), dtype=bool)
        probe = HighOrderHipsTask(config=self._make_config("afw", tmp))
        for array, in_wcs, in_box in inputs:
            db = probe._footprint_bbox(in_wcs, in_box, target_wcs, exp_bbox)
            if db.isEmpty():
                continue
            map_x, map_y = warp_maps(in_wcs, in_box, target_wcs, db)
            margin = 10
            dbad = (
                ~np.isfinite(map_x)
                | ~np.isfinite(map_y)
                | (map_x < margin)
                | (map_x > in_box.getWidth() - 1 - margin)
                | (map_y < margin)
                | (map_y > in_box.getHeight() - 1 - margin)
            )
            bad[db.getMinY(): db.getMaxY() + 1, db.getMinX(): db.getMaxX() + 1] = dbad
        # Dilate by the final 4096->256 Lanczos resize support (~8 source px
        # per dest px = 128 px) plus the 16 px block size, so the compared
        # 256-resolution pixels are pure interior plateau.
        bad = cv2.dilate(bad.astype(np.uint8), np.ones((300, 300), np.uint8)).astype(bool)
        # run() y-flips the super-tile before resampling to 256: map the
        # (pre-flip) mask into output row orientation as well.
        good = ~bad[::-1, ::][::16, ::16]  # 256x256 block sampling of the zoomed output

        self.assertGreater(good.mean(), 0.5, "mask must keep most of the tile")
        # Sanity: data actually landed on the tile (the mask keeps a
        # substantial interior footprint region for both backends).
        plateau_a = a[good][a[good] > 0]
        plateau_b = b[good][b[good] > 0]
        self.assertGreater(plateau_a.size, 1000, "afw backend deposited no data")
        self.assertGreater(plateau_b.size, 1000, "cv2 backend deposited no data")
        np.testing.assert_allclose(a[good], b[good], atol=5e-3)
        # The constant inputs produce a non-trivial constant plateau.
        self.assertGreater(np.median(plateau_a), 0.0)
        # and it covers a comparable area in both backends.
        self.assertLess(abs(plateau_a.size - plateau_b.size) / plateau_a.size, 0.05)
