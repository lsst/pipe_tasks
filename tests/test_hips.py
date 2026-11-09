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
    """Equivalence of the "afw" (afwMath Warper, reference) and "rust"
    (stellaRGB.warp engine) warp backends of HighOrderHipsTask."""

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

    def test_warp_backend_choicefield_defaults_to_rust(self):
        """Default-pin (T6.3b): the ChoiceField default is "rust" (flipped
        from "afw" on 2026-10-29, owner election after the acceptance rerun
        validated the rust path end-to-end); "afw" stays a valid fallback.
        (Builds the config directly: the old test's _make_config("afw", ...)
        overrode the field, so it never actually read the default.)"""
        from lsst.pipe.tasks.rgb2hips._high_order_hips import HighOrderHipsTask

        config = HighOrderHipsTask.ConfigClass()
        self.assertEqual(config.warp_backend, "rust")
        allowed = {k for k in type(config).warp_backend.allowed if k is not None}
        self.assertEqual(allowed, {"afw", "rust"})

    def _run_backend(self, backend, inputs, healpix_id, tmp):
        """Run the task on the given inputs with one backend (tile writing
        mocked; returns the persisted 256x256 output_hpx raster)."""
        from unittest import mock

        from lsst.pipe.tasks.rgb2hips._high_order_hips import HighOrderHipsTask

        config = self._make_config(backend, tmp)
        task = HighOrderHipsTask(config=config)
        with mock.patch("lsst.pipe.tasks.rgb2hips._high_order_hips._write_hips_image"):
            return task.run(list(inputs), healpix_id).output_hpx

    def _interior_mask(self, inputs, healpix_id, tmp):
        """Pixels of the 256x256 output where backends must agree: ring
        around each footprint boundary (source-edge proximity from the
        warp_maps probe), dilated for the 4096->256 box-average support plus
        the 16 px block size, mapped through run()'s y-flip."""
        from lsst.afw.geom import makeHpxWcs
        from lsst.geom import Box2I, Extent2I, Point2I
        from lsst.pipe.tasks.prettyPictureMaker._utils import warp_maps
        from lsst.pipe.tasks.rgb2hips._high_order_hips import HighOrderHipsTask

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
        bad = cv2.dilate(bad.astype(np.uint8), np.ones((300, 300), np.uint8)).astype(bool)
        # run() y-flips the super-tile before downsampling to 256: map the
        # (pre-flip) mask into output row orientation as well.
        return ~bad[::-1, ::][::16, ::16]  # 256x256 block sampling

    def _healpix_id(self, skymap):
        """The healpix order-8 tile containing the input patches. The task's
        makeHpxWcs(8, id, 12, False) takes (order, pixel, shift_order,
        validate_pixel_id) -- sphgeom/afw use NESTED ordering throughout."""
        info = skymap[0][0]
        center = info.getWcs().pixelToSky(info.getOuterBBox().getCenter())
        return hpg.angle_to_pixel(
            256, center.getRa().asDegrees(), center.getDec().asDegrees(), nest=True
        )

    def _assert_backend_matches_afw(self, backend):
        """The given backend must produce the same persisted order-8 tile
        (the returned 256x256 output_hpx) within 5e-3 on all pixels not near
        a warped footprint edge.

        The afw config uses lanczos5 while rust uses Lanczos4 (rust
        renders through fit-verified control-grid maps with residual
        <= 0.05 source px), so the thin ring around each footprint
        (different NaN-kernel bands and kernel truncation) plus its smear
        through the 16x downsampling to 256 is excluded; the interior
        constant plateau must match, and it is the majority of the tile.
        Measured (this harness): rust masked max-diff 2.4e-6 -- the 5e-3
        tolerance is 2000x margin for rust, not a loosening.
        """
        import tempfile

        skymap = self._make_discrete_skymap()
        inputs = self._make_inputs(skymap)
        healpix_id = self._healpix_id(skymap)

        with tempfile.TemporaryDirectory() as tmp:
            a = self._run_backend("afw", inputs, healpix_id, tmp)
            b = self._run_backend(backend, inputs, healpix_id, tmp)
            good = self._interior_mask(inputs, healpix_id, tmp)

        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
        self.assertGreater(good.mean(), 0.5, "mask must keep most of the tile")
        # Sanity: data actually landed on the tile (the mask keeps a
        # substantial interior footprint region for both backends).
        plateau_a = a[good][a[good] > 0]
        plateau_b = b[good][b[good] > 0]
        self.assertGreater(plateau_a.size, 1000, "afw backend deposited no data")
        self.assertGreater(plateau_b.size, 1000, f"{backend} backend deposited no data")
        np.testing.assert_allclose(a[good], b[good], atol=5e-3)
        # The constant inputs produce a non-trivial constant plateau.
        self.assertGreater(np.median(plateau_a), 0.0)
        # and it covers a comparable area in both backends.
        self.assertLess(abs(plateau_a.size - plateau_b.size) / plateau_a.size, 0.05)

    def test_high_order_hips_rust_matches_afw(self):
        self._assert_backend_matches_afw("rust")

    def test_high_order_hips_rust_transform_cache(self):
        """One fit_transform per (image, tile): two patches -> 2 fits on the
        first run, 0 additional fits when the SAME task instance is re-run
        (the retry scenario the memo exists for). Curvilinear HPX tiles make
        the fit correctly return maps mode with a verified residual."""
        import tempfile
        from unittest import mock

        from lsst.pipe.tasks.rgb2hips import _high_order_hips
        from lsst.pipe.tasks.rgb2hips._high_order_hips import HighOrderHipsTask

        skymap = self._make_discrete_skymap()
        inputs = self._make_inputs(skymap)
        healpix_id = self._healpix_id(skymap)
        fits = []
        real = _high_order_hips.fit_transform

        def spy(*args, **kwargs):
            transform = real(*args, **kwargs)
            fits.append(transform)
            return transform

        with tempfile.TemporaryDirectory() as tmp:
            # ONE task instance across both runs -- the cache is the task's
            # (a butler retry reuses the task; a fresh task legitimately refits).
            task = HighOrderHipsTask(config=self._make_config("rust", tmp))
            with mock.patch.object(_high_order_hips, "fit_transform", spy), mock.patch(
                "lsst.pipe.tasks.rgb2hips._high_order_hips._write_hips_image"
            ):
                task.run(list(inputs), healpix_id)
                self.assertEqual(len(fits), 2, "expected one fit per (image, tile)")
                self.assertEqual([t.mode for t in fits], ["maps", "maps"])
                for t in fits:
                    self.assertLessEqual(t.max_residual, 0.05)
                    self.assertIsNotNone(t.dest_bbox)
                task.run(list(inputs), healpix_id)
                self.assertEqual(len(fits), 2, "second run must reuse the transform cache")

    def test_high_order_hips_fit_failure_falls_back_to_afw(self):
        """A failing fit is a per-image robustness event: one warning per
        image, the failed fit cached (no retry on the second run), and the
        tile is served from the afw Warper -- the output is then byte-for-byte
        the afw backend's output and map-build never crashed the run."""
        import tempfile
        from unittest import mock

        from lsst.pipe.tasks.rgb2hips import _high_order_hips
        from lsst.pipe.tasks.rgb2hips._high_order_hips import HighOrderHipsTask

        skymap = self._make_discrete_skymap()
        inputs = self._make_inputs(skymap)
        healpix_id = self._healpix_id(skymap)
        calls = []

        def broken_fit(*args, **kwargs):
            calls.append(args)
            raise ValueError("noninvertible/off-domain projection pair (injected)")

        with tempfile.TemporaryDirectory() as tmp:
            config = self._make_config("rust", tmp)
            task = HighOrderHipsTask(config=config)
            warned = []
            with mock.patch.object(_high_order_hips, "fit_transform", broken_fit), mock.patch(
                "lsst.pipe.tasks.rgb2hips._high_order_hips._write_hips_image"
            ), mock.patch.object(task.log, "warning", side_effect=lambda *a: warned.append(a)):
                rust_out = task.run(list(inputs), healpix_id).output_hpx
                task.run(list(inputs), healpix_id)  # retry: cache suppresses re-fit
            self.assertEqual(len(calls), 2, "one fit attempt per image, cached after")
            self.assertEqual(len(warned), 2, "one warning per image")
            self.assertIn("falling back to the afw Warper", warned[0][0])
            afw_out = self._run_backend("afw", inputs, healpix_id, tmp)
            np.testing.assert_array_equal(np.asarray(rust_out), np.asarray(afw_out))


class _TanPatchInfo:
    """Patch-info wrapper reporting the tract's replaced WCS (see
    _TanSkyMap)."""

    def __init__(self, patch_info, wcs):
        self._info = patch_info
        self._wcs = wcs

    def getWcs(self):
        return self._wcs

    def __getattr__(self, name):
        return getattr(self._info, name)


class _TanTractInfo:
    """Tract-info wrapper replacing the WCS reported for the tract and
    every patch in it."""

    def __init__(self, tract_info, wcs):
        self._info = tract_info
        self._wcs = wcs

    def getWcs(self):
        return self._wcs

    def __getitem__(self, patch):
        return _TanPatchInfo(self._info[patch], self._wcs)

    def __getattr__(self, name):
        return getattr(self._info, name)


class _TanSkyMap:
    """SkyMap wrapper handing the task production-shape WCSs: each tract's
    WCS (and its patches' WCSs) is the strict FITS TAN rebuilt from its own
    (pixelOrigin, skyOrigin, CD) triple.

    Runtime probe for T8.6: the raw DiscreteSkyMap fixture reports
    CTYPE1/2 = RA---STG/DEC--STG (legacy frame dict), which the shared
    pure-FITS-TAN gate provably declines -- so every raw-fixture "rust"
    test above genuinely exercises the fit/maps fallback. A production
    post-acalexn patch WCS is a plain TAN FITS header afw loads as exactly
    this reconstruction, which is what drives the direct gate to accept
    (same wrapper idiom as test_meta_mosaic's _TanSkymap, kept local
    because rgb2hips only needs the (getWcs, getOuterBBox) surface).
    """

    def __init__(self, skymap):
        from lsst.afw.geom import makeSkyWcs

        self._skymap = skymap
        self._cache = {}
        self._makeSkyWcs = makeSkyWcs

    def __getitem__(self, tract):
        if tract not in self._cache:
            wcs = self._skymap[tract].getWcs()
            tan = self._makeSkyWcs(
                crpix=wcs.getPixelOrigin(),
                crval=wcs.getSkyOrigin(),
                cdMatrix=np.asarray(wcs.getCdMatrix()),
            )
            self._cache[tract] = _TanTractInfo(self._skymap[tract], tan)
        return self._cache[tract]


class ColorHipsRustDirectTestCase(unittest.TestCase):
    """T8.6 direct per-pixel HPX warp mode for HighOrderHipsTask: the rust
    backend prefers a closed-form DirectWarp (tile WCS through the shared
    HiPS-header gate, source through the shared pure-FITS-TAN gate) over
    the T6.2 fit/maps ladder. Pins: direct-vs-afw parity at the battery's
    atol with a positional hot spot (F2 lesson: constant plateaus are
    translation-blind, so non-vacuity is proven by injecting a +2 px source
    crpix shift and demanding bad pixels), zero fit calls with "hpx-direct"
    logs and Timings map-build n=0 for a full-direct run, and the untouched
    maps ladder (fits==1, "rust-maps" line, map-build n=1) for a source
    carrying real SIP cards."""

    def setUp(self):
        # Reuse the backend-equivalence harness (config builder, tile-writing
        # mock, interior mask, healpix-id lookup) without inheriting (and
        # re-running) its tests.
        self.harness = ColorHipsWarpBackendTestCase()

    def _make_hot_inputs(self, skymap, healpix_id, tmp):
        """One tract-0 patch image as run() receives it, carrying a smooth
        Gaussian hot spot (R+G bump, sigma=15 px, amp=2.0 over the 0.5
        base).

        The stamp position is self-calibrated, not hardcoded: the
        single-image backend-agreement mask (the diamond footprint minus
        its edge/NaN rings and the 150 px dilation band) is eroded by ±7
        blocks and the blob is placed at the source pixel that lands at
        the center of the deepest surviving 16x16 output block, so the
        bump AND its whole shifted-difference halo stay inside the
        compared region (probed lesson: the patch center itself sits
        ~127 px from the tile edge -- inside the dilation band -- and the
        overlapping second patch's ring leaves no room for a deep stamp in
        the two-image fixture, which is why this battery warps one image).
        The smooth bump makes the comparison misalignment-sensitive rather
        than translation-blind (the meta-mosaic F2 lesson)."""
        import cv2
        from scipy.ndimage import distance_transform_edt
        from lsst.afw.geom import makeHpxWcs
        from lsst.geom import Point2D

        info = skymap[0][0]
        bbox = info.getOuterBBox()
        height, width = bbox.getHeight(), bbox.getWidth()
        flat = [
            (np.full((height, width, 3), 0.5, dtype=np.float32), info.getWcs(), bbox)
        ]
        good = self.harness._interior_mask(flat, healpix_id, tmp)
        raster = self.harness._run_backend("afw", flat, healpix_id, tmp)
        inside = good & (raster[..., 0] > 0.05)
        self.assertGreater(inside.sum(), 1000, "fixture deposits no compared data")
        eroded = cv2.erode(inside.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        self.assertTrue(eroded.any(), "no deeply interior region in the agreement mask")
        dist = distance_transform_edt(eroded)
        row, col = np.unravel_index(np.argmax(dist), dist.shape)
        # run() flips the super-tile rows before the 16x block sampling:
        # map the block back to a pre-flip tile pixel.
        tile_point = Point2D(float(col * 16 + 8), float(4095 - (row * 16 + 8)))
        target = makeHpxWcs(8, healpix_id, 12, False)
        src = info.getWcs().skyToPixel(target.pixelToSky(tile_point))
        cx = src.getX() - bbox.getMinX()
        cy = src.getY() - bbox.getMinY()
        self.assertTrue(
            60 < cx < width - 60 and 60 < cy < height - 60,
            f"hot spot stamp ({cx:.0f}, {cy:.0f}) not interior to the "
            f"{width}x{height} patch",
        )
        array = flat[0][0]
        ys, xs = np.mgrid[0:height, 0:width]
        r2 = (xs - (cx - 0.5)) ** 2 + (ys - (cy - 0.5)) ** 2
        blob = (2.0 * np.exp(-r2 / (2.0 * 15.0 * 15.0))).astype(np.float32)
        array[..., :2] += blob[..., np.newaxis]
        return [(array, info.getWcs(), bbox)]

    def _run_rust_shifted_src(self, inputs, healpix_id, tmp, shift_px=2.0):
        """Run the rust backend with the shared TAN builder stubbed to
        inject +shift_px into the SOURCE crpix (patched in the CONSUMER
        namespace): if direct mode really renders, the hot spot lands
        shifted; if anything silently fell back to maps, it would not."""
        from unittest import mock

        from lsst.pipe.tasks import _warpDirect
        from lsst.pipe.tasks.rgb2hips import _high_order_hips
        from stellaRGB.warp import DirectWcs

        real = _warpDirect.tan_wcs_from_afw

        def shifted(wcs):
            t = real(wcs)
            return DirectWcs(
                crpix=(t.crpix[0] + shift_px, t.crpix[1]),
                crval_deg=t.crval_deg,
                cd_deg_per_pix=t.cd_deg_per_pix,
            )

        with mock.patch.object(_high_order_hips, "tan_wcs_from_afw", shifted):
            return self.harness._run_backend("rust", inputs, healpix_id, tmp)

    def test_high_order_hips_rust_direct_matches_afw_with_hot_spot(self):
        """Direct-mode twin of the backend battery: with production-shape
        TAN sources onto the HPX tile the direct engine (no fit, no maps)
        reproduces the afwMath reference at the same atol=5e-3 on the
        interior mask; the stamped hot spot must be visible in both (its
        peak survives the 4096->256 area pyramid, so any >=1 px positional
        error moves real gradient into compared pixels -- the non-vacuity
        leg is the sibling test below)."""
        import tempfile

        skymap = _TanSkyMap(self.harness._make_discrete_skymap())
        healpix_id = self.harness._healpix_id(skymap)
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self._make_hot_inputs(skymap, healpix_id, tmp)
            a = self.harness._run_backend("afw", inputs, healpix_id, tmp)
            b = self.harness._run_backend("rust", inputs, healpix_id, tmp)
            good = self.harness._interior_mask(inputs, healpix_id, tmp)
        self.assertTrue(np.all(np.isfinite(a)) and np.all(np.isfinite(b)))
        self.assertGreater(good.mean(), 0.5, "mask must keep most of the tile")
        # The hot spot is present inside the kept data region of BOTH
        # backends: green rises well above the data-region median (the
        # warp's pixel-area scaling puts all values below input scale).
        data = a[..., 0] > 0.09  # the (scaled) 0.5 plateau is ~0.18
        self.assertGreater((good & data).sum(), 1000)
        for arr, name in ((a, "afw"), (b, "rust")):
            green = arr[good & data][..., 1]
            self.assertGreater(
                green.max() - np.median(green), 0.15, f"hot spot lost in {name}"
            )
        np.testing.assert_allclose(a[good], b[good], atol=5e-3)

    def test_high_order_hips_rust_direct_hot_spot_detects_shift(self):
        """F2 non-vacuity pin: with a +2 px source-crpix shift injected into
        the shared TAN builder, the hot spot is misregistered and the
        masked comparison MUST breach the 5e-3 battery gate (bad pixels > 0)
        -- proving the parity test above is positional, not translation-
        blind, and that the compared pixels really come from the direct
        engine (a maps-fallback render would be shift-blind at the
        residual tolerance)."""
        import tempfile

        skymap = _TanSkyMap(self.harness._make_discrete_skymap())
        healpix_id = self.harness._healpix_id(skymap)
        with tempfile.TemporaryDirectory() as tmp:
            inputs = self._make_hot_inputs(skymap, healpix_id, tmp)
            a = self.harness._run_backend("afw", inputs, healpix_id, tmp)
            b = self._run_rust_shifted_src(inputs, healpix_id, tmp)
            good = self.harness._interior_mask(inputs, healpix_id, tmp)
        diff = np.abs(a[good] - b[good])
        self.assertGreater(float(diff.max()), 5e-3, "injected shift went undetected")
        self.assertGreater(int((diff > 5e-3).sum()), 0, "injected shift went undetected")

    def test_high_order_hips_rust_direct_zero_fits_and_logs(self):
        """Zero-fit pin: a full-direct run (both sources pure TAN onto the
        plain HPX tile) never touches the CONSUMER-namespace maps-building
        symbol, logs one "hpx-direct" INFO line per (source, tile) class
        with no "rust-maps" line, and reports a parseable Timings line with
        map-build n=0 while both images still count in warp."""
        import logging
        import re
        import tempfile
        from unittest import mock

        from lsst.pipe.tasks.rgb2hips import _high_order_hips

        skymap = _TanSkyMap(self.harness._make_discrete_skymap())
        healpix_id = self.harness._healpix_id(skymap)
        fits = []
        real_fit = _high_order_hips.fit_transform

        def spy(*args, **kwargs):
            transform = real_fit(*args, **kwargs)
            fits.append(transform)
            return transform

        with tempfile.TemporaryDirectory() as tmp:
            # Plain constant-color two-image inputs (the hot spot is only
            # needed by the positional battery; mode selection and its
            # observability are WCS-driven).
            inputs = self.harness._make_inputs(skymap)
            with mock.patch.object(_high_order_hips, "fit_transform", spy), \
                    self.assertLogs("lsst.highOrderHipsTask", level=logging.INFO) as cm:
                self.harness._run_backend("rust", inputs, healpix_id, tmp)
        self.assertEqual(fits, [], "direct mode must never build maps")
        direct = [m for m in cm.output if "hpx-direct" in m]
        self.assertEqual(len(direct), 2, "one hpx-direct line per (image, tile)")
        self.assertIn("dest box", direct[0])
        self.assertFalse(any("rust-maps" in m for m in cm.output))
        line = [m for m in cm.output if "Timings:" in m][-1]
        self.assertIsNotNone(re.search(r"warp=[0-9.]+s \(n=2\)", line), line)
        match = re.search(r"map-build=[0-9.]+s \(n=(\d+)\)", line)
        self.assertIsNotNone(match, line)
        self.assertEqual(int(match.group(1)), 0)

    def test_high_order_hips_rust_maps_fallback_on_sip_source(self):
        """Maps-fallback pin: a source carrying real SIP cards (afw
        makeTanSipWcs -- isFits True, so only the exact card scan can
        reject it; the probe-pure HPX tile still builds its direct dest) is
        declined by the shared gate and served by the untouched T6.2 ladder:
        exactly one fit, one "rust-maps" spacing/residual line, no
        "hpx-direct" line, Timings map-build n=1."""
        import logging
        import re
        import tempfile
        from unittest import mock

        from lsst.afw.geom import makeTanSipWcs
        from lsst.pipe.tasks.rgb2hips import _high_order_hips

        skymap = self.harness._make_discrete_skymap()
        info = skymap[0][0]
        bbox = info.getOuterBBox()
        w = info.getWcs()
        sip_wcs = makeTanSipWcs(
            w.getPixelOrigin(),
            w.getSkyOrigin(),
            np.asarray(w.getCdMatrix()),
            np.array([[0.0, 0.0], [0.0, 1.0e-7]]),
            np.array([[0.0, 0.0], [0.0, -1.0e-7]]),
        )
        # Precondition: the SIP cards are really in the header the gate
        # reads (isFits stays True -- cards are the only discriminator).
        self.assertTrue(sip_wcs.isFits)
        inputs = [(np.full((bbox.getHeight(), bbox.getWidth(), 3), 0.5, dtype=np.float32),
                   sip_wcs, bbox)]
        healpix_id = self.harness._healpix_id(skymap)
        fits = []
        real_fit = _high_order_hips.fit_transform

        def spy(*args, **kwargs):
            transform = real_fit(*args, **kwargs)
            fits.append(transform)
            return transform

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(_high_order_hips, "fit_transform", spy), \
                    self.assertLogs("lsst.highOrderHipsTask", level=logging.INFO) as cm:
                out = self.harness._run_backend("rust", inputs, healpix_id, tmp)
        self.assertEqual(len(fits), 1, "one maps fit for the declined source")
        self.assertEqual([t.mode for t in fits], ["maps"])
        maps = [m for m in cm.output if "rust-maps" in m]
        self.assertEqual(len(maps), 1)
        self.assertIn("tile", maps[0])
        self.assertIsNotNone(re.search(r"spacing=[0-9.]+ residual=[0-9.eE+-]+", maps[0]))
        self.assertFalse(any("hpx-direct" in m for m in cm.output))
        line = [m for m in cm.output if "Timings:" in m][-1]
        match = re.search(r"map-build=[0-9.]+s \(n=(\d+)\)", line)
        self.assertIsNotNone(match, line)
        self.assertEqual(int(match.group(1)), 1)
        self.assertTrue(np.all(np.isfinite(out)))


class ColorHipsBlendTestCase(unittest.TestCase):
    """The per-tile combine of HighOrderHipsTask: feather-weighted
    accumulation through the shared FeatheredMosaicCreator masked-add /
    NaN-reweight contract (what MetaMosaicTask runs), replacing the legacy
    first-valid-wins + unweighted 50/50 blend whose un-feathered hard edges
    left a rectangular grid of brightness steps on every contributor
    boundary of the rendered tile."""

    def setUp(self):
        # Reuse the warp-backend harness (config builder, tile-writing
        # mock, input builders) without inheriting (and re-running) its tests.
        self.harness = ColorHipsWarpBackendTestCase()

    def _make_wide_skymap(self):
        """The harness two-tract skymap widened to 8-cell-inner patches
        (pitch 800 px, outer 1040 px at the 0.2 "/px tract scale; the tile
        pixel is ~0.6x the tract one, so the union spans ~3000 of the
        order-8 tile's 4096 px): the contributor seam (2*cellBorder worth,
        ~400 tile px of overlap, of it) and both contributor rims land
        well inside the tile -- away from the feather ramp's span and from
        the edge rule at the tile border."""
        config = lsst.skymap.DiscreteSkyMap.ConfigClass()
        tract_builder = config.tractBuilder["cells"]
        tract_builder.cellInnerDimensions = [100, 100]
        tract_builder.cellBorder = 20
        tract_builder.numCellsPerPatchInner = 8
        tract_builder.numCellsInPatchBorder = 1
        config.tractBuilder.name = "cells"
        config.raList = [10.0, 10.05]
        config.decList = [-30.0, -30.0]
        config.radiusList = [0.2, 0.2]
        return lsst.skymap.DiscreteSkyMap(config)

    def _make_level_inputs(self, skymap, levels):
        """Constant-color patch images, as run() receives them:
        (ndarray, SkyWcs, Box2I), one entry per (patch, level) pair."""
        inputs = []
        for patch, level in levels:
            info = skymap[0][patch]
            bbox = info.getOuterBBox()
            array = np.full((bbox.getHeight(), bbox.getWidth(), 3), level, dtype=np.float32)
            inputs.append((array, info.getWcs(), bbox))
        return inputs

    def _union_healpix_id(self, skymap):
        """The order-8 tile containing the midpoint of the two patch
        centers, so the patch union sits near the tile center."""
        centers = [
            skymap[0][patch].getWcs().pixelToSky(skymap[0][patch].getOuterBBox().getCenter())
            for patch in (0, 1)
        ]
        ra = sum(c.getRa().asDegrees() for c in centers) / 2.0
        dec = sum(c.getDec().asDegrees() for c in centers) / 2.0
        return hpg.angle_to_pixel(256, ra, dec, nest=True)

    def test_contributor_seams_do_not_survive_combine(self):
        """Two contributors at different constant levels composite into a
        continuous raster: the maximum adjacent-pixel step inside the data
        interior must stay under a quarter of the plateau difference. The
        legacy blend produced a step of the full plateau difference at each
        edge of its unweighted 50/50 overlap band (ratio ~1.0 measured);
        feather-weighted accumulation spreads the same transition over ~30
        output columns (ratio ~0.03). No pixel may overshoot beyond the two
        contributor plateaus either."""
        import tempfile

        skymap = self._make_wide_skymap()
        inputs = self._make_level_inputs(skymap, ((0, 0.4), (1, 0.6)))
        healpix_id = self._union_healpix_id(skymap)
        with tempfile.TemporaryDirectory() as tmp:
            raster = self.harness._run_backend("rust", inputs, healpix_id, tmp)[..., 0]

        self.assertTrue(np.all(np.isfinite(raster)))
        # Empty-sky pixels are exactly 0 (finalize() NaN -> 0); any
        # positive value is contributor data.
        data = raster > 1e-3
        # Erode the union rim (its 0 -> plateau step is the data boundary,
        # not a seam) with margin over the feather transition
        # (2*feather_grow tile px = 16 output rows) and the warp's NaN ring.
        interior = cv2.erode(data.astype(np.uint8), np.ones((35, 35), np.uint8)).astype(bool)
        vals = raster[interior]
        self.assertGreater(vals.size, 1000, "eroded away the whole data interior")
        mid = (vals.min() + vals.max()) / 2.0
        p_lo = np.median(vals[vals < mid])
        p_hi = np.median(vals[vals > mid])
        plateau_diff = p_hi - p_lo
        self.assertGreater(
            plateau_diff, 0.2 * p_hi, "the two contributor plateaus did not separate"
        )
        self.assertGreaterEqual(vals.min(), p_lo - 0.02 * plateau_diff)
        self.assertLessEqual(vals.max(), p_hi + 0.02 * plateau_diff)
        horiz = np.abs(np.diff(raster, axis=1))
        ok_h = interior[:, :-1] & interior[:, 1:]
        vert = np.abs(np.diff(raster, axis=0))
        ok_v = interior[:-1, :] & interior[1:, :]
        max_step = max(horiz[ok_h].max(), vert[ok_v].max())
        self.assertLess(max_step, 0.25 * plateau_diff)

    def test_all_nan_contributor_is_skipped(self):
        """A contributor whose warp has no valid pixels is a no-op: it must
        deposit neither data nor weight, leaving the raster bit-identical
        to running without it (and finite everywhere)."""
        import tempfile

        skymap = self.harness._make_discrete_skymap()
        inputs = self.harness._make_inputs(skymap)
        only_good = [inputs[0]]
        with_nan = [
            inputs[0],
            (np.full_like(inputs[1][0], np.nan), inputs[1][1], inputs[1][2]),
        ]
        healpix_id = self.harness._healpix_id(skymap)
        with tempfile.TemporaryDirectory() as tmp:
            a = self.harness._run_backend("rust", only_good, healpix_id, tmp)
            b = self.harness._run_backend("rust", with_nan, healpix_id, tmp)
        self.assertTrue(np.all(np.isfinite(b)))
        np.testing.assert_array_equal(a, b)
        self.assertGreater(a.max(), 0.0, "the surviving contributor deposited no data")

    def test_thin_contributor_clamps_feather_ramp(self):
        """A contributor clipped to a ~100 tile px sliver is thinner than
        the feather ramp (2*feather_grow = 256 px): the ramp must clamp to
        the footprint instead of outgrowing the feather masks (unclamped,
        the mask build raises a broadcast error), the sliver must keep its
        constant level, and its edge truncated by the tile border must not
        ramp down (that data continues into the adjacent order-8 quantum).
        The shifted-bbox placement keeps the source WCS untouched: the box
        is where the image lives in that WCS' parent frame."""
        import tempfile

        from lsst.afw.geom import makeHpxWcs

        skymap = self.harness._make_discrete_skymap()
        healpix_id = self.harness._healpix_id(skymap)
        array, in_wcs, in_box = self.harness._make_inputs(skymap)[0]

        target = makeHpxWcs(8, healpix_id, 12, False)

        def tile_x(src_x):
            sky = in_wcs.pixelToSky(lsst.geom.Point2D(src_x, in_box.getCenterY()))
            return target.skyToPixel(sky).getX()

        w_tile = abs(tile_x(in_box.getMaxX()) - tile_x(in_box.getMinX()))
        inset = 100  # tile px of the patch left inside the tile's +x border
        dx_src = ((4096.0 - inset) - tile_x(in_box.getMinX())) * in_box.getWidth() / w_tile
        sliver_box = lsst.geom.Box2I(
            lsst.geom.Point2I(in_box.getMinX() + int(round(dx_src)), in_box.getMinY()),
            in_box.getDimensions(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            raster = self.harness._run_backend(
                "rust", [(array, in_wcs, sliver_box)], healpix_id, tmp
            )[..., 0]

        self.assertTrue(np.all(np.isfinite(raster)))
        level = raster.max()
        self.assertGreater(level, 0.0, "the sliver deposited no data")
        sliver = raster > 0.5 * level
        self.assertGreater(sliver.sum(), 100, "the sliver plateau collapsed")
        # The plateau reaches the tile border at full level (clipped edges
        # keep weight; a ramp here would fall below the 0.5*level threshold
        # before column 254).
        self.assertGreaterEqual(np.flatnonzero(sliver.any(axis=0))[-1], 254)
        self.assertLess(abs(np.median(raster[sliver]) - level), 0.05 * level)

    def test_assemble_sub_region_reweights_seam(self):
        """Patch-to-sub-region assembly (the per-tract mosaic that feeds
        run()) must be the same NaN-reweighted feather blend the
        meta-mosaic runs: without the weight accumulator the feathering
        degrades to an unweighted sum whose overlap bands overshoot
        (w_A + w_B != 1: legacy measured 0.68 for 0.4/0.6 contributors)
        and whose NaN borders leak into the mosaic instead of being
        reweighted away. This -- not the tract-to-tract combine -- is
        where contributor seams actually enter single-tract data."""
        from lsst.pipe.tasks.rgb2hips._high_order_hips import HighOrderHipsTask

        skymap = self.harness._make_discrete_skymap()
        task = HighOrderHipsTask(config=self.harness._make_config("afw", "/tmp"))

        class _Handle:
            """Stand-in for DeferredDatasetHandle delivering a bare array."""

            def __init__(self, array):
                self.array = array
                self.metadata = None

            def get(self):
                return self

        inputs = {0: []}
        for patch, level in ((0, 0.4), (1, 0.6)):
            info = skymap[0][patch]
            bbox = info.getOuterBBox()
            array = np.full((bbox.getHeight(), bbox.getWidth(), 3), level, dtype=np.float32)
            # NaN bands facing the overlap (both other-covered, so a
            # reweighted blend must leave no hole; the legacy unweighted
            # sum poisons them with NaN permanently).
            if patch == 0:
                array[:, -8:] = np.nan
            else:
                array[:, :8] = np.nan
            inputs[0].append((_Handle(array), info.getWcs(), bbox))
        sub, _, _ = task._assemble_sub_region(
            inputs, skymap[0][0].getCellInnerDimensions().getX()
        )[0]

        # The union rectangle is fully covered once NaN reweights away.
        self.assertTrue(np.all(np.isfinite(sub)), "NaN leaked through the seam")
        vals = sub[..., 0]
        # Levels stay inside the contributors' range: no overshoot band.
        self.assertLessEqual(vals.max(), 0.6 + 0.01, "overlap band overshoots")
        self.assertGreaterEqual(vals.min(), 0.4 - 0.01, "seam dips below plateaus")
        # The overlap genuinely blends (strictly intermediate values exist).
        self.assertTrue(np.any((vals > 0.41) & (vals < 0.59)), "no crossfade band")


class ColorHipsPyramidTestCase(unittest.TestCase):
    """The area-kernel (D10 box-average) pyramid prefilter: exactness,
    engine equivalence on feasible (tile-sized) grids, and the aliasing
    defect it replaces."""

    @staticmethod
    def _engine_area(src, out_hw):
        """Engine area remap on the strided lattice (feasible only because
        this grid is tile-sized -- see _area_downsample's feasibility note)."""
        from stellaRGB.warp import remap

        h, w = out_hw
        fx = src.shape[1] / w
        fy = src.shape[0] / h
        xs = (np.arange(w, dtype=np.float32) * np.float32(fx) + np.float32((fx - 1) / 2))
        ys = (np.arange(h, dtype=np.float32) * np.float32(fy) + np.float32((fy - 1) / 2))
        return remap(
            src,
            np.repeat(xs[None, :], h, axis=0),
            np.repeat(ys[:, None], w, axis=1),
            kernel="area",
        )

    def test_area_downsample_matches_engine_area(self):
        """On grids where the engine call is feasible (tile-sized, so its
        uniform global-span tap window stays bounded), the helper reproduces
        the engine's D10 area kernel bit-exactly (measured maxdiff 0.0 on
        random data; f32 engine vs f64 helper)."""
        from lsst.pipe.tasks.rgb2hips._utils import _area_downsample

        rng = np.random.default_rng(3)
        for shape, out in (((16, 16, 3), (8, 8)), ((32, 32, 3), (16, 16))):
            src = rng.random(shape).astype(np.float32)
            mine = _area_downsample(src, out)
            eng = self._engine_area(src, out)
            self.assertEqual(mine.shape, eng.shape)
            np.testing.assert_allclose(mine, eng, atol=1e-7)

    def test_area_prefilter_kills_nyquist_energy(self):
        """A 1-pixel checkerboard is pure Nyquist energy on both axes: the
        area prefilter returns a FLAT 0.5 level (exact), while the legacy
        cv2.resize(INTER_LANCZOS4) -- no prefilter -- aliases it into 0.064
        of ringing (measured). The near-Nyquist sine case is identical:
        helper == exact 2x2 box mean (dev 0.0), legacy dev 0.146. The
        pyramid swap is a strict aliasing improvement (defect fix)."""
        from lsst.pipe.tasks.rgb2hips._utils import _area_downsample

        grid = np.add.outer(np.arange(64), np.arange(64))
        checker = (grid % 2).astype(np.float32)[..., None] * np.ones((1, 1, 3), np.float32)
        flat = _area_downsample(checker, (32, 32))
        np.testing.assert_allclose(flat, 0.5, atol=1e-6)
        legacy = cv2.resize(checker, (32, 32), interpolation=cv2.INTER_LANCZOS4)
        self.assertGreater(float(np.abs(legacy - 0.5).max()), 0.01)

        xx = np.arange(64, dtype=np.float32)
        sig = (0.5 + 0.5 * np.sin(2 * np.pi * 0.45 * xx))[None, :, None]
        sig = sig * np.ones((64, 1, 3), np.float32)
        exact = sig.reshape(64, 32, 2, 3).mean(axis=2)[::2]
        mine = _area_downsample(sig, (32, 32))
        np.testing.assert_allclose(mine[:, :32, :], exact, atol=1e-6)
        legacy_s = cv2.resize(sig, (32, 32), interpolation=cv2.INTER_LANCZOS4)
        self.assertGreater(float(np.abs(legacy_s[:, :32, :] - exact).max()), 0.05)

    def test_area_downsample_exact_box_mean_vs_reference(self):
        """Independent reference (cv2 INTER_AREA pixel-area mean): exact
        within float noise for integer and non-integer (odd-dim) factors --
        cv2 lanczos has no defined box semantics there; the helper matches
        the clip-overlap definition (measured maxdiff <= 1.2e-7)."""
        from lsst.pipe.tasks.rgb2hips._utils import _area_downsample

        rng = np.random.default_rng(1)
        src = rng.random((65, 90, 3)).astype(np.float32)
        for out in ((30, 36), (13, 18)):
            mine = _area_downsample(src, out)
            ref = cv2.resize(src, (out[1], out[0]), interpolation=cv2.INTER_AREA)
            np.testing.assert_allclose(mine, ref, atol=1e-5)
        with self.assertRaises(ValueError):
            _area_downsample(src, (66, 90))  # upsampling is not a downsample

    def test_area_downsample_nan_local(self):
        """NaN poisons exactly the boxes touching it (the engine's D9
        contract), not the whole row (which a naive integral-image would
        do); a NaN-free grid has no NaNs in any level."""
        from lsst.pipe.tasks.rgb2hips._utils import _area_downsample

        rng = np.random.default_rng(2)
        src = rng.random((16, 16, 3)).astype(np.float32)
        src[1, 1, :] = np.nan
        out = _area_downsample(src, (8, 8))
        nan_boxes = np.any(np.isnan(out), axis=-1)
        self.assertTrue(np.all(np.isnan(out[0, 0])))
        self.assertEqual(int(nan_boxes.sum()), 1)
        self.assertTrue(np.all(np.isfinite(out[~nan_boxes])))

    def test_downsample_kernel_choicefield_defaults(self):
        """The downsample_kernel knob exists on both pyramid tasks with each
        task's historical default (area for the single-pass high-order
        pyramid, lanczos for the cascaded low-order resampler) and unknown
        kernels are rejected."""
        from lsst.pipe.tasks.rgb2hips import HighOrderHipsTaskConfig, LowOrderHipsTaskConfig
        from lsst.pex.config import FieldValidationError

        high = HighOrderHipsTaskConfig()
        self.assertEqual(high.downsample_kernel, "area")
        low = LowOrderHipsTaskConfig()
        self.assertEqual(low.downsample_kernel, "lanczos")
        with self.assertRaises(FieldValidationError):
            high.downsample_kernel = "bogus"
        with self.assertRaises(FieldValidationError):
            low.downsample_kernel = "bogus"

    def test_downsample_kernel_dispatch_bit_exact(self):
        """downsample_hips_array dispatches bit-exactly: "lanczos" is the
        legacy cv2.resize(INTER_LANCZOS4) call, "area" is _area_downsample
        (float32 out) -- the kernel choice changes pixels, never plumbing."""
        from lsst.pipe.tasks.rgb2hips._utils import _area_downsample, downsample_hips_array

        rng = np.random.default_rng(7)
        src = rng.random((64, 64, 3)).astype(np.float32)
        np.testing.assert_array_equal(
            downsample_hips_array(src, (32, 32), "lanczos"),
            cv2.resize(src, (32, 32), interpolation=cv2.INTER_LANCZOS4),
        )
        area = downsample_hips_array(src, (32, 32), "area")
        np.testing.assert_array_equal(area, _area_downsample(src, (32, 32)))
        self.assertEqual(area.dtype, np.float32)
        # The two kernels genuinely differ on structure (guards against both
        # branches collapsing onto one code path).
        lanczos = downsample_hips_array(src, (32, 32), "lanczos")
        self.assertGreater(float(np.abs(area - lanczos).max()), 1e-3)


class ColorHipsHdrAvifTestCase(unittest.TestCase):
    """HDR AVIF delivery: config pairing, PQ-writer delegation, AllSky
    float-canvas geometry, and the (now conditional) is_hdr guard."""

    @staticmethod
    def _allsky_config(base_uri, **overrides):
        from lsst.pipe.tasks.rgb2hips import AllSkyHipsTaskConfig

        config = AllSkyHipsTaskConfig()
        config.hips_base_uri = base_uri
        config.color_ordering = "rgb"
        config.properties.creator_did_template = "doi:test-{band}"
        config.properties.obs_title_template = "test {band}"
        config.properties.obs_description_template = "test {band}"
        for name, value in overrides.items():
            setattr(config, name, value)
        return config

    def test_configs_pair_hdr_with_avif(self):
        """is_hdr and file_extension='avif' must come as a pair on all
        three tasks (mismatches raise, including high-order's lanczos-on-
        HDR veto); lum_norm defaults to 1.0 -> 100 nits."""
        from lsst.pipe.tasks.rgb2hips import (
            AllSkyHipsTaskConfig,
            HighOrderHipsTaskConfig,
            LowOrderHipsTaskConfig,
        )

        for config_class in (HighOrderHipsTaskConfig, LowOrderHipsTaskConfig, AllSkyHipsTaskConfig):
            with self.subTest(config=config_class.__name__):
                config = config_class()
                config.hips_base_uri = "/tmp/hips_hdr_avif_config"
                config.color_ordering = "rgb"
                if config_class is AllSkyHipsTaskConfig:
                    config.properties.creator_did_template = "doi:t-{band}"
                    config.properties.obs_title_template = "t {band}"
                    config.properties.obs_description_template = "t {band}"
                self.assertFalse(config.is_hdr)
                self.assertEqual(config.lum_norm, 100.0)
                self.assertIsNone(config.clamp_values)
                # HDR on an SDR format: refused before anything can clip.
                config.is_hdr = True
                with self.assertRaises(ValueError):
                    config.validate()
                config.file_extension = "avif"
                if config_class is not AllSkyHipsTaskConfig:
                    # The HDR kernel veto fires on the low-order default.
                    config.downsample_kernel = "area"
                config.validate()  # the avif+is_hdr pairing is accepted
                if config_class is not AllSkyHipsTaskConfig:
                    config.downsample_kernel = "lanczos"
                    with self.assertRaises(ValueError):
                        config.validate()
        # The reverse mismatch (avif without is_hdr) is refused too.
        config = LowOrderHipsTaskConfig()
        config.hips_base_uri = "/tmp/hips_hdr_avif_config"
        config.color_ordering = "rgb"
        config.file_extension = "avif"
        with self.assertRaises(ValueError):
            config.validate()

    def test_write_hips_image_hdr_delegates(self):
        """The HDR writer branch sanitizes (NaN/inf/negatives -> 0), does
        NOT clip above 1, and hands the array to the stellaRGB PQ writer
        with lum_norm/clamp_values passed through."""
        import tempfile
        from unittest import mock

        from lsst.pipe.tasks.rgb2hips._utils import _write_hips_image
        from lsst.resources import ResourcePath

        base = ResourcePath(tempfile.mkdtemp(), forceDirectory=True)
        src = np.array(
            [[[2.5, 0.5, 0.0], [-3.0, np.nan, 1.0]], [[np.inf, 1.0, 0.25], [0.0, 0.0, 0.0]]],
            dtype=np.float32,
        )
        with mock.patch("stellaRGB._color_image.ColorImage", autospec=True) as ctor:
            ctor.return_value.write_array_hdr_avif.side_effect = (
                lambda path, **kw: open(path, "wb").write(b"avif-stub")
            )
            _write_hips_image(
                src, 42, 11, base, "avif", "uint8", is_hdr=True, lum_norm=250.0, clamp_values=12.5
            )

        ctor.assert_called_once()
        delivered = ctor.call_args[0][0]
        self.assertEqual(delivered.shape, src.shape)
        self.assertTrue(np.all(np.isfinite(delivered)))
        self.assertTrue(np.all(delivered >= 0.0))
        # Value above 1 survived: the SDR clip must not run on HDR data.
        self.assertAlmostEqual(float(delivered[0, 0, 0]), 2.5)
        ctor.return_value.write_array_hdr_avif.assert_called_once_with(
            mock.ANY, clamp_values=12.5, lum_norm=250.0
        )
        self.assertTrue(base.join("Norder11/Dir0/Npix42.avif").exists())

    def test_allsky_hdr_canvas_geometry(self):
        """Four order-8 children of the origin order-7 node of one order-3
        tile land in the [[b, d], [c, e]] quadrant slots of their canvas
        cells, area-downsampled from the 256x256 butler resolution, and
        the canvas goes to the PQ writer with the configured lum_norm."""
        import os
        import tempfile
        from unittest import mock

        from lsst.pipe.tasks.rgb2hips import AllSkyHipsTask

        base = tempfile.mkdtemp()
        task = AllSkyHipsTask(
            config=self._allsky_config(
                base, is_hdr=True, file_extension="avif", min_order=3, allsky_tilesize=64
            )
        )

        # Order-8 children of the order-7 node 0 inside order-3 pixel 100:
        # nested ids (100 << 10) + {0,1,2,3} sit at template positions
        # (0,0), (1,0), (0,1), (1,1) — the [[b, d], [c, e]] rule.
        ancestor = 100
        kids = [(ancestor << 10) + k for k in range(4)]
        values = {kids[0]: 0.1, kids[1]: 0.25, kids[2]: 0.5, kids[3]: 0.75}
        arrays = [(pid, np.full((256, 256, 3), v, dtype=np.float32)) for pid, v in values.items()]
        # One NaN and one negative pixel must sanitize to 0, not poison (they
        # dilute one output cell slightly; checked as median/min below).
        arrays[0][1][0, 0, 0] = np.nan
        arrays[0][1][0, 0, 1] = -2.0

        captured = {}
        with mock.patch("stellaRGB._color_image.ColorImage", autospec=True) as ctor:
            ctor.return_value.write_array_hdr_avif.side_effect = (
                lambda path, **kw: captured.update(kwargs=kw) or open(path, "wb").write(b"x")
            )
            task._write_allsky_file_hdr(3, arrays)

        canvas = ctor.call_args[0][0]
        # Order 3: 768 tiles, 27 wide (spec S4.3.2), ceil(768/27) = 29 rows.
        self.assertEqual(canvas.shape, (29 * 64, 27 * 64, 3))
        self.assertTrue(np.all(np.isfinite(canvas)))
        self.assertTrue(np.all(canvas >= 0.0))
        row, col = ancestor // 27, ancestor % 27
        sub = 64 >> 5  # one order-8 cell: 2x2 px of the 32x32 subdivision
        quadrant = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}  # [[b, d], [c, e]]
        for k in range(4):
            r_sub, c_sub = quadrant[k]
            block = canvas[
                row * 64 + r_sub * sub : row * 64 + (r_sub + 1) * sub,
                col * 64 + c_sub * sub : col * 64 + (c_sub + 1) * sub,
            ]
            if k == 0:  # sanitized-probe block: diluted in one output px, not exact.
                self.assertAlmostEqual(float(np.median(block)), 0.1, places=5)
                self.assertGreaterEqual(float(block.min()), 0.1 * (128 - 1) / 128 - 1e-6)
            else:
                np.testing.assert_allclose(block, values[kids[k]], rtol=1e-6)
        # Exactly the four 2x2 cells are lit; the rest of the canvas is empty.
        self.assertEqual(int(np.count_nonzero(canvas)), 4 * 2 * 2 * 3)
        self.assertEqual(captured["kwargs"]["lum_norm"], 100.0)
        self.assertIsNone(captured["kwargs"]["clamp_values"])
        self.assertTrue(os.path.exists(os.path.join(base, "color_rgb", "Norder3", "Allsky.avif")))

    def test_allsky_hdr_refuses_empty_inputs(self):
        import tempfile

        from lsst.pipe.tasks.rgb2hips import AllSkyHipsTask

        task = AllSkyHipsTask(
            config=self._allsky_config(
                tempfile.mkdtemp(), is_hdr=True, file_extension="avif", min_order=7
            )
        )
        with self.assertRaises(ValueError):
            task._write_allsky_file_hdr(7, [])

    def test_hdr_guard_lifted_when_configured(self):
        """Mirror of test_high_order_hips_rejects_hdr: with is_hdr=True +
        avif the assembly proceeds instead of raising."""
        from lsst.images import ColorImage
        from lsst.pipe.base import InMemoryDatasetHandle
        from lsst.pipe.tasks.rgb2hips._high_order_hips import HighOrderHipsTask

        config = HighOrderHipsTask.ConfigClass()
        config.hips_base_uri = "/tmp/hips_hdr_guard_allowed"
        config.color_ordering = "gri"
        config.is_hdr = True
        config.file_extension = "avif"
        task = HighOrderHipsTask(config=config)
        image = ColorImage(np.zeros((16, 16, 3), dtype=np.float32), metadata={"is_hdr": True})
        handle = InMemoryDatasetHandle(inMemoryDataset=image)
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(16, 16))
        # Must not raise: feathering proceeds on the linear data unclipped
        # (grow=4 leaves the feather ramp inside the 16x16 patch).
        task._assemble_sub_region({1: [(handle, None, bbox)]}, 4)

    def test_properties_bitpix_follows_encoding(self):
        """An HDR tree must declare hips_pixel_bitpix=10 (the real PQ AVIF
        encoder depth, deliberately outside the HiPS-registered values),
        while the SDR path keeps the hips_bitpix config mapping."""
        import os
        import tempfile

        from lsst.pipe.tasks.rgb2hips import AllSkyHipsTask

        def bitpix_value(**overrides):
            base = tempfile.mkdtemp()
            task = AllSkyHipsTask(config=self._allsky_config(base, **overrides))
            task._write_properties_and_moc(11, np.array([0, 1, 2]), task.config.shift_order, "rgb")
            with open(os.path.join(base, "color_rgb", "properties")) as fh:
                for line in fh:
                    key, _, value = line.partition("=")
                    if key.strip() == "hips_pixel_bitpix":
                        return value.strip()
            raise AssertionError("hips_pixel_bitpix missing from properties")

        self.assertEqual(bitpix_value(is_hdr=True, file_extension="avif"), "10")
        self.assertEqual(bitpix_value(), "8")  # SDR default: hips_bitpix uint8
