import logging
import re

import cv2
import numpy as np
import pytest

from lsst.afw.geom import makeSkyWcs
from lsst.geom import Box2I, degrees, Extent2I, Point2D, Point2I, SpherePoint
from lsst.images import Box, ColorImage, SkyProjection, TractFrame
from lsst.pipe.base import InMemoryDatasetHandle
from lsst.skymap import DiscreteSkyMap

from lsst.pipe.tasks.prettyPictureMaker._meta_mosaic import MetaMosaicConfig, MetaMosaicTask
from lsst.pipe.tasks.prettyPictureMaker._utils import FeatheredMosaicCreator, warp_maps


def _make_task(reference_tract=None):
    config = MetaMosaicConfig()
    config.reference_tract = reference_tract
    task = MetaMosaicTask.__new__(MetaMosaicTask)
    task.config = config
    return task


def _make_skymap(tracts=2):
    """Build a multi-tract DiscreteSkyMap with overlapping patch borders.

    Patches overlap their neighbours (outer 640 px vs 400 px pitch) and the
    feathering region (``patch_grow`` = inner cell size = 100 px) is comfortably
    smaller than the pitch, so each patch's center stays pure while the borders
    blend. Tracts step 0.05 deg in RA (tract 2 at 0.10 deg, claimed-cell
    footprints still land on the reference canvas).
    """
    config = DiscreteSkyMap.ConfigClass()
    tract_builder = config.tractBuilder["cells"]
    tract_builder.cellInnerDimensions = [100, 100]
    tract_builder.cellBorder = 20
    tract_builder.numCellsPerPatchInner = 4
    tract_builder.numCellsInPatchBorder = 1
    config.tractBuilder.name = "cells"
    config.raList = [10.0 + 0.05 * i for i in range(tracts)]
    config.decList = [-30.0] * tracts
    config.radiusList = [0.15] * tracts
    return DiscreteSkyMap(config)


def _make_handle(skymap, tract, patch, value, metadata=None):
    """Build a deferred handle returning a constant-color patch of the right size."""
    bbox = skymap[tract][patch].getOuterBBox()
    array = np.empty((bbox.getHeight(), bbox.getWidth(), 3), dtype=np.float32)
    array[...] = value
    image = ColorImage(array, bbox=Box.from_legacy(bbox), metadata=metadata)
    return InMemoryDatasetHandle(inMemoryDataset=image, skymap="test", tract=tract, patch=patch)


def _make_run_task(reference_tract=0, bin_factor=1, crop_mode="full_coverage"):
    config = MetaMosaicConfig()
    config.reference_tract = reference_tract
    config.bin_factor = bin_factor
    config.crop_mode = crop_mode
    return MetaMosaicTask(config=config)


class TestReferenceTractSelection:
    def test_auto_most_patches(self):
        task = _make_task()
        inputs = {0: [1, 2], 1: [1, 2, 3, 4], 2: [1, 2, 3]}
        assert task._select_reference_tract(inputs) == 1

    def test_auto_tie_breaks_lower_tract(self):
        task = _make_task()
        inputs = {5: [1, 2, 3], 9: [4, 5, 6]}
        assert task._select_reference_tract(inputs) == 5

    def test_config_overrides_auto(self):
        task = _make_task(reference_tract=7)
        inputs = {5: [1, 2, 3], 9: [4, 5, 6]}
        assert task._select_reference_tract(inputs) == 7


class TestToFloat32:
    def test_uint8_normalized(self):
        arr = np.array([[[255], [0], [128]]], dtype=np.uint8)
        result = MetaMosaicTask._to_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result[0, 0, 0], 1.0)
        np.testing.assert_allclose(result[0, 1, 0], 0.0)
        np.testing.assert_allclose(result[0, 2, 0], 128.0 / 255.0)

    def test_uint16_normalized(self):
        arr = np.array([[65535]], dtype=np.uint16)
        result = MetaMosaicTask._to_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result[0, 0], 1.0)

    def test_float32_passthrough(self):
        arr = np.array([[0.5]], dtype=np.float32)
        result = MetaMosaicTask._to_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 0.5)


class TestFeatheredMosaicNaN:
    """Unit tests for the NaN-reweighting featherer added to FeatheredMosaicCreator."""

    @staticmethod
    def _box(x, y, w, h):
        return Box.from_legacy(Box2I(Point2I(x, y), Extent2I(w, h)))

    def test_nan_reweights_overlap(self):
        maker = FeatheredMosaicCreator(patch_grow=10)
        h = 40
        out_w = 60
        full = self._box(0, 0, out_w, h)
        sum_img = np.zeros((h, out_w, 3), dtype=np.float32)
        # Per-pixel (all-channels-finite) weight accumulator: one value per pixel.
        weight = np.zeros((h, out_w), dtype=np.float32)
        # A: constant 1 over x in [0, 40), but NaN on its right quarter (x in [30, 40)).
        a = np.ones((h, 40, 3), dtype=np.float32)
        a[:, 30:, :] = np.nan
        # B: constant 0 over x in [20, 60).
        b = np.zeros((h, 40, 3), dtype=np.float32)
        maker.add_to_image(sum_img, a, full, self._box(0, 0, 40, h), reverse=False, weight=weight)
        maker.add_to_image(sum_img, b, full, self._box(20, 0, 40, h), reverse=False, weight=weight)
        out = FeatheredMosaicCreator.finalize(sum_img, weight)

        assert np.all(np.isfinite(out))
        # Where A is NaN in the overlap, only B contributes -> B's value (0).
        np.testing.assert_allclose(out[20, 35, :], 0.0, atol=1e-6)
        # Outside the overlap A is fully valid -> A's value (1).
        np.testing.assert_allclose(out[20, 10, :], 1.0, atol=1e-6)

    def test_no_weight_preserves_feathered_sum(self):
        """Without a weight array the old feathered-sum (NaN-propagating) path holds."""
        maker = FeatheredMosaicCreator(patch_grow=10)
        h, w = 40, 60
        full = self._box(0, 0, w, h)
        img = np.zeros((h, w, 3), dtype=np.float32)
        a = np.ones((h, 60, 3), dtype=np.float32)
        a[:, 30:, :] = np.nan
        maker.add_to_image(img, a, full, self._box(0, 0, 60, h), reverse=False)
        # The NaN propagates (no reweighting), as in the legacy feathered sum.
        assert np.isnan(img).any()


class TestMapPatchBoxToRef:
    def test_foreign_patch_maps_to_ref_coordinates(self):
        skymap = _make_skymap()
        task = _make_run_task()
        ref_tract = skymap[0]
        foreign_tract = skymap[1]
        ref_proj = SkyProjection.from_legacy(
            ref_tract.getWcs(),
            TractFrame(skymap="test", tract=0, bbox=Box.from_legacy(ref_tract.bbox)),
        )
        patch_proj = SkyProjection.from_legacy(
            foreign_tract.getWcs(),
            TractFrame(skymap="test", tract=1, bbox=Box.from_legacy(foreign_tract.bbox)),
        )
        patch_bbox = foreign_tract[0].getOuterBBox()

        mapped = task._map_patch_box_to_ref(patch_proj, patch_bbox, ref_proj)

        assert not mapped.isEmpty()
        # The two tracts are at different RA, so the mapped box must differ from
        # the source box (i.e. the transform is doing something).
        assert mapped.getBeginX() != patch_bbox.getBeginX()
        # Round trip: the center of the mapped box should map back to the center
        # of the source patch through the two projections.
        center = mapped.getCenter()
        sky = ref_proj.pixel_to_sky(x=np.array([center.getX()]), y=np.array([center.getY()]))
        pixels = patch_proj.sky_to_pixel(sky)
        np.testing.assert_allclose(
            [np.asarray(pixels.x)[0], np.asarray(pixels.y)[0]],
            [patch_bbox.getCenterX(), patch_bbox.getCenterY()],
            rtol=1e-3,
            atol=1.0,
        )


class TestScaledProjection:
    def test_binned_pixel_anchors_to_ref_origin(self):
        skymap = _make_skymap()
        task = _make_run_task(reference_tract=0)
        ref_wcs = skymap[0].getWcs()
        ref_proj = SkyProjection.from_legacy(
            ref_wcs,
            TractFrame(skymap="test", tract=0, bbox=Box.from_legacy(skymap[0].bbox)),
        )
        full_origin = Point2D(12500.0, 12500.0)
        out_box = Box2I(Point2I(0, 0), Extent2I(10, 10))
        proj = task._scaled_projection(ref_wcs, 0, "test", full_origin, 2, out_box)

        # Binned pixel (0, 0) must map back to the full-resolution origin pixel.
        sky = proj.pixel_to_sky(x=np.array([0.0]), y=np.array([0.0]))
        pix = ref_proj.sky_to_pixel(sky)
        np.testing.assert_allclose(
            [np.asarray(pix.x)[0], np.asarray(pix.y)[0]],
            [full_origin.getX(), full_origin.getY()],
            rtol=0,
            atol=1e-6,
        )

        # Binned pixel (1, 1) maps back to the origin + one bin step per axis.
        sky = proj.pixel_to_sky(x=np.array([1.0]), y=np.array([1.0]))
        pix = ref_proj.sky_to_pixel(sky)
        np.testing.assert_allclose(
            [np.asarray(pix.x)[0], np.asarray(pix.y)[0]],
            [full_origin.getX() + 2, full_origin.getY() + 2],
            rtol=0,
            atol=1e-6,
        )


class TestRunEndToEnd:
    def test_mosaic_shape_and_placement(self):
        skymap = _make_skymap()
        task = _make_run_task(reference_tract=0, crop_mode="data_extent")
        ref_bbox = skymap[0][0].getOuterBBox()

        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),  # reference tract: red
            _make_handle(skymap, 1, 0, [0.0, 0.0, 1.0]),  # foreign tract: blue
        ]

        try:
            result = task.run(inputRGB=handles, skyMap=skymap)
        finally:
            if hasattr(task, "imageHandle"):
                task.imageHandle.close()
        image = result.outputRGBMosaic
        array = image.array

        # Missing data is converted to zero at the very end.
        assert np.all(np.isfinite(array))

        # The output is cropped to the data footprint, so it must enclose the
        # full reference patch (reference cells are never dropped: even a
        # feathered seam facing a gap carries 1e-17 weight) and the mapped
        # foreign patch's center (inside its warped footprint, hence inside the
        # crop), but NOT the whole inflated mapped bounding box -- the crop
        # exists precisely to drop those empty margin pixels.
        ref_proj = SkyProjection.from_legacy(
            skymap[0].getWcs(),
            TractFrame(skymap="test", tract=0, bbox=Box.from_legacy(skymap[0].bbox)),
        )
        patch_proj = SkyProjection.from_legacy(
            skymap[1].getWcs(),
            TractFrame(skymap="test", tract=1, bbox=Box.from_legacy(skymap[1].bbox)),
        )
        mapped = task._map_patch_box_to_ref(patch_proj, skymap[1][0].getOuterBBox(), ref_proj)
        out_box = image.bbox.to_legacy()
        assert out_box.contains(ref_bbox)
        assert out_box.contains(Point2I(mapped.getCenter()))

        # Reference-tract patch lands 1:1 at its own location.
        ox, oy = image.bbox.x.start, image.bbox.y.start
        ax = int(ref_bbox.getCenterX() - ox)
        ay = int(ref_bbox.getCenterY() - oy)
        np.testing.assert_allclose(array[ay, ax, :], [1.0, 0.0, 0.0], atol=1e-2)

        # Foreign-tract patch is present somewhere.
        assert np.any(array[..., 2] > 0.5)

    def test_nan_foreign_patch_does_not_corrupt(self):
        skymap = _make_skymap()
        task = _make_run_task(reference_tract=0)
        ref_bbox = skymap[0][0].getOuterBBox()

        blue = np.empty((ref_bbox.getHeight(), ref_bbox.getWidth(), 3), dtype=np.float32)
        blue[...] = [0.0, 0.0, 1.0]
        # A NaN block that the warp will encounter.
        blue[:10, :10, :] = np.nan
        foreign = InMemoryDatasetHandle(
            inMemoryDataset=ColorImage(blue, bbox=Box.from_legacy(skymap[1][0].getOuterBBox())),
            skymap="test",
            tract=1,
            patch=0,
        )
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),
            foreign,
        ]

        try:
            result = task.run(inputRGB=handles, skyMap=skymap)
        finally:
            if hasattr(task, "imageHandle"):
                task.imageHandle.close()
        array = result.outputRGBMosaic.array

        # The NaN from the warped foreign patch must not propagate.
        assert np.all(np.isfinite(array))
        # The reference patch must remain intact.
        ox, oy = result.outputRGBMosaic.bbox.x.start, result.outputRGBMosaic.bbox.y.start
        ax = int(ref_bbox.getCenterX() - ox)
        ay = int(ref_bbox.getCenterY() - oy)
        np.testing.assert_allclose(array[ay, ax, :], [1.0, 0.0, 0.0], atol=1e-2)

    def test_overlapping_patches_are_averaged(self):
        """Two foreign patches overlapping one cell should be combined by
        the NaN-reweighted feather (weighted mean where both are valid).

        Run with crop_mode="data_extent": warped parallelogram margins leave
        holes in every claimed cell, so "full_coverage" would (correctly)
        refuse to crop this mosaic at all; this test is about blending.
        """
        skymap = _make_skymap()
        task = _make_run_task(reference_tract=0, crop_mode="data_extent")

        # Two adjacent foreign-tract patches (0 and 1) of different colors.
        handles = [
            _make_handle(skymap, 1, 0, [0.0, 0.0, 1.0]),  # blue
            _make_handle(skymap, 1, 1, [0.0, 1.0, 0.0]),  # green
        ]

        try:
            result = task.run(inputRGB=handles, skyMap=skymap)
        finally:
            if hasattr(task, "imageHandle"):
                task.imageHandle.close()
        array = result.outputRGBMosaic.array
        assert np.all(np.isfinite(array))
        assert np.any(array[..., 2] > 0.5)  # blue present
        assert np.any(array[..., 1] > 0.5)  # green present

    def test_adjacent_ref_patches_feather_blend(self):
        """Adjacent, overlapping reference patches blend without seams or NaN."""
        skymap = _make_skymap()
        task = _make_run_task(reference_tract=0)

        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),  # red
            _make_handle(skymap, 0, 1, [0.0, 1.0, 0.0]),  # green
        ]

        try:
            result = task.run(inputRGB=handles, skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(task, attr):
                    getattr(task, attr).close()
        array = result.outputRGBMosaic.array
        assert np.all(np.isfinite(array))

        ox, oy = result.outputRGBMosaic.bbox.x.start, result.outputRGBMosaic.bbox.y.start
        for patch, color in ((0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0])):
            bbox = skymap[0][patch].getOuterBBox()
            ax = int(bbox.getCenterX() - ox)
            ay = int(bbox.getCenterY() - oy)
            np.testing.assert_allclose(array[ay, ax, :], color, atol=1e-2)

    def test_bin_factor_halves_dimensions(self):
        """A bin factor of 2 must halve each output dimension.

        Cropping is disabled (crop_mode="none") so the base rectangle is the
        full touched-cells grid and the pure binning math is exercised.
        """
        skymap = _make_skymap()
        task = _make_run_task(reference_tract=0, crop_mode="none")
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),
            _make_handle(skymap, 1, 0, [0.0, 0.0, 1.0]),
        ]
        try:
            full = task.run(inputRGB=handles, skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(task, attr):
                    getattr(task, attr).close()
        fshape = full.outputRGBMosaic.array.shape

        config = MetaMosaicConfig()
        config.bin_factor = 2
        config.reference_tract = 0
        config.crop_mode = "none"
        binned = MetaMosaicTask(config=config)
        try:
            result = binned.run(inputRGB=handles, skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(binned, attr):
                    getattr(binned, attr).close()
        array = result.outputRGBMosaic.array
        assert np.all(np.isfinite(array))
        assert array.shape[1] == fshape[1] // 2
        assert array.shape[0] == fshape[0] // 2

    def test_crops_empty_margin(self):
        """With crop_mode="data_extent" the output is cropped to the data
        footprint, not the touched-cells rectangle: corner cells the foreign
        patch only inflates into are dropped.
        """
        skymap = _make_skymap()
        ref_bbox = skymap[0][0].getOuterBBox()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),  # reference tract: red
            _make_handle(skymap, 1, 0, [0.0, 0.0, 1.0]),  # foreign tract: blue
        ]

        task = _make_run_task(reference_tract=0, crop_mode="data_extent")
        try:
            cropped = task.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(task, attr):
                    getattr(task, attr).close()

        full_task = _make_run_task(reference_tract=0, crop_mode="none")
        try:
            uncropped = full_task.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(full_task, attr):
                    getattr(full_task, attr).close()

        crop_array = cropped.outputRGBMosaic.array
        full_array = uncropped.outputRGBMosaic.array

        # The crop is strictly smaller than the uncropped rectangle in both
        # dimensions (the mapped foreign box leaves empty corner cells).
        assert crop_array.shape[0] < full_array.shape[0]
        assert crop_array.shape[1] < full_array.shape[1]

        # Both rasters keep all the data: finite everywhere with the reference
        # patch intact at its 1:1 location and the foreign color present.
        for image in (cropped.outputRGBMosaic, uncropped.outputRGBMosaic):
            array = image.array
            assert np.all(np.isfinite(array))
            ox, oy = image.bbox.x.start, image.bbox.y.start
            ax = int(ref_bbox.getCenterX() - ox)
            ay = int(ref_bbox.getCenterY() - oy)
            np.testing.assert_allclose(array[ay, ax, :], [1.0, 0.0, 0.0], atol=1e-2)
            assert np.any(array[..., 2] > 0.5)

    def test_feathered_seam_toward_gap_is_not_black(self):
        """A reference cell's feathered outer edge (descending ramp, used for
        the bottom/right masks) facing an empty gap cell must keep the patch
        data, not zero weight. The descending ramp's outermost value is the
        tiny nonzero ``1e-17`` rather than exact zero, so the seam pixel still
        normalizes to the reference color instead of going black.

        The foreign patch is placed far enough along the tract-1 row that two
        fully unclaimed (gap) cells separate it from the reference patch, so
        the reference cell's feathered seam has no neighbor covering it and
        sits strictly inside the output raster (data_extent mode keeps the
        raster spanning the gap and the outlier).
        """
        skymap = _make_skymap()
        ref_bbox = skymap[0][0].getOuterBBox()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),  # reference tract: red
            _make_handle(skymap, 1, 5, [0.0, 0.0, 1.0]),  # foreign tract: blue, 5 cells away
        ]
        task = _make_run_task(reference_tract=0, crop_mode="data_extent")
        try:
            result = task.run(inputRGB=handles, skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(task, attr):
                    getattr(task, attr).close()
        image = result.outputRGBMosaic
        array = image.array
        assert np.all(np.isfinite(array))

        # Seam: the reference cell's outermost column (its right feather-mask
        # edge) in output-array coordinates. Box2I.getMaxX is INCLUSIVE (the
        # exclusive end would be one column past the data).
        ox, oy = image.bbox.x.start, image.bbox.y.start
        seam_col = ref_bbox.getMaxX() - ox
        mid_row = int(ref_bbox.getCenterY()) - oy
        # The seam is strictly interior because the gap cells extend the raster
        # beyond it, and no contributor covers the columns just past the seam.
        assert 0 < seam_col < array.shape[1] - 1
        assert not np.any(array[mid_row, seam_col + 1 : seam_col + 41, :])
        # The seam pixel carries the reference data (full red after the
        # NaN-reweighted normalization at 1e-17 weight), not black.
        np.testing.assert_allclose(array[mid_row, seam_col, :], [1.0, 0.0, 0.0], atol=1e-2)
        # The foreign color is present somewhere in the mosaic.
        assert np.any(array[..., 2] > 0.5)

    def test_crop_with_binning(self):
        """Cropping and binning compose: crop first, then halve the cropped
        rectangle (rounded down), keeping the output metadata attached."""
        skymap = _make_skymap()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),
            _make_handle(skymap, 1, 0, [0.0, 0.0, 1.0], metadata={"is_hdr": True}),
        ]

        base = _make_run_task(reference_tract=0)
        try:
            full_res = base.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(base, attr):
                    getattr(base, attr).close()
        crop_h, crop_w = full_res.outputRGBMosaic.array.shape[:2]

        binned = _make_run_task(reference_tract=0, bin_factor=2)
        try:
            result = binned.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(binned, attr):
                    getattr(binned, attr).close()

        array = result.outputRGBMosaic.array
        assert np.all(np.isfinite(array))
        # Half (rounded down) of the CROPPED full-resolution dimensions.
        assert array.shape[1] == crop_w // 2
        assert array.shape[0] == crop_h // 2
        # Both colors survive the crop+bin, and the metadata is carried out.
        assert np.any(array[..., 2] > 0.5)
        assert np.any(array[..., 0] > 0.5)
        assert result.outputRGBMosaic.metadata["is_hdr"] is True
        assert result.outputRGBMosaic.sky_projection is not None

        # WCS anchor: binned pixel (0, 0) must land on the same sky as the
        # full-resolution reference pixel at the cropped origin
        # ``full_origin = grid_origin + min_i * pitch_x + cx0`` (and likewise
        # for y). With bin_factor=1 the cropped out_box origin already equals
        # that reference-tract pixel, so it pins the binning arithmetic here.
        full_origin_x = float(full_res.outputRGBMosaic.bbox.x.start)
        full_origin_y = float(full_res.outputRGBMosaic.bbox.y.start)
        ref_proj = SkyProjection.from_legacy(
            skymap[0].getWcs(),
            TractFrame(skymap="test", tract=0, bbox=Box.from_legacy(skymap[0].bbox)),
        )
        sky_ref = ref_proj.pixel_to_sky(
            x=np.array([full_origin_x]), y=np.array([full_origin_y])
        )
        sky_binned = result.outputRGBMosaic.sky_projection.pixel_to_sky(
            x=np.array([0.0]), y=np.array([0.0])
        )
        # Angular closeness: RA/Dec agree to well under a milliarcsecond
        # (1e-9 deg ~ 3.6 uas).
        np.testing.assert_allclose(sky_binned.ra.deg, sky_ref.ra.deg, atol=1e-9)
        np.testing.assert_allclose(sky_binned.dec.deg, sky_ref.dec.deg, atol=1e-9)
        assert sky_binned.separation(sky_ref).arcsec[0] < 1e-3

    def test_bin_factor_larger_than_crop_raises(self):
        """A bin factor beyond the cropped output dims must fail with a clear
        ValueError instead of an opaque cv2.error on a 0-sized resize."""
        skymap = _make_skymap()
        task = _make_run_task(reference_tract=0, bin_factor=2000)
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),
            _make_handle(skymap, 1, 0, [0.0, 0.0, 1.0]),
        ]
        try:
            with pytest.raises(ValueError, match="exceeds cropped output dimensions"):
                task.run(inputRGB=handles, skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(task, attr):
                    getattr(task, attr).close()


class TestCropModes:
    """End-to-end tests for the ``crop_mode`` options ("none",
    "full_coverage", "data_extent"). Fixtures are fully deterministic (no RNG).
    """

    @staticmethod
    def _projections(skymap):
        ref_proj = SkyProjection.from_legacy(
            skymap[0].getWcs(),
            TractFrame(skymap="test", tract=0, bbox=Box.from_legacy(skymap[0].bbox)),
        )
        patch_proj = SkyProjection.from_legacy(
            skymap[1].getWcs(),
            TractFrame(skymap="test", tract=1, bbox=Box.from_legacy(skymap[1].bbox)),
        )
        return ref_proj, patch_proj

    @pytest.mark.parametrize("crop_mode", ["full_coverage", "data_extent"])
    def test_no_data_raises_when_cropping(self, crop_mode):
        """An input whose pixels are all NaN leaves no weighted pixel: both
        crop modes must fail with a clear ValueError instead of emitting an
        all-black mosaic (crop_mode="none" keeps the old no-crop behavior).
        """
        skymap = _make_skymap()
        bbox = skymap[0][0].getOuterBBox()
        nan_rgb = np.full((bbox.getHeight(), bbox.getWidth(), 3), np.nan, dtype=np.float32)
        handles = [
            InMemoryDatasetHandle(
                inMemoryDataset=ColorImage(nan_rgb, bbox=Box.from_legacy(bbox)),
                skymap="test",
                tract=0,
                patch=0,
            ),
        ]
        task = _make_run_task(reference_tract=0, crop_mode=crop_mode)
        try:
            with pytest.raises(ValueError, match="no pixels that received data"):
                task.run(inputRGB=handles, skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(task, attr):
                    getattr(task, attr).close()

    def test_data_extent_retains_outlier(self):
        """data_extent keeps data that full_coverage-style dense cropping would
        leave stranded: an outlier foreign patch separated from the reference
        by gap cells must remain inside the output raster, non-black.
        """
        skymap = _make_skymap()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),  # reference tract: red
            _make_handle(skymap, 1, 5, [0.0, 0.0, 1.0]),  # foreign tract: blue outlier
        ]
        task = _make_run_task(reference_tract=0, crop_mode="data_extent")
        try:
            result = task.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(task, attr):
                    getattr(task, attr).close()
        image = result.outputRGBMosaic
        array = image.array
        assert np.all(np.isfinite(array))

        # The mosaic covers the outlier: its mapped center is inside the
        # output box, and a sample of far-right outlier pixels is non-black.
        ref_proj, patch_proj = self._projections(skymap)
        mapped = task._map_patch_box_to_ref(patch_proj, skymap[1][5].getOuterBBox(), ref_proj)
        out_box = image.bbox.to_legacy()
        assert out_box.contains(Point2I(mapped.getCenter()))
        assert np.any(array[..., -100:, 2] > 0.5)
        # The reference data is still intact at its 1:1 location.
        ref_bbox = skymap[0][0].getOuterBBox()
        ax = int(ref_bbox.getCenterX() - out_box.getMinX())
        ay = int(ref_bbox.getCenterY() - out_box.getMinY())
        np.testing.assert_allclose(array[ay, ax, :], [1.0, 0.0, 0.0], atol=1e-2)

    def test_data_extent_contains_full_coverage(self):
        """The data_extent rectangle must contain the full_coverage one for
        the same inputs (data_extent trims no more than full_coverage).
        """
        skymap = _make_skymap()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),
            _make_handle(skymap, 1, 5, [0.0, 0.0, 1.0]),
        ]
        boxes = {}
        for mode in ("full_coverage", "data_extent"):
            task = _make_run_task(reference_tract=0, crop_mode=mode)
            try:
                result = task.run(inputRGB=list(handles), skyMap=skymap)
            finally:
                for attr in ("imageHandle", "weightHandle"):
                    if hasattr(task, attr):
                        getattr(task, attr).close()
            boxes[mode] = result.outputRGBMosaic.bbox.to_legacy()
        assert boxes["data_extent"].contains(boxes["full_coverage"])

    def test_data_extent_interior_hole_and_data_edges_survive(self):
        """data_extent trims only entirely empty outer rows/columns: the
        output bbox starts/ends at the first/last row/column containing any
        data, and interior black holes survive inside the raster.
        """
        skymap = _make_skymap()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),  # reference tract: red
            _make_handle(skymap, 1, 0, [0.0, 0.0, 1.0]),  # foreign tract: blue
        ]
        task = _make_run_task(reference_tract=0, crop_mode="data_extent")
        try:
            result = task.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(task, attr):
                    getattr(task, attr).close()
        image = result.outputRGBMosaic
        array = image.array
        assert np.all(np.isfinite(array))

        # Edge-only trimming: expected origin/shape derived independently from
        # an uncropped run of the same inputs, by first/last rows/columns with
        # any non-black pixel (the solid colors normalize to 1.0 in at least
        # one channel wherever weight > 0; empty pixels are exactly 0).
        none_task = _make_run_task(reference_tract=0, crop_mode="none")
        try:
            none_res = none_task.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(none_task, attr):
                    getattr(none_task, attr).close()
        none_box = none_res.outputRGBMosaic.bbox.to_legacy()
        nonblack = np.any(np.asarray(none_res.outputRGBMosaic.array) > 0.5, axis=-1)
        rr = np.flatnonzero(nonblack.any(axis=1))
        cc = np.flatnonzero(nonblack.any(axis=0))
        expected = Box2I(
            Point2I(none_box.getMinX() + int(cc[0]), none_box.getMinY() + int(rr[0])),
            Extent2I(int(cc[-1]) - int(cc[0]) + 1, int(rr[-1]) - int(rr[0]) + 1),
        )
        assert image.bbox.to_legacy() == expected

        # Interior holes: strictly-interior black pixels survive, and the
        # rows/columns through them still cross real data (so they were
        # legitimately kept, not trimmable outer rows/columns).
        interior_black = np.all(array[5:-5, 5:-5] == 0.0, axis=-1)
        assert interior_black.any()
        hole_y, hole_x = (np.argwhere(interior_black)[0] + 5).astype(int)
        assert np.any(array[hole_y] > 0.5)
        assert np.any(array[:, hole_x] > 0.5)

    def test_data_extent_with_binning(self):
        """data_extent and binning compose: crop to the data extent first,
        then halve the cropped rectangle (rounded down).
        """
        skymap = _make_skymap()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),
            _make_handle(skymap, 1, 5, [0.0, 0.0, 1.0]),
        ]
        base = _make_run_task(reference_tract=0, crop_mode="data_extent")
        try:
            full_res = base.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(base, attr):
                    getattr(base, attr).close()
        crop_h, crop_w = full_res.outputRGBMosaic.array.shape[:2]

        binned = _make_run_task(reference_tract=0, bin_factor=2, crop_mode="data_extent")
        try:
            result = binned.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(binned, attr):
                    getattr(binned, attr).close()
        image = result.outputRGBMosaic
        array = image.array
        assert np.all(np.isfinite(array))
        # Cropped-binned dims are half (floored) of the cropped full-res dims,
        # at the binned-box origin (0, 0).
        assert array.shape[1] == crop_w // 2
        assert array.shape[0] == crop_h // 2
        binned_box = image.bbox.to_legacy()
        assert binned_box.getMinX() == 0 and binned_box.getMinY() == 0
        assert binned_box.getWidth() == crop_w // 2
        assert binned_box.getHeight() == crop_h // 2
        # Both tracts' colors survive crop+bin.
        assert np.any(array[..., 2] > 0.5)
        assert np.any(array[..., 0] > 0.5)

    def test_full_coverage_drops_outlier_cluster(self):
        """full_coverage (the default) crops to fully covered virtual-patch
        cells only: a far outlier patch whose warped parallelogram leaves
        holes in its cells falls outside the crop entirely.
        """
        skymap = _make_skymap()
        ref_bbox = skymap[0][0].getOuterBBox()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),  # reference tract: red
            _make_handle(skymap, 1, 7, [0.0, 0.0, 1.0]),  # foreign tract: far outlier cluster
        ]
        task = _make_run_task(reference_tract=0)  # crop_mode defaults to "full_coverage"
        try:
            result = task.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(task, attr):
                    getattr(task, attr).close()
        image = result.outputRGBMosaic
        array = image.array

        # (a) The outlier blue cluster is gone: no strong blue anywhere.
        assert array[..., 2].max() < 0.5
        # (b) The crop is NaN-free and never black: finite everywhere with
        # every pixel carrying (red) data.
        assert np.all(np.isfinite(array))
        assert (array.sum(axis=-1) > 0).all()
        # (c) The reference patch survives 1:1 at its center.
        out_box = image.bbox.to_legacy()
        ax = int(ref_bbox.getCenterX() - out_box.getMinX())
        ay = int(ref_bbox.getCenterY() - out_box.getMinY())
        np.testing.assert_allclose(array[ay, ax, :], [1.0, 0.0, 0.0], atol=1e-2)
        # (d) Strictly smaller than the uncropped raster.
        none_task = _make_run_task(reference_tract=0, crop_mode="none")
        try:
            none_res = none_task.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(none_task, attr):
                    getattr(none_task, attr).close()
        none_shape = none_res.outputRGBMosaic.array.shape
        assert array.shape[0] < none_shape[0]
        assert array.shape[1] < none_shape[1]

    def test_full_coverage_retains_fully_covered_region(self):
        """The standard adjacent two-tract fixture keeps both tracts inside
        the full_coverage crop (the foreign parallelogram overlaps the
        reference cell, which is the fully covered region), and it excludes
        the inflated corner cells (strictly smaller than "none").
        """
        skymap = _make_skymap()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),  # reference tract: red
            _make_handle(skymap, 1, 0, [0.0, 0.0, 1.0]),  # foreign tract: blue, adjacent
        ]
        task = _make_run_task(reference_tract=0)  # crop_mode defaults to "full_coverage"
        try:
            result = task.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(task, attr):
                    getattr(task, attr).close()
        image = result.outputRGBMosaic
        array = image.array
        assert np.all(np.isfinite(array))
        # Fully covered: no all-black pixel.
        assert (array.sum(axis=-1) > 0).all()
        # Both tracts present: red dominates the reference interior, and the
        # blue channel dominates where the foreign patch overrides the
        # reference feather weights.
        assert np.any(array[..., 0] > 0.5)
        assert np.any(array[..., 2] > 0.5)
        # Strictly smaller than the uncropped rectangle (inflated corner
        # cells with holes are excluded).
        none_task = _make_run_task(reference_tract=0, crop_mode="none")
        try:
            none_res = none_task.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(none_task, attr):
                    getattr(none_task, attr).close()
        none_shape = none_res.outputRGBMosaic.array.shape
        assert array.shape[0] < none_shape[0]
        assert array.shape[1] < none_shape[1]

    def test_full_coverage_with_binning(self):
        """full_coverage + bin_factor=2: crop to the fully covered cell first,
        then halve the cropped rectangle (rounded down).
        """
        skymap = _make_skymap()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),
            _make_handle(skymap, 1, 7, [0.0, 0.0, 1.0]),
        ]
        base = _make_run_task(reference_tract=0)
        try:
            full_res = base.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(base, attr):
                    getattr(base, attr).close()
        crop_h, crop_w = full_res.outputRGBMosaic.array.shape[:2]

        binned = _make_run_task(reference_tract=0, bin_factor=2)
        try:
            result = binned.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(binned, attr):
                    getattr(binned, attr).close()
        image = result.outputRGBMosaic
        array = image.array
        assert np.all(np.isfinite(array))
        assert array.shape[1] == crop_w // 2
        assert array.shape[0] == crop_h // 2
        binned_box = image.bbox.to_legacy()
        assert binned_box.getMinX() == 0 and binned_box.getMinY() == 0
        assert binned_box.getWidth() == crop_w // 2
        assert binned_box.getHeight() == crop_h // 2
        assert (array.sum(axis=-1) > 0).all()
        assert np.any(array[..., 0] > 0.5)

    def test_full_coverage_bin_exceeds_crop_raises(self):
        """A bin factor beyond the full_coverage crop dims raises the clear
        ValueError from run (same guard as the other crop modes).
        """
        skymap = _make_skymap()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),
            _make_handle(skymap, 1, 7, [0.0, 0.0, 1.0]),
        ]
        task = _make_run_task(reference_tract=0, bin_factor=2000)
        try:
            with pytest.raises(ValueError, match="bin_factor"):
                task.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(task, attr):
                    getattr(task, attr).close()


def _cell_grid_weight(grid, cell=10):
    """Render a boolean ``(ncy, ncx)`` cell-validity grid into a weight array
    with pitch == cell == ``cell`` (cells exactly tile the canvas).
    """
    ncy, ncx = grid.shape
    weight = np.zeros((ncy * cell, ncx * cell), dtype=np.float32)
    for cj in range(ncy):
        for ci in range(ncx):
            if grid[cj, ci]:
                weight[cj * cell : (cj + 1) * cell, ci * cell : (ci + 1) * cell] = 1.0
    return weight


def _brute_force_max_rect(grid, ncx, ncy, ref_cell):
    """Brute-force cell-rectangle search with the documented ordering:
    max area, prefer containing ref_cell, then smaller j, then smaller i."""
    best = None
    for ci0 in range(ncx):
        for ci1 in range(ci0, ncx):
            for cj0 in range(ncy):
                for cj1 in range(cj0, ncy):
                    if not grid[cj0 : cj1 + 1, ci0 : ci1 + 1].all():
                        continue
                    candidate = (ci0, ci1, cj0, cj1)
                    if best is None:
                        best = candidate
                        continue
                    bi0, bi1, bj0, bj1 = best
                    area = (ci1 - ci0 + 1) * (cj1 - cj0 + 1)
                    best_area = (bi1 - bi0 + 1) * (bj1 - bj0 + 1)
                    if area != best_area:
                        if area > best_area:
                            best = candidate
                        continue
                    if ref_cell is not None:
                        has_ref = ci0 <= ref_cell[0] <= ci1 and cj0 <= ref_cell[1] <= cj1
                        best_has_ref = bi0 <= ref_cell[0] <= bi1 and bj0 <= ref_cell[1] <= bj1
                        if has_ref != best_has_ref:
                            if has_ref:
                                best = candidate
                            continue
                    if cj0 != bj0:
                        if cj0 < bj0:
                            best = candidate
                        continue
                    if ci0 < bi0:
                        best = candidate
    return best


class TestFullCoverageBBox:
    """Unit tests for :meth:`MetaMosaicTask._full_coverage_bbox` on synthetic
    weight arrays with explicit (pitch == cell) cell geometry."""

    @staticmethod
    def _bbox(grid, cell=10, ref_cell=None):
        ncy, ncx = grid.shape
        task = _make_run_task()
        return task._full_coverage_bbox(
            (ncy * cell, ncx * cell),
            _cell_grid_weight(grid, cell),
            pitch_x=cell,
            pitch_y=cell,
            cell_w=cell,
            cell_h=cell,
            ncx=ncx,
            ncy=ncy,
            ref_cell=ref_cell,
        )

    def test_hole_carves_out_cells(self):
        """A single hole cell splits the grid; the max rectangle is the two
        intact rows below it (hand-computed).
        """
        grid = np.ones((3, 3), dtype=bool)
        grid[0, 1] = False  # hole in the top-middle cell
        bbox = self._bbox(grid)
        # Rows 1..2, cols 0..2 fully valid -> pixels x [0, 29], y [10, 29].
        assert bbox == Box2I(Point2I(0, 10), Extent2I(30, 20))

    def test_no_valid_cell_returns_none(self):
        assert self._bbox(np.zeros((2, 2), dtype=bool)) is None

    def test_tie_prefers_rectangle_containing_ref_cell(self):
        # Two disjoint 2x2 blocks of equal area; ref_cell sits inside the
        # lower-right one, which must win over the (otherwise later) winner.
        grid = np.zeros((3, 4), dtype=bool)
        grid[0:2, 0:2] = True
        grid[1:3, 2:4] = True
        expected = Box2I(Point2I(20, 10), Extent2I(20, 20))  # cells (2..3, 1..2)
        assert self._bbox(grid, ref_cell=(3, 2)) == expected
        # Without the reference anchor the tie falls to smaller j, then i:
        # the upper-left block (j_min = 0).
        assert self._bbox(grid, ref_cell=None) == Box2I(Point2I(0, 0), Extent2I(20, 20))

    def test_tie_falls_back_to_leftmost(self):
        # Two equal-area 1x2 strips in the same row, separated by an invalid
        # cell: smaller i wins.
        grid = np.zeros((2, 5), dtype=bool)
        grid[0, 0:2] = True
        grid[0, 3:5] = True
        assert self._bbox(grid, ref_cell=None) == Box2I(Point2I(0, 0), Extent2I(20, 10))

    @pytest.mark.parametrize("shape, seed", [((4, 5), 42), ((3, 6), 43), ((5, 4), 44)])
    def test_random_patterns_match_brute_force(self, shape, seed):
        """The monotone-stack search must equal a brute-force rectangle search
        (with the documented tie-break) on seeded random cell patterns."""
        ncy, ncx = shape
        rng = np.random.default_rng(seed)
        grid = rng.random((ncy, ncx)) > 0.35
        if not grid.any():
            pytest.skip("degenerate seed")
        ref_cell = (ncx // 2, ncy // 2) if seed == 42 else None
        bbox = self._bbox(grid, ref_cell=ref_cell)
        rect = _brute_force_max_rect(grid, ncx, ncy, ref_cell)
        ci0, ci1, cj0, cj1 = rect
        assert bbox == Box2I(
            Point2I(ci0 * 10, cj0 * 10),
            Extent2I((ci1 - ci0) * 10 + 10, (cj1 - cj0) * 10 + 10),
        )


class TestDataBBox:
    """Unit tests for the band-wise data-extent scan on the weight array."""

    def test_known_rectangle(self):
        task = _make_run_task()
        weight = np.zeros((100, 80), dtype=np.float32)
        weight[20:35, 10:30] = 1.0
        bbox = task._data_bbox((100, 80), weight)
        assert bbox == Box2I(Point2I(10, 20), Point2I(29, 34))

    def test_banded_scan_matches(self):
        """Small bands straddling the data must give the same box as one band."""
        task = _make_run_task()
        weight = np.zeros((50, 60), dtype=np.float32)
        weight[12:17, 40:55] = 0.5
        expected = task._data_bbox((50, 60), weight)
        assert expected == Box2I(Point2I(40, 12), Point2I(54, 16))
        for band_rows in (1, 5, 13, 60, 1000):
            assert task._data_bbox((50, 60), weight, band_rows=band_rows) == expected

    def test_scattered_pixels_enclosed(self):
        task = _make_run_task()
        weight = np.zeros((40, 40), dtype=np.float32)
        weight[3, 5] = 1.0
        weight[30, 37] = 1.0
        weight[15, 2] = 1e-17  # tiny but strictly positive weight
        bbox = task._data_bbox((40, 40), weight)
        assert bbox == Box2I(Point2I(2, 3), Point2I(37, 30))

    @pytest.mark.parametrize("mode", ["full_coverage", "data_extent"])
    def test_all_zero_returns_none(self, mode):
        task = _make_run_task()
        assert task._data_bbox((50, 60), np.zeros((50, 60), dtype=np.float32), mode) is None
        # Zero weight in a band beyond the rows scanned so far is still None.
        assert (
            task._data_bbox((50, 60), np.zeros((50, 60), dtype=np.float32), mode, band_rows=7)
            is None
        )


def _wcs(cd=None, crpix=(0.0, 0.0), ra=10.0, dec=-30.0):
    """Small helper: a plain TAN SkyWcs at (ra, dec)."""
    if cd is None:
        cd = np.array([[-1.0e-4, 0.0], [0.0, 1.0e-4]])
    return makeSkyWcs(
        crpix=Point2D(*crpix),
        crval=SpherePoint(ra, dec, degrees),
        cdMatrix=cd,
    )


class TestWarpMaps:
    """Unit tests for the shared warp_maps (WCS -> cv2.remap coordinate maps)
    helper."""

    def test_identity_wcs_pair_gives_pixel_grid(self):
        wcs = _wcs()
        box = Box2I(Point2I(0, 0), Extent2I(32, 16))
        map_x, map_y = warp_maps(wcs, box, wcs, box)
        yy, xx = np.mgrid[0:16, 0:32]
        assert map_x.dtype == np.float32 and map_y.dtype == np.float32
        assert map_x.shape == (16, 32) and map_y.shape == (16, 32)
        np.testing.assert_allclose(map_x, xx, atol=1e-3)
        np.testing.assert_allclose(map_y, yy, atol=1e-3)

    def test_maps_measured_from_src_bbox_min(self):
        """Map values are source pixel indices measured from the src_bbox min
        (cv2 array convention), even for negative origins."""
        wcs = _wcs()
        src_box = Box2I(Point2I(-10, -5), Extent2I(64, 40))
        dest_box = Box2I(Point2I(0, 0), Extent2I(32, 16))
        map_x, map_y = warp_maps(wcs, src_box, wcs, dest_box)
        yy, xx = np.mgrid[0:16, 0:32]
        np.testing.assert_allclose(map_x, xx + 10, atol=1e-3)
        np.testing.assert_allclose(map_y, yy + 5, atol=1e-3)

    def test_unreachable_dest_pixels_are_nan(self):
        """A dest pixel whose sky position is unreachable from the source
        (far side of the sphere) keeps NaN map entries."""
        src_wcs = _wcs(ra=0.0, dec=89.0)
        dest_wcs = _wcs(ra=180.0, dec=-89.0)
        src_box = Box2I(Point2I(0, 0), Extent2I(10, 10))
        dest_box = Box2I(Point2I(0, 0), Extent2I(10, 10))
        map_x, map_y = warp_maps(src_wcs, src_box, dest_wcs, dest_box)
        assert np.isnan(map_x).all() and np.isnan(map_y).all()

    def test_nan_contract_via_remap(self):
        """Dest pixels outside the source coverage remap to NaN with
        borderValue=np.nan (the NaN-as-missing-data contract)."""
        wcs = _wcs()
        src_box = Box2I(Point2I(0, 0), Extent2I(32, 16))
        src = np.full((16, 32), 0.25, dtype=np.float32)
        # Dest box 50 px further along +x than the source: every dest pixel
        # lands beyond the source array (map x in [50, 81] > 31).
        map_x, map_y = warp_maps(wcs, src_box, wcs, Box2I(Point2I(50, 0), Extent2I(32, 16)))
        out = cv2.remap(
            src, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan
        )
        assert np.isnan(out).all()
        # And when part of the dest box does overlap the source, only the
        # overlapping part carries data.
        map_x, map_y = warp_maps(wcs, src_box, wcs, Box2I(Point2I(20, 0), Extent2I(32, 16)))
        out = cv2.remap(
            src, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan
        )
        # Map x runs from 20 to 51: columns 0..11 sample source pixels 20..31
        # (source x range is [0, 31]), columns >= 12 are fully outside it.
        # The Lanczos4 kernel reaches 4 px, so a pixel is finite only when
        # its map coordinate is >= 4 from every source-array edge (and NaN
        # wherever a tap would fall outside it - the borderValue contract).
        assert np.isfinite(out[4:12, :8]).all()
        np.testing.assert_allclose(out[4, 4], 0.25, atol=1e-6)
        assert np.isnan(out[:, 12:]).all()


def _backends_agree_mask(skymap, ref_tract, tract_id, patches, out_box, margin=10):
    """Pixels where the "afw" and "cv2" warp backends must agree to full
    precision.

    The two backends use different resampling kernels (lanczos5 vs
    Lanczos4), so values may legitimately differ within a narrow ring around
    each warped footprint (NaN band widths and kernel truncation differ at
    the source-array edge). This mask is True everywhere except ``margin``
    pixels around each foreign patch's warped footprint boundary (where the
    map coordinate is within ``margin`` of leaving the patch array, or is
    NaN).
    """
    ref_proj = SkyProjection.from_legacy(
        skymap[ref_tract].getWcs(),
        TractFrame(skymap="test", tract=ref_tract, bbox=Box.from_legacy(skymap[ref_tract].bbox)),
    )
    good = np.ones((out_box.getHeight(), out_box.getWidth()), dtype=bool)
    task = _make_run_task()
    for patch in patches:
        patch_info = skymap[tract_id][patch]
        patch_proj = SkyProjection.from_legacy(
            patch_info.getWcs(),
            TractFrame(skymap="test", tract=tract_id, bbox=Box.from_legacy(skymap[tract_id].bbox)),
        )
        mapped = task._map_patch_box_to_ref(patch_proj, patch_info.getOuterBBox(), ref_proj)
        db = Box2I(
            Point2I(mapped.getMinX() - 2, mapped.getMinY() - 2),
            Point2I(mapped.getMaxX() + 2, mapped.getMaxY() + 2),
        )
        map_x, map_y = warp_maps(
            patch_info.getWcs(), skymap[tract_id].bbox, skymap[ref_tract].getWcs(), db
        )
        pb = patch_info.getOuterBBox()
        tb = skymap[tract_id].bbox
        lo_x, hi_x = pb.getMinX() - tb.getMinX(), pb.getMaxX() - tb.getMinX()
        lo_y, hi_y = pb.getMinY() - tb.getMinY(), pb.getMaxY() - tb.getMinY()
        bad = (
            ~np.isfinite(map_x)
            | ~np.isfinite(map_y)
            | (map_x < lo_x + margin)
            | (map_x > hi_x - margin)
            | (map_y < lo_y + margin)
            | (map_y > hi_y - margin)
        )
        x0 = db.getMinX() - out_box.getMinX()
        y0 = db.getMinY() - out_box.getMinY()
        gx0, gy0 = max(0, x0), max(0, y0)
        gx1 = min(good.shape[1], x0 + bad.shape[1])
        gy1 = min(good.shape[0], y0 + bad.shape[0])
        good[gy0:gy1, gx0:gx1] &= ~bad[gy0 - y0 : gy1 - y0, gx0 - x0 : gx1 - x0]
    return good


class TestWarpBackends:
    """End-to-end equivalence of the "afw" (reference) and "cv2"
    (precomputed-map) warp backends, plus the phase-timing log."""

    @staticmethod
    def _run(skymap, backend, crop_mode, handles):
        task = _make_run_task(reference_tract=0, crop_mode=crop_mode)
        task.config.warp_backend = backend
        try:
            return task.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(task, attr):
                    getattr(task, attr).close()

    def _standard_handles(self, skymap):
        return [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),  # reference tract: red
            _make_handle(skymap, 1, 0, [0.0, 0.0, 1.0]),  # foreign tract: blue
        ]

    @pytest.mark.parametrize("crop_mode", ["none", "full_coverage"])
    def test_cv2_matches_afw(self, crop_mode):
        """cv2.remap (Lanczos4) output agrees with the afwMath reference to
        5e-3 outside a narrow kernel-ring around the foreign footprint, for
        both an uncropped and the default-cropped raster (crop/feather
        interactions included: identical crop boxes, feather weights are
        backend-independent)."""
        skymap = _make_skymap()
        afw = self._run(skymap, "afw", crop_mode, self._standard_handles(skymap)).outputRGBMosaic
        cv2r = self._run(skymap, "cv2", crop_mode, self._standard_handles(skymap)).outputRGBMosaic
        assert afw.bbox == cv2r.bbox
        a, b = afw.array, cv2r.array
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        # The foreign color is present (not silently dropped) in both.
        assert np.any(a[..., 2] > 0.5) and np.any(b[..., 2] > 0.5)
        good = _backends_agree_mask(skymap, 0, 1, [0], cv2r.bbox.to_legacy())
        assert good.mean() > 0.9
        np.testing.assert_allclose(a[good], b[good], atol=5e-3)
        # And a strong-agreement majority everywhere (the excluded ring is
        # only a small band around the warped footprint edges).
        diff = np.abs(a - b).max(axis=-1)
        assert (diff > 5e-3).mean() < 0.05

    def test_data_extent_bbox_agrees_within_kernel_width(self):
        """data_extent crops to the outermost row/column with any weight; the
        two backends' warp NaN bands differ by at most a couple of pixels
        (kernel radius), so their crop boxes must agree within 3 px."""
        skymap = _make_skymap()
        afw = self._run(skymap, "afw", "data_extent", self._standard_handles(skymap))
        cv2r = self._run(skymap, "cv2", "data_extent", self._standard_handles(skymap))
        ba = afw.outputRGBMosaic.bbox.to_legacy()
        bb = cv2r.outputRGBMosaic.bbox.to_legacy()
        for get in (lambda b: b.getMinX(), lambda b: b.getMaxX(), lambda b: b.getMinY(),
                    lambda b: b.getMaxY()):
            assert abs(get(ba) - get(bb)) <= 3

    def test_nan_hole_black_after_finalize_both_backends(self):
        """NaN contract: a NaN block in a foreign patch yields zero-weight
        (black after finalize) pixels for both backends, and the black
        regions agree up to the kernel ring (the hole exists and is at the
        same place in both)."""
        skymap = _make_skymap()
        ref_bbox = skymap[0][0].getOuterBBox()
        blue = np.empty((ref_bbox.getHeight(), ref_bbox.getWidth(), 3), dtype=np.float32)
        blue[...] = [0.0, 0.0, 1.0]
        blue[:10, :10, :] = np.nan
        foreign = InMemoryDatasetHandle(
            inMemoryDataset=ColorImage(blue, bbox=Box.from_legacy(skymap[1][0].getOuterBBox())),
            skymap="test",
            tract=1,
            patch=0,
        )
        handles = [_make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]), foreign]

        zeros = {}
        for backend in ("afw", "cv2"):
            image = self._run(skymap, backend, "none", handles).outputRGBMosaic
            array = image.array
            assert np.all(np.isfinite(array))
            zeros[backend] = np.all(array == 0.0, axis=-1)

        def dilate(m):
            return cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        # Both backends blacken the hole (and the empty canvas): non-trivial
        # amount of black pixels...
        assert zeros["afw"].sum() > 1000 and zeros["cv2"].sum() > 1000
        # ... and their black sets agree modulo a 3-pixel kernel ring.
        assert (zeros["afw"] & ~dilate(zeros["cv2"])).sum() == 0
        assert (zeros["cv2"] & ~dilate(zeros["afw"])).sum() == 0

    def test_timings_log_and_map_build_cache(self, caplog):
        """The run always ends with a 'Timings:' line; with the cv2 backend
        one coordinate map is built per foreign TRACT (two foreign patches of
        tract 1 share one map build), and the afw backend builds none."""
        skymap = _make_skymap()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),
            _make_handle(skymap, 1, 0, [0.0, 0.0, 1.0]),
            _make_handle(skymap, 1, 1, [0.0, 1.0, 0.0]),
        ]
        for backend, expected_maps in (("cv2", 1), ("afw", 0)):
            with caplog.at_level(logging.INFO, logger="lsst.metaMosaic"):
                self._run(skymap, backend, "none", handles)
            lines = [r.getMessage() for r in caplog.records if "Timings:" in r.getMessage()]
            assert lines, "no Timings log line emitted"
            line = lines[-1]
            match = re.search(r"map-build=[0-9.]+s \(n=(\d+)\)", line)
            assert match is not None, line
            assert int(match.group(1)) == expected_maps
            match = re.search(r"decode=[0-9.]+s \(n=(\d+)\)", line)
            assert int(match.group(1)) == 3
            match = re.search(r"warp=[0-9.]+s \(n=(\d+)\)", line)
            assert int(match.group(1)) == 2
            assert "accumulate=" in line and "finalize=" in line and "crop=" in line

    @pytest.mark.parametrize("crop_mode", ["none", "full_coverage"])
    def test_rust_matches_afw(self, crop_mode):
        """The "rust" backend (fitted-transform warp_into, Lanczos4) agrees
        with the afwMath (Lanczos5) reference to 5e-3 outside the same kernel
        ring the cv2 battery uses, uncropped and default-cropped. The
        tolerance is carried from the cv2 battery (same kernel family, same
        flat-color scenes); measured masked max-diff is 1.6e-5 (the fitted
        transform's verified residual is <=0.05 source px), so this is a 300x
        margin, not a loosening."""
        skymap = _make_skymap()
        afw = self._run(skymap, "afw", crop_mode, self._standard_handles(skymap)).outputRGBMosaic
        rust = self._run(skymap, "rust", crop_mode, self._standard_handles(skymap)).outputRGBMosaic
        assert afw.bbox == rust.bbox
        a, b = afw.array, rust.array
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        assert np.any(a[..., 2] > 0.5) and np.any(b[..., 2] > 0.5)
        good = _backends_agree_mask(skymap, 0, 1, [0], rust.bbox.to_legacy())
        assert good.mean() > 0.9
        np.testing.assert_allclose(a[good], b[good], atol=5e-3)
        diff = np.abs(a - b).max(axis=-1)
        assert (diff > 5e-3).mean() < 0.05

    def test_rust_nan_hole_black_after_finalize(self):
        """NaN contract for "rust": a NaN block in a foreign patch yields
        zero-weight (black after finalize) pixels, agreeing with the afw
        reference up to a 3-pixel kernel ring (same assertion shape as the
        cv2/afw test; the masked-add contract lives in the shared
        FeatheredMosaicCreator, so only the warp's NaN footprint differs)."""
        skymap = _make_skymap()
        ref_bbox = skymap[0][0].getOuterBBox()
        blue = np.empty((ref_bbox.getHeight(), ref_bbox.getWidth(), 3), dtype=np.float32)
        blue[...] = [0.0, 0.0, 1.0]
        blue[:10, :10, :] = np.nan
        foreign = InMemoryDatasetHandle(
            inMemoryDataset=ColorImage(blue, bbox=Box.from_legacy(skymap[1][0].getOuterBBox())),
            skymap="test",
            tract=1,
            patch=0,
        )
        handles = [_make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]), foreign]

        zeros = {}
        for backend in ("afw", "rust"):
            image = self._run(skymap, backend, "none", handles).outputRGBMosaic
            array = image.array
            assert np.all(np.isfinite(array))
            zeros[backend] = np.all(array == 0.0, axis=-1)

        def dilate(m):
            return cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        assert zeros["afw"].sum() > 1000 and zeros["rust"].sum() > 1000
        assert (zeros["afw"] & ~dilate(zeros["rust"])).sum() == 0
        assert (zeros["rust"] & ~dilate(zeros["afw"])).sum() == 0

    def test_timings_log_and_transform_cache(self, caplog, monkeypatch):
        """"rust" fits exactly one WarpTransform per foreign TRACT (two
        patches of tract 1 and one of tract 2 share the tract cache: spy
        count == 2 foreign tracts, not 3 patches), and the Timings line keeps
        the parser format with map-build n=2. map-build times the
        control-grid fits: <1.0 s each on this tiny scene (measured ~0.3-0.4
        s; the guard is machine-loose — the structural anti-pattern pin is
        the spy count and the absence of dense maps, vs the cv2 path's
        303 s/tract dense map build at production scale)."""
        from lsst.pipe.tasks.prettyPictureMaker import _meta_mosaic

        skymap = _make_skymap(tracts=3)
        fits = []
        real_fit = _meta_mosaic.fit_transform

        def spy(*args, **kwargs):
            transform = real_fit(*args, **kwargs)
            fits.append(transform)
            return transform

        monkeypatch.setattr(_meta_mosaic, "fit_transform", spy)
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),
            _make_handle(skymap, 1, 0, [0.0, 0.0, 1.0]),
            _make_handle(skymap, 1, 1, [0.0, 1.0, 0.0]),
            _make_handle(skymap, 2, 0, [0.0, 1.0, 0.0]),
        ]
        with caplog.at_level(logging.INFO, logger="lsst.metaMosaic"):
            self._run(skymap, "rust", "none", handles)
        assert len(fits) == 2, "expected one fit_transform per foreign tract"
        assert all(t.dest_bbox is not None for t in fits)  # D7 always populated
        lines = [r.getMessage() for r in caplog.records if "Timings:" in r.getMessage()]
        assert lines
        line = lines[-1]
        match = re.search(r"map-build=([0-9.]+)s \(n=(\d+)\)", line)
        assert match is not None, line
        assert int(match.group(2)) == 2
        float_per_fit = float(match.group(1)) / 2
        assert float_per_fit < 1.0, f"fit amortization regressed: {float_per_fit:.3f}s"
        match = re.search(r"warp=[0-9.]+s \(n=(\d+)\)", line)
        assert int(match.group(1)) == 3  # one warp_into per foreign patch
        match = re.search(r"decode=[0-9.]+s \(n=(\d+)\)", line)
        assert int(match.group(1)) == 4

    def test_rust_fit_failure_falls_back_to_afw(self, caplog, monkeypatch):
        """A failing fit (e.g. off-domain projection pair) is a per-tract
        robustness event: exactly one warning names the tract, the failed fit
        is cached (two patches -> one attempt, one warning) and the output is
        bit-identical to the afw backend (which is literally the code path
        that served it). map-build counts no fits."""
        from lsst.pipe.tasks.prettyPictureMaker import _meta_mosaic

        skymap = _make_skymap()
        calls = []

        def broken_fit(*args, **kwargs):
            calls.append(args)
            raise ValueError("noninvertible/off-domain projection pair (injected)")

        monkeypatch.setattr(_meta_mosaic, "fit_transform", broken_fit)
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),
            _make_handle(skymap, 1, 0, [0.0, 0.0, 1.0]),
            _make_handle(skymap, 1, 1, [0.0, 1.0, 0.0]),
        ]
        with caplog.at_level(logging.INFO, logger="lsst.metaMosaic"):
            rust = self._run(skymap, "rust", "none", handles)
        warnings = [r for r in caplog.records if "falling back to the afw Warper" in r.getMessage()]
        assert len(warnings) == 1  # once per tract, not per patch
        assert "tract 1" in warnings[0].getMessage()
        assert len(calls) == 1  # failed fit cached; second patch did not retry
        afw = self._run(skymap, "afw", "none", handles)
        assert rust.outputRGBMosaic.bbox == afw.outputRGBMosaic.bbox
        np.testing.assert_array_equal(
            np.asarray(rust.outputRGBMosaic.array), np.asarray(afw.outputRGBMosaic.array)
        )
        line = [r.getMessage() for r in caplog.records if "Timings:" in r.getMessage()][-1]
        match = re.search(r"map-build=[0-9.]+s \(n=(\d+)\)", line)
        assert int(match.group(1)) == 0

    def test_warp_backend_config(self):
        config = MetaMosaicConfig()
        assert config.warp_backend == "afw"
        config.warp_backend = "rust"
        assert config.warp_backend == "rust"
        with pytest.raises(Exception):
            config.warp_backend = "bogus"
