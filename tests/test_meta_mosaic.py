import numpy as np
import pytest

from lsst.geom import Box2I, Extent2I, Point2D, Point2I
from lsst.images import Box, ColorImage, SkyProjection, TractFrame
from lsst.pipe.base import InMemoryDatasetHandle
from lsst.skymap import DiscreteSkyMap

from lsst.pipe.tasks.prettyPictureMaker._meta_mosaic import MetaMosaicConfig, MetaMosaicTask
from lsst.pipe.tasks.prettyPictureMaker._utils import FeatheredMosaicCreator


def _make_task(reference_tract=None):
    config = MetaMosaicConfig()
    config.reference_tract = reference_tract
    task = MetaMosaicTask.__new__(MetaMosaicTask)
    task.config = config
    return task


def _make_skymap():
    """Build a two-tract DiscreteSkyMap with overlapping patch borders.

    Patches overlap their neighbours (outer 640 px vs 400 px pitch) and the
    feathering region (``patch_grow`` = inner cell size = 100 px) is comfortably
    smaller than the pitch, so each patch's center stays pure while the borders
    blend.
    """
    config = DiscreteSkyMap.ConfigClass()
    tract_builder = config.tractBuilder["cells"]
    tract_builder.cellInnerDimensions = [100, 100]
    tract_builder.cellBorder = 20
    tract_builder.numCellsPerPatchInner = 4
    tract_builder.numCellsInPatchBorder = 1
    config.tractBuilder.name = "cells"
    config.raList = [10.0, 10.05]
    config.decList = [-30.0, -30.0]
    config.radiusList = [0.15, 0.15]
    return DiscreteSkyMap(config)


def _make_handle(skymap, tract, patch, value, metadata=None):
    """Build a deferred handle returning a constant-color patch of the right size."""
    bbox = skymap[tract][patch].getOuterBBox()
    array = np.empty((bbox.getHeight(), bbox.getWidth(), 3), dtype=np.float32)
    array[...] = value
    image = ColorImage(array, bbox=Box.from_legacy(bbox), metadata=metadata)
    return InMemoryDatasetHandle(inMemoryDataset=image, skymap="test", tract=tract, patch=patch)


def _make_run_task(reference_tract=0, bin_factor=1, crop_output=True):
    config = MetaMosaicConfig()
    config.reference_tract = reference_tract
    config.bin_factor = bin_factor
    config.crop_output = crop_output
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
        task = _make_run_task(reference_tract=0)
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
        the NaN-reweighted feather (weighted mean where both are valid)."""
        skymap = _make_skymap()
        task = _make_run_task(reference_tract=0)

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

        Cropping is disabled so the base rectangle is the full touched-cells
        grid and the pure binning math is exercised.
        """
        skymap = _make_skymap()
        task = _make_run_task(reference_tract=0, crop_output=False)
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
        config.crop_output = False
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
        """The output is cropped to the data footprint, not the touched-cells
        rectangle: corner cells the foreign patch only inflates into are dropped.
        """
        skymap = _make_skymap()
        ref_bbox = skymap[0][0].getOuterBBox()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),  # reference tract: red
            _make_handle(skymap, 1, 0, [0.0, 0.0, 1.0]),  # foreign tract: blue
        ]

        task = _make_run_task(reference_tract=0)  # crop_output defaults to True
        try:
            cropped = task.run(inputRGB=list(handles), skyMap=skymap)
        finally:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(task, attr):
                    getattr(task, attr).close()

        full_task = _make_run_task(reference_tract=0, crop_output=False)
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
        sits strictly inside the output raster.
        """
        skymap = _make_skymap()
        ref_bbox = skymap[0][0].getOuterBBox()
        handles = [
            _make_handle(skymap, 0, 0, [1.0, 0.0, 0.0]),  # reference tract: red
            _make_handle(skymap, 1, 5, [0.0, 0.0, 1.0]),  # foreign tract: blue, 5 cells away
        ]
        task = _make_run_task(reference_tract=0)
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

    def test_all_zero_returns_none(self):
        task = _make_run_task()
        assert task._data_bbox((50, 60), np.zeros((50, 60), dtype=np.float32)) is None
        # Zero weight in a band beyond the rows scanned so far is still None.
        assert task._data_bbox((50, 60), np.zeros((50, 60), dtype=np.float32), 7) is None
