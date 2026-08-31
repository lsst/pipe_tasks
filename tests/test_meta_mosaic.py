import numpy as np

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


def _make_handle(skymap, tract, patch, value):
    """Build a deferred handle returning a constant-color patch of the right size."""
    bbox = skymap[tract][patch].getOuterBBox()
    array = np.empty((bbox.getHeight(), bbox.getWidth(), 3), dtype=np.float32)
    array[...] = value
    image = ColorImage(array, bbox=Box.from_legacy(bbox))
    return InMemoryDatasetHandle(inMemoryDataset=image, skymap="test", tract=tract, patch=patch)


def _make_run_task(reference_tract=0):
    config = MetaMosaicConfig()
    config.reference_tract = reference_tract
    config.bin_factor = 1
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
        weight = np.zeros_like(sum_img)
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

        # Output box must enclose the reference patch and the mapped foreign box.
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
        assert out_box.contains(mapped)

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
        """Two foreign patches overlapping one cell should be combined by nan-mean."""
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
        """A bin factor of 2 must halve each output dimension."""
        skymap = _make_skymap()
        task = _make_run_task(reference_tract=0)
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
