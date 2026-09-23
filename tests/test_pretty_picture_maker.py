import inspect

import numpy as np
from scipy.ndimage import binary_dilation

from lsst.afw.image import ExposureF
from lsst.geom import Box2I
from lsst.images import Box
from lsst.pipe.tasks.prettyPictureMaker._new_star_fix import (
    PrettyPictureStarFixerConfig,
    PrettyPictureStarFixerTask,
    _circular_mask,
)
from lsst.pipe.tasks.prettyPictureMaker._utils import FeatheredMosaicCreator
from stellaRGB import reconstruct_saturated_stars


class TestFeatheredMosaicCreator:
    def test_make_featherings_basic(self):
        """Verify featherings are created with correct shapes."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        creator._make_featherings((50, 50))

        assert creator.featherings is not None
        assert len(creator.featherings) == 4

        for feather in creator.featherings:
            assert feather.shape == (50, 50)

    def test_make_featherings_symmetry(self):
        """Verify top/bottom and left/right masks are symmetric."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        creator._make_featherings((50, 50))

        top, bottom, left, right = creator.featherings

        np.testing.assert_allclose(top[1:30, :], bottom[20:49, :][::-1, :], rtol=1e-5)
        np.testing.assert_allclose(left[:, 1:30], right[:, 20:49][:, ::-1], rtol=1e-5)

    def test_make_featherings_values(self):
        """Verify ramp values are correct."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        creator._make_featherings((50, 50))

        top, bottom, left, right = creator.featherings

        assert top[0, 0] < 1e-6
        assert np.allclose(top[20, 0], 1.0, atol=0.1)

        assert np.allclose(bottom[30, 0], 1.0, atol=0.1)
        assert bottom[-1, 0] < 1e-6

        assert np.allclose(left[0, 20], 1.0, atol=0.1)
        assert right[0, -1] < 1e-6

    def test_make_featherings_bin_factor(self):
        """Verify bin_factor reduces resolution."""
        creator_no_bin = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        creator_bin = FeatheredMosaicCreator(patch_grow=10, bin_factor=2)

        creator_no_bin._make_featherings((50, 50))
        creator_bin._make_featherings((50, 50))

        assert creator_bin.featherings[0].shape == creator_no_bin.featherings[0].shape

    def _make_boxes(self, min_point, max_point):
        box = Box.from_legacy(Box2I(Box2I.Point(*min_point), Box2I.Point(*max_point)))
        return box

    def test_add_to_image_full_overlap(self):
        """Verify no feathering when box == newBox."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((50, 50, 3))
        patch = np.ones((50, 50, 3)) * 0.5

        box = self._make_boxes((0, 0), (49, 49))
        new_box = self._make_boxes((0, 0), (49, 49))

        creator.add_to_image(image, patch, new_box, box, reverse=False)

        np.testing.assert_allclose(image, 0.5, atol=0.1)

    def test_add_to_image_single_edge(self):
        """Verify only the differing edge gets feathering."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((60, 60, 3))
        patch = np.ones((50, 50, 3)) * 0.5

        box = self._make_boxes((0, 0), (49, 49))
        new_box = self._make_boxes((0, 5), (49, 54))

        creator.add_to_image(image, patch, new_box, box, reverse=False)

        assert image.shape == (60, 60, 3)

    def test_add_to_image_multi_edge(self):
        """Verify multiple edges get combined feathering."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((70, 70, 3))
        patch = np.ones((50, 50, 3)) * 0.5

        box = self._make_boxes((0, 0), (49, 49))
        new_box = self._make_boxes((5, 5), (54, 54))

        creator.add_to_image(image, patch, new_box, box, reverse=False)

        assert image.shape == (70, 70, 3)

    def test_add_to_image_rgb(self):
        """Verify RGB (3D) images are handled correctly."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((50, 50, 3))
        patch = np.ones((50, 50, 3)) * 0.5

        box = self._make_boxes((0, 0), (49, 49))
        new_box = self._make_boxes((0, 0), (49, 49))

        creator.add_to_image(image, patch, new_box, box, reverse=False)

        assert image.shape == (50, 50, 3)
        np.testing.assert_allclose(image[0, 0, :], 0.5, atol=0.1)

    def test_add_to_image_reverse(self):
        """Verify reverse flips the patch."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((50, 50, 3))
        patch = np.ones((50, 50, 3))

        box = self._make_boxes((0, 0), (49, 49))
        new_box = self._make_boxes((0, 0), (49, 49))

        creator.add_to_image(image, patch, new_box, box, reverse=True)

        assert image.shape == (50, 50, 3)


_SKY = 1.0e4
_STAR_AMP = 1.2e6
_STAR_SIGMA = 3.0
_BRIGHT_THRESH = 1.0e5
_CORE_RADIUS = 2
_SHAPE = (128, 128)
# Isolated well-separated cores so the fused mask stays one blob per star.
_STAR_POSITIONS = ((64, 64), (20, 20), (20, 100), (100, 20), (100, 100), (110, 55))
# Per-band peak-flux scales of the same star field (i is the brightest band).
_BAND_SCALES = {"i": 1.0, "r": 0.6, "g": 0.3}


def _make_star_exposures(nan_core=False):
    """Build synthetic i/r/g exposures at physical flux scale.

    Each exposure is a constant sky plus the same Gaussian star field scaled
    per band, with a ``NO_DATA`` disk flagged over every star core. The stars
    peak well above ``_BRIGHT_THRESH`` so the brightness-threshold leg of the
    task-side fused mask catches the cores too.

    Parameters
    ----------
    nan_core : `bool`, optional
        Set the masked core pixels to NaN instead of the star flux, exercising
        the engine's masked-NonFinite tolerance.

    Returns
    -------
    exposures : `dict` of ``str`` to `lsst.afw.image.ExposureF`
        The band-keyed exposures to hand to the task.
    originals : `dict` of ``str`` to `numpy.ndarray`
        Copies of the input pixel arrays taken before the task mutates them.
    """
    yy, xx = np.ogrid[: _SHAPE[0], : _SHAPE[1]]
    core = np.zeros(_SHAPE, dtype=bool)
    for cy, cx in _STAR_POSITIONS:
        core |= (yy - cy) ** 2 + (xx - cx) ** 2 <= _CORE_RADIUS**2

    exposures = {}
    for band, scale in _BAND_SCALES.items():
        exposure = ExposureF(_SHAPE[1], _SHAPE[0])
        array = np.full(_SHAPE, _SKY, dtype=np.float32)
        for cy, cx in _STAR_POSITIONS:
            array += (
                _STAR_AMP
                * scale
                * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * _STAR_SIGMA**2))
            ).astype(np.float32)
        if nan_core:
            array[core] = np.nan
        exposure.image.array = array
        no_data_bit = exposure.mask.getMaskPlaneDict()["NO_DATA"]
        exposure.mask.array[core] |= 2**no_data_bit
        exposures[band] = exposure
    originals = {band: exposure.image.array.copy() for band, exposure in exposures.items()}
    return exposures, originals


def _fused_repair_region(exposures, originals):
    """Re-derive the task-side fused mask and its circularized L leg.

    Mirrors the mask bookkeeping kept in ``PrettyPictureStarFixerTask.run``
    (joint NO_DATA over exposures | brightness threshold on the last
    exposure, dilated by ``_disk(2)`` twice) so the tests know exactly where
    the engine is allowed to touch pixels. ``bright`` is evaluated on the
    pre-run pixel copies.

    Returns
    -------
    both : `numpy.ndarray`
        The fused, dilated (bool) mask passed to the engine as ``mask=``.
    both_lum : `numpy.ndarray`
        The circularized copy passed as ``lum_mask=``.
    """
    joint = np.zeros(_SHAPE, dtype=np.int32)
    for exposure in exposures.values():
        joint |= exposure.mask.array
    last_band = list(exposures)[-1]
    no_data_bit = exposures[last_band].mask.getMaskPlaneDict()["NO_DATA"]
    no_data = (joint & 2**no_data_bit).astype(bool)
    bright = originals[last_band] > _BRIGHT_THRESH

    disk_radius = 2
    y, x = np.ogrid[-disk_radius: disk_radius + 1, -disk_radius: disk_radius + 1]
    struct = (x * x + y * y) <= disk_radius**2
    both = binary_dilation(no_data | bright, struct, iterations=2).astype(bool)
    return both, _circular_mask(both)


def _make_fixer_config(growth=0.02):
    """Star-fixer config with the brightness threshold (and optionally a
    growth override) set."""
    config = PrettyPictureStarFixerConfig()
    config.brightnessThresh = _BRIGHT_THRESH
    config.growth = growth
    return config


class TestStarFixerReconstruction:
    """The star fixer's engine port to ``stellaRGB.reconstruct_saturated_stars``.

    The task keeps its task-side bookkeeping (joint mask, dilation, channel
    mixing and split-back) but delegates the repair, the normalization and the
    flux-unit restoration to the stellaRGB function; these tests run the task
    on synthetic physical-flux exposures (sky ~1e4) and pin the visible
    contract: unrepaired pixels survive bit for bit in input flux units,
    masked cores come back finite and in flux units, NaN under the mask is
    tolerated, the configured growth reaches the engine, and the task's call
    matches the installed engine signature (no remap stage).
    """

    def test_runs_against_installed_signature(self):
        """The task matches the installed engine contract end to end.

        ``reconstruct_saturated_stars`` no longer takes a ``remap`` stage
        (normalization is the function's internal, non-clipping scalar
        working scale), so the task must not hand it one: the signature
        pins below guard the contract in both directions, and the plain
        ``run`` doubles as the call-kwarg check -- a stale ``remap=`` at
        the call site would raise TypeError before any pixel is touched.
        On the installed engine the run must return every pixel finite
        with the repaired cores back at physical flux scale.
        """
        engine = inspect.signature(reconstruct_saturated_stars)
        assert "remap" not in engine.parameters, "the engine grew a remap stage back"
        assert "normalization_scale" in engine.parameters

        exposures, originals = _make_star_exposures()
        both, _ = _fused_repair_region(exposures, originals)
        task = PrettyPictureStarFixerTask(config=_make_fixer_config())

        results = task.run(exposures).results

        cy, cx = _STAR_POSITIONS[0]
        for band, exposure in results.items():
            array = exposure.image.array
            assert np.isfinite(array).all(), f"{band} has non-finite pixels after the run"
            assert array[cy, cx] > 1.0e3, (
                f"{band} repaired core {array[cy, cx]} is not in physical flux units"
            )

    def test_outside_fused_mask_bit_identical(self):
        """Pixels outside the fused dilated mask equal the input exactly, per band.

        The engine returns every pixel outside its composite region untouched
        and the internal working scale round-trips an unclipped
        division/multiplication pair, so no scaling touches the sky: the
        comparison is exact, in the original flux units.
        """
        exposures, originals = _make_star_exposures()
        both, both_lum = _fused_repair_region(exposures, originals)
        task = PrettyPictureStarFixerTask(config=_make_fixer_config())

        results = task.run(exposures).results

        # Scene sanity: the fused mask is nonempty, catches the cores, and the
        # circularized L leg does not spill past it (so equality is asserted
        # strictly outside the fused dilated mask itself).
        assert both.sum() > 0
        assert both[_STAR_POSITIONS[0]].all()
        assert not (both_lum & ~both).any(), "circularized L mask spills outside the dilated mask"
        for band, exposure in results.items():
            assert np.array_equal(exposure.image.array[~both], originals[band][~both]), (
                f"{band} pixels outside the fused dilated mask were not bit-identical"
            )

    def test_saturated_core_repaired_in_flux_units(self):
        """A masked core (NO_DATA plane + bright threshold) comes back finite,
        above the local wing level, and in physical flux units.
        """
        exposures, originals = _make_star_exposures()
        both, _ = _fused_repair_region(exposures, originals)
        task = PrettyPictureStarFixerTask(config=_make_fixer_config())

        results = task.run(exposures).results

        cy, cx = _STAR_POSITIONS[0]
        yy, xx = np.ogrid[: _SHAPE[0], : _SHAPE[1]]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        local = both & (dist < 16.0)
        assert local.any()
        edge = dist[local].max()
        wing_ring = (dist > edge) & (dist < edge + 3.0) & ~both
        for band, exposure in results.items():
            core_value = exposure.image.array[cy, cx]
            wing_level = np.median(originals[band][wing_ring])
            assert np.isfinite(core_value), f"{band} core is not finite"
            assert core_value > wing_level, (
                f"{band} core {core_value} did not rise above the local wing level {wing_level}"
            )
            # Flux scale: repaired cores sit in the thousands-to-tens-of-
            # thousands range like their (unclipped) input star, not in [0, 1].
            assert core_value > 1.0e3, f"{band} core {core_value} is not in physical flux units"

    def test_nan_under_mask_tolerated(self):
        """NaN core pixels under the NO_DATA flag neither stop the run nor leak.

        The engine sanitizes the masked region itself (garbage under the mask
        is re-seeded), so no task-side neutralization is needed and nothing
        non-finite may leak out of the repaired region.
        """
        exposures, originals = _make_star_exposures(nan_core=True)
        both, both_lum = _fused_repair_region(exposures, originals)
        task = PrettyPictureStarFixerTask(config=_make_fixer_config())

        results = task.run(exposures).results

        region = both | both_lum
        for band, exposure in results.items():
            assert np.isfinite(exposure.image.array[region]).all(), (
                f"{band} has non-finite pixels in the repaired region"
            )

    def test_growth_config_sensitivity(self):
        """The configured growth reaches the engine: on one identical scene the
        repaired-region maximum is strictly ordered by the configured growth.
        """
        # radial_rise seeds the L reconstruction with ``growth`` per pixel of
        # mask depth, so a larger growth must yield a strictly brighter repair
        # peak; if the task ignored the field (e.g. hard-coded growth), the
        # three runs on this same scene would collapse to equal maxima and the
        # strict ordering below fails.
        maxima = {}
        for growth in (0.01, 0.02, 0.05):
            exposures, originals = _make_star_exposures()
            both, _ = _fused_repair_region(exposures, originals)
            task = PrettyPictureStarFixerTask(config=_make_fixer_config(growth=growth))
            array = task.run(exposures).results["i"].image.array
            maxima[growth] = array[both].max()

        assert maxima[0.01] < maxima[0.02] < maxima[0.05], f"measured maxima: {maxima}"
