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
from __future__ import annotations

__all__ = ("HighOrderHipsTaskConnections", "HighOrderHipsTaskConfig", "HighOrderHipsTask")

import time
import numpy as np
from enum import Enum
from numpy.typing import NDArray

from lsst.afw.geom import makeHpxWcs
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
    QuantumContext,
    InputQuantizedConnection,
    OutputQuantizedConnection,
)
from lsst.pex.config import ConfigField, Field, ChoiceField
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.skymap import BaseSkyMap
from lsst.afw.geom import SkyWcs
from lsst.geom import Box2I, Point2I, Point2D, Extent2I
from lsst.images import Box, ColorImage, GeneralFrame, SkyProjection
from lsst.afw.math import Warper
from lsst.daf.butler import DeferredDatasetHandle
from lsst.afw.image import ImageF
from lsst.resources import ResourcePath

from collections.abc import Iterable
from lsst.sphgeom import RangeSet

import cv2

from stellaRGB.warp import DirectWarp, WarpTransform, fit_transform, warp_into

from .._warpDirect import hpx_direct_wcs_from_afw, pure_fits_tan, tan_wcs_from_afw
from ._utils import _write_hips_image
from ..prettyPictureMaker import FeatheredMosaicCreator, probe_warp_scale, warp_maps

# The "rust" backend's resampling kernel: mirrors the cv2 backend's effective
# kernel (cv2.INTER_LANCZOS4; the afw reference is lanczos5 per
# HighOrderHipsTaskConfig.setDefaults). Unlike the meta-mosaic integration
# (skymap tracts share the reference CD matrix, |det J| = 1), HiPS tiles have
# their own pixel scale, so the "rust" branch mirrors afwMath's pixel-area
# normalization with probe_warp_scale exactly as the cv2 branch does.
_RUST_KERNEL = "lanczos4"


def _area_downsample(image: NDArray, out_hw: tuple[int, int]) -> NDArray:
    """Exact box-average downsample via the stellaRGB.warp AREA kernel (D10).

    Strict improvement over the previous ``cv2.resize(..., INTER_LANCZOS4)``
    pyramid step, which was a defect not a preference: a Lanczos resize has
    NO prefilter, so any energy above the output Nyquist aliases into the
    level as ringing; the area kernel is the proper prefilter (exact
    weighted box mean, verified against ``cv2.resize(INTER_AREA)`` and
    bit-exact 2x2 box averages in the engine's parity suite).

    Destination pixel ``(y, x)`` is the clip-overlap box mean over the
    source box ``[fy*y - 0.5, fy*y + fy - 0.5] x [fx*x - 0.5, fx*x + fx -
    0.5]`` — exactly what the engine's D10 AREA kernel computes for the
    strided lattice ``map_x = fx*x + (fx-1)/2`` (whose per-axis diff-mean
    scale is ``fx``): integer factors reduce to uniform weight 1.0 on the
    ``f`` source pixels (touching taps carry exact 0.0 weight, the engine's
    support-exempt structure), non-integer (odd-dim) factors get the exact
    partial-overlap weights — cv2 lanczos has no defined box semantics
    there at all.

    ENGINE-FEASIBILITY NOTE (why the box mean is evaluated here in numpy,
    not by calling ``stellaRGB.warp.remap(kernel="area")`` per level): the
    engine's area tap window is the UNIFORM GLOBAL SPAN of the call
    ("area is tile-size only" — the whole-table ``AREA_TOP_CAP`` /
    ``AREA_WEIGHTS_CAP`` prescan). A whole-level 4096->2048 strided remap
    has per-axis spans of the full source (4098 taps) and trips those caps
    with ``ValueError``; at factor 8/16 it would fit the caps yet burn
    ``O(span^2 x dest)`` work (the bench's affine-area hang lesson). The
    pyramid therefore evaluates the IDENTICAL D10 box mean as a separable
    integral-image reduction (float64 accumulation, cast last). D10
    equivalence on grids where the engine call IS feasible (tile-sized)
    is pinned by ``tests/test_hips.py`` (this helper vs the engine
    ``remap(kernel="area")`` on the same strided lattice).

    NaN source pixels poison exactly the boxes touching them (matching the
    engine's D9 contract; ``run()`` zeroes NaN before pyramiding, so
    production is unaffected).

    Parameters
    ----------
    image : `numpy.ndarray`
        ``(H, W)`` or ``(H, W, 3)`` source array (any float dtype).
    out_hw : `tuple` of ``int``
        Output ``(height, width)``; must not exceed the source shape.

    Returns
    -------
    downsampled : `numpy.ndarray`
        Float32 box-averaged array of shape ``out_hw + image.shape[2:]``.
    """
    src_h, src_w = image.shape[:2]
    out_h, out_w = out_hw
    if out_h > src_h or out_w > src_w or out_h < 1 or out_w < 1:
        raise ValueError(
            f"out_hw {out_hw} must be a non-empty downsample of {image.shape[:2]}"
        )
    out = np.empty((out_h, out_w) + image.shape[2:], dtype=np.float32)

    def _reduce_axis(plane: NDArray, axis: int, out_len: int) -> NDArray:
        """Exact D10 clip-overlap box mean along one axis, float64.

        ``plane`` is 2-D; all per-destination-index vectors are broadcast
        along ``axis`` explicitly (a bare ``(out_len,)`` vector would
        broadcast against the LAST dimension and silently misalign the
        axis=0 reduction whenever the grid is square).
        """
        n = plane.shape[axis]
        f = n / out_len
        pos = np.arange(out_len, dtype=np.float64)
        lo = f * pos - 0.5
        hi = lo + f  # == f*pos + f - 0.5, the box edges (see above)
        # Full pixels are [ceil(lo + 0.5), floor(hi - 0.5)] inclusive; the
        # touching partials carry the sub-pixel overhang weights (never
        # reaching outside the array for this lattice: hi <= n - 0.5).
        first = np.ceil(lo + 0.5).astype(np.intp)
        last = np.floor(hi - 0.5).astype(np.intp)
        clean = np.nan_to_num(plane)
        pref = np.cumsum(clean, axis=axis)
        nan_pref = np.cumsum(np.isnan(plane), axis=axis)

        def _take(arr: NDArray, idx: NDArray) -> NDArray:
            return np.take(arr, np.clip(idx, 0, n - 1), axis=axis)

        def _b(vec: NDArray) -> NDArray:
            return vec[:, None] if axis == 0 else vec[None, :]

        wl = _b((first - 0.5) - lo)  # in [0, 1); partial pixel `first - 1`
        wr = _b(hi - (last + 0.5))  # in [0, 1); partial pixel `last + 1`
        has_l = _b(first > 0)
        has_r = _b(last + 1 < n)
        # sum of full pixels [first .. last] (empty window => 0):
        s = _take(pref, last) - np.where(has_l, _take(pref, first - 1), 0.0)
        s = s + np.where(has_l, wl * _take(clean, first - 1), 0.0)
        s = s + np.where(has_r, wr * _take(clean, last + 1), 0.0)
        # NaN poisons exactly the boxes whose window touched a NaN pixel.
        nan_box = _take(nan_pref, last) - np.where(has_l, _take(nan_pref, first - 1), 0.0)
        nan_box = nan_box + np.where(
            has_l & (wl > 0.0),
            _take(nan_pref, first - 1) - np.where(_b(first > 1), _take(nan_pref, first - 2), 0.0),
            0.0,
        ) + np.where(
            has_r & (wr > 0.0),
            _take(nan_pref, last + 1) - _take(nan_pref, last),
            0.0,
        )
        result = s / f
        return np.where(nan_box > 0.0, np.nan, result)

    planes = image if image.ndim == 3 else image[..., None]
    for ch in range(planes.shape[2]):
        plane = np.ascontiguousarray(planes[..., ch], dtype=np.float64)
        reduced = _reduce_axis(plane, axis=1, out_len=out_w)
        reduced = _reduce_axis(np.ascontiguousarray(reduced), axis=0, out_len=out_h)
        out[..., ch] = reduced
    return out if image.ndim == 3 else out[..., 0]


class ColorChannel(Enum):
    """Enum representing color channels for image processing."""

    RED = 0
    GREEN = 1
    BLUE = 2


class HighOrderHipsTaskConnections(PipelineTaskConnections, dimensions=("healpix8",)):
    input_images = Input(
        doc="Color images which are to be turned into hips tiles",
        name="rgb_picture",
        storageClass="ColorImage",
        dimensions=("tract", "patch", "skymap"),
        multiple=True,
        deferLoad=True,
    )
    skymap = Input(
        doc="The skymap which the data has been mapped onto",
        storageClass="SkyMap",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        dimensions=("skymap",),
    )
    output_hpx = Output(
        doc="Healpix tiles at order 8, but binned to 256x256",
        name="rgb_picture_hips8",
        storageClass="NumpyArray",
        dimensions=("healpix8",),
    )


class HighOrderHipsTaskConfig(PipelineTaskConfig, pipelineConnections=HighOrderHipsTaskConnections):
    """Configuration class for the HighOrderHipsTask pipeline task."""

    hips_order = 8
    """HealPix order to generate tiles for."""
    warp = ConfigField[Warper.ConfigClass](
        doc="Warper configuration",
    )
    hips_base_uri = Field[str](
        doc="URI to HiPS base for output.",
        optional=False,
    )
    color_ordering = Field[str](
        doc=(
            "A string of the astrophysical bands that correspond to the RGB channels in the color image "
            "inputs to high_order_hips task. This is in making the hips metadata"
        ),
        optional=False,
    )
    file_extension = ChoiceField[str](
        doc="Extension for the presisted image.",
        allowed={"png": "Use the png image extension", "webp": "Use the webp image extension"},
        default="png",
    )
    array_type = ChoiceField[str](
        doc="The dataset type for the output image array",
        default="uint8",
        allowed={
            "uint8": "Use 8 bit arrays, 255 max",
            "uint16": "Use 16 bit arrays, 65535 max",
            "half": "Use 16 bit float arrays, 1 max",
            "float": "Use 32 bit float arrays, 1 max",
        },
    )
    warp_backend = ChoiceField[str](
        doc=(
            "Backend used to resample input images onto the HiPS tile WCS. "
            "DESIGN NOTE (rust): preferred mode is the DIRECT per-pixel "
            "engine (T8.6/D18): when the tile WCS is a plain HiPS HPX "
            "header and the source WCS passes the shared pure-FITS-TAN "
            "gate (lsst.pipe.tasks._warpDirect), the (tile, source) pair is "
            "rendered by a closed-form HPX-on-TAN DirectWarp -- zero fits, "
            "zero coordinate maps. The (image WCS, tile WCS) control-grid "
            "fit_transform ladder remains as the FALLBACK for anything the "
            "gate declines (SIP or other distortion, non-TAN sources, "
            "exotic tile headers): the fit correctly returns maps mode "
            "(densified control nodes) for the curvilinear pair, and "
            "fit-verified control-grid maps -- never a dense per-pixel WCS "
            "map build -- serve those images. Pyramid levels are "
            "backend-independent (D14)."
        ),
        allowed={
            "afw": "afwMath Warper (default, reference)",
            "cv2": "precomputed-map cv2.remap (Lanczos4)",
            "rust": "stellaRGB.warp engine: direct per-pixel HPX-on-TAN "
            "warp (DirectWarp) for plain HiPS + pure FITS TAN pairs, "
            "fit-verified control-grid maps otherwise (Lanczos4)",
        },
        default="afw",
    )

    def setDefaults(self):
        self.warp.warpingKernelName = "lanczos5"


class HighOrderHipsTask(PipelineTask):
    """Pipeline task that generates high-order HealPix tiles from RGB images.

    Of Note; This task has special dispensation to write "out-of-tree" to a
    location not within the butler. DO NOT model other tasks on this one.

    This task takes in RGB images generated on a tract patch grid. It assembles
    them into a 4096 x 4096 image aligned with the wcs coordinates of hips
    order 8 pixels. This is then divided up into an 8x8 grid to produce 512x512
    images at hips order 11. The images is then resampled using an exact
    box-average (area kernel) prefilter such that the image is half the size.
    The original image is then divided
    into a 4x4 grid to produce hips images at order 10. The process is repeated
    to produce hips images at order 9, and finally the image is resampled down
    to 512x512 and saved out at hips order 8.

    The order 8 image is resampled one more time to 256x256 and presisted by
    the butler for later consumption in the `LowOrderHipsTask`.

    The difference at producding wcs at order 8 and working up to 11, is tested
    to be less than 6 decimal places when converting ra dec to pixel coordinates,
    and even that is likely to be due to differences in warping kernels,
    and not an intrinsic error. Doing processing like this allows hips generation
    to be more effectively split across compute nodes.
    """

    _DefaultName = "highOrderHipsTask"
    ConfigClass = HighOrderHipsTaskConfig

    config: ConfigClass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.warper = Warper.fromConfig(self.config.warp)

        # Per-(image, tile) direct-warp memo for the "rust" backend
        # (T8.6): DirectWarp when both gates pass, None when either
        # declines (the None verdict is cached too, so a declined source
        # is never re-gated on a task retry). DirectWarp holds two
        # frozen header triples (no maps, cheap); the win is skipping
        # re-gating (the 9-point probe) on task retry.
        self._rust_direct_warps: dict[tuple, DirectWarp | None] = {}

        # Per-(image, tile) WarpTransform memo for the "rust" backend
        # FALLBACK ladder (content-keyed; values may be None = failed fit,
        # which also suppresses per-retry warn spam). Maps-mode transforms
        # hold only control nodes (cheap); the win is skipping re-fit on
        # task retry.
        self._rust_transforms: dict[tuple, WarpTransform | None] = {}

        # Set the base resource path that will be used for all outputs
        self.hips_base_path = ResourcePath(self.config.hips_base_uri, forceDirectory=True)
        self.hips_base_path = self.hips_base_path.join(
            f"color_{self.config.color_ordering}", forceDirectory=True
        )

    def run(self, input_images: Iterable[tuple[NDArray, SkyWcs, Box2I]], healpix_id) -> Struct:
        """Main execution method for generating HealPix tiles.

        Parameters
        ----------
        input_images : Iterable[tuple[NDArray, SkyWcs, Box2I]]
            Iterable of tuples containing image data, WCS, and bounding box information.
        healpix_id : int
            The HealPix order 8 ID to process.

        Returns
        -------
        Struct
            Output structure containing the processed HealPix order 8 tile.
            This has been downsampled to 256x256 corresponding to a quarter of a healpix
            order 7 image.
        """
        # Make the WCS for the transform, intentionally over-sampled to shift order 12.
        # This creates as 4096 x 4096 image that can be broken apart to form the higher
        # orders, binning each as needed
        target_wcs = makeHpxWcs(8, healpix_id, 12, False)

        # construct a bounding box that holds the warping results for each channel
        exp_bbox = Box2I(corner=Point2I(0, 0), dimensions=Extent2I(2**12, 2**12))

        output_array_hpx = np.zeros((4096, 4096, 3), dtype=np.float32)
        output_array_hpx[:, :, :] = np.nan

        use_cv2 = self.config.warp_backend == "cv2"
        use_rust = self.config.warp_backend == "rust"
        # Warp invocation timings (per engine/remap/warpImage call; the
        # rust direct path performs exactly one warp_into per image, the
        # per-channel afw/cv2 backends three) and, for the "rust" backend
        # fallback ladder, the fit clock: "map-build" times control-grid
        # fits and counts only REAL fits — direct-mode images (T8.6)
        # perform no fit and contribute nothing, so a full-direct run
        # reports map-build n=0 (mirrors the MetaMosaicTask Timings line).
        t_warp = t_map = 0.0
        n_warp = n_map = 0

        self.log.info("Warping input exposures and populating hpx8 super tile.")
        # Need to loop over input arrays then channel
        # Warp and combine input images into the HealPix tile
        for input_image, in_wcs, in_box in input_images:
            tmp_image = ImageF(in_box)
            in_image: NDArray = input_image

            # Normalize image data based on dtype
            match in_image.dtype:
                case np.uint8:
                    in_image = in_image.astype(np.float32) / 255.0
                case np.uint16:
                    in_image = in_image.astype(np.float32) / 65535
                case np.float16:
                    in_image = in_image.astype(np.float32)

            rust_warped = None
            if use_cv2 or use_rust:
                # Replicate the destination domain afwMath warps onto: the
                # source footprint in target_wcs coordinates clipped to the
                # 4096 x 4096 tile, computed conservatively (see
                # _footprint_bbox). Empty footprints contribute nothing.
                warped_box = self._footprint_bbox(in_wcs, in_box, target_wcs, exp_bbox)
                if warped_box.isEmpty():
                    continue
                # Both non-afw backends mirror the (kernel-independent)
                # pixel-area scaling afwMath's Warper applies (the HiPS tile
                # scale differs materially from the tract scale, unlike the
                # meta-mosaic |det J| ~ 1 case), and share the footprint box
                # so the blend-block slicing is backend-independent.
                warp_scale = probe_warp_scale(self.warper, in_wcs, target_wcs, in_box)
                warpped_box_slices = warped_box.slices
                if use_cv2:
                    # Coordinate maps are built once per input image (over the
                    # dense footprint box) and reused for all three channels.
                    map_x, map_y = warp_maps(in_wcs, in_box, target_wcs, warped_box)
                else:
                    # T8.6 (D18/D19) mode selection, once per (source, tile)
                    # class and cached on the task: a plain HiPS HPX tile
                    # fed by a pure FITS TAN source (the production case:
                    # post-acalexn warps) renders through the direct
                    # per-pixel engine — no fit, no maps. Anything the
                    # shared gate declines (SIP or other distortion,
                    # non-TAN sources, exotic tile headers) takes the
                    # T6.2 fit/maps fallback below, unchanged.
                    direct = self._rust_direct_warp(in_wcs, in_box, target_wcs, healpix_id)
                    if direct is not None:
                        # Direct logs per image always; maps self-suppresses
                        # on cache hits. Intentional: volume bounded by quanta.
                        self.log.info(
                            "rust backend tile %s: hpx-direct (dest box %d x %d px)",
                            healpix_id,
                            warped_box.getWidth(),
                            warped_box.getHeight(),
                        )
                        # The same bbox/origin plumbing as the maps branch:
                        # the ColorImage frame contract places the source
                        # array at its own bbox in in_wcs parent coordinates
                        # and warp_into renders exactly the footprint box
                        # the maps path already computed (direct mode has no
                        # D7 containment rule — the requested box is the
                        # whole render contract, D16).
                        t0 = time.perf_counter()
                        rust_warped = warp_scale * warp_into(
                            ColorImage(in_image, bbox=Box.from_legacy(in_box)),
                            direct,
                            kernel=_RUST_KERNEL,
                            dest_bbox=Box.from_legacy(warped_box),
                        )
                        t_warp += time.perf_counter() - t0
                        n_warp += 1
                    else:
                        n_before = len(self._rust_transforms)
                        tm0 = time.perf_counter()
                        transform = self._rust_tile_transform(
                            in_wcs, in_box, target_wcs, warped_box, healpix_id
                        )
                        t_map += time.perf_counter() - tm0
                        if transform is not None:
                            # "map-build" ticks only on a REAL fit (the
                            # per-(image, tile) memo makes a cache hit
                            # free); the mode line is logged once per class,
                            # mirroring meta-mosaic's "rust-maps" style.
                            if len(self._rust_transforms) > n_before:
                                n_map += 1
                                self.log.info(
                                    "rust backend tile %s: rust-maps spacing=%.1f residual=%.2e"
                                    " (dest box %d x %d px)",
                                    healpix_id,
                                    transform.control_spacing,
                                    transform.max_residual,
                                    warped_box.getWidth(),
                                    warped_box.getHeight(),
                                )
                            # One engine render for all three channels (the
                            # ColorImage frame contract places the source
                            # array at its own bbox in in_wcs parent
                            # coordinates; the render is the contained
                            # sub-box == the transform's dest_bbox).
                            t0 = time.perf_counter()
                            rust_warped = warp_scale * warp_into(
                                ColorImage(in_image, bbox=Box.from_legacy(in_box)),
                                transform,
                                kernel=_RUST_KERNEL,
                                dest_bbox=Box.from_legacy(warped_box),
                            )
                            t_warp += time.perf_counter() - t0
                            n_warp += 1
                        # else: the fit failed (already warned once per
                        # image) -- this image falls through to the afw
                        # Warper below.

            # Process each color channel separately
            for channel in ColorChannel:
                # existing data
                existing = output_array_hpx[..., channel.value]

                # construct an Exposure object from one channel in the array
                channel_array = in_image[..., channel.value]
                if rust_warped is not None:
                    warpped_array = rust_warped[..., channel.value]
                elif use_cv2:
                    t0 = time.perf_counter()
                    warpped_array = warp_scale * cv2.remap(
                        np.ascontiguousarray(channel_array),
                        map_x,
                        map_y,
                        cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=np.nan,
                    )
                    t_warp += time.perf_counter() - t0
                    n_warp += 1
                else:
                    tmp_image.array[:, :] = channel_array
                    # Warp the image to the target WCS
                    t0 = time.perf_counter()
                    warpped = self.warper.warpImage(target_wcs, tmp_image, in_wcs, maxBBox=exp_bbox)
                    t_warp += time.perf_counter() - t0
                    n_warp += 1
                    warpped_array = warpped.array
                    warpped_box_slices = warpped.getBBox().slices

                # Update the output array with valid (non-NaN) values
                are_warpped = np.isfinite(warpped_array)
                # where this existing array is not nan, there is already data
                pre_populated = np.isfinite(existing[warpped_box_slices])
                new_info = np.logical_not(pre_populated)
                new_and_warp = np.logical_and(new_info, are_warpped)
                pre_and_warp = np.logical_and(pre_populated, are_warpped)
                existing[warpped_box_slices][new_and_warp] = warpped_array[new_and_warp]
                existing[warpped_box_slices][pre_and_warp] = (
                    warpped_array[pre_and_warp] + existing[warpped_box_slices][pre_and_warp]
                ) * 0.5

        self.log.info(
            "Timings: warp=%.1fs (n=%d), map-build=%.1fs (n=%d)",
            t_warp,
            n_warp,
            t_map,
            n_map,
        )

        # Replace any remaining NaN values with zeros
        output_array_hpx[np.isnan(output_array_hpx)] = 0

        # Flip the y-axis to match HealPix indexing
        output_array_hpx = output_array_hpx[::-1, :, :]

        # Generate tiles for different HealPix orders using an exact
        # box-average (area-kernel) prefilter instead of binning or Lanczos.
        # This handles how intensities should change as the hips level changes
        # AND anti-aliases: the previous cv2.resize(INTER_LANCZOS4) had no
        # prefilter, so energy above the level Nyquist aliased as ringing
        # (see _area_downsample).
        #
        # what this does is take a single 4096 x 4096 image and resamples it in a courser grain such
        # that the output pixels correspond to a 4x4 grid of hips pixels at an increasingly lower scale.
        # This works because hips is a hierarchy of tiles all contained in the same area of the sky.
        # This allows us to generate all the output images by resampling the inputs and saves the time
        # required to generate whole new images at each scale.
        #
        # The loop variables are the resampling factor, the hips order, and the number of sub-divisions
        # a pixel has gone through (used to determine quadrant).
        for zoom, hips_level, factor in zip((0, 2, 4, 8), (11, 10, 9, 8), (3, 2, 1, 0)):
            self.log.info("Generating tiles for hxp level %d", hips_level)
            if zoom:
                size = 4096 // zoom
                binned_array = _area_downsample(output_array_hpx, (size, size))
            else:
                binned_array = output_array_hpx
            # always create blocks of 512x512 as that is native shift order 9 size
            #
            # Figure out the hips pixel ids at this hips order. This is complicated because each hipx pixel
            # turns into 4 at a higher level, but must be in a specific order to correspond to how the data
            # is layed out in an y,x grid. So if a hips order 8 pixel A turns into four pixels b,c,d,e, they
            # are layed out like [[b,d], [c,e]]. This is true for every pixel as you go up in order. So
            # if you start at order 8 with one pixel, you need to do order 9 and calculate the layout. Then
            # for each order 9 pixel, do the same to get the layout in order 10, etc. This leaves a grid
            # of pixels that are the ids of the corresponding 512,512 sub grid pixel in the input image.
            tmp_pixels = np.array([[healpix_id]])
            for _ in range(factor):
                tmp_array = np.zeros(np.array(tmp_pixels.shape) * 2)
                for ii in range(tmp_pixels.shape[0]):
                    for jj in range(tmp_pixels.shape[1]):
                        tmp_array_view = tmp_array[ii * 2 : ii * 2 + 2, jj * 2 : jj * 2 + 2]
                        tmp_range_set = RangeSet(int(tmp_pixels[ii, jj]))
                        tmp_array_view[:, :] = (
                            np.array([x for x in range(*tmp_range_set.scaled(4)[0])], dtype=int)[[0, 2, 1, 3]]
                        ).reshape(2, 2)
                tmp_pixels = tmp_array

            # now for each 512x512 sub pixel region write the hips image with the corresponding healpix id
            hpx_id_array = tmp_pixels
            for i in range(binned_array.shape[0] // 512):
                for j in range(binned_array.shape[1] // 512):
                    pixel_id = int(hpx_id_array[i, j])
                    sub_pixel = binned_array[i * 512 : i * 512 + 512, j * 512 : j * 512 + 512, :]
                    self.log.info(f"writing sub_pixel {pixel_id}")
                    _write_hips_image(
                        sub_pixel,
                        pixel_id,
                        hips_level,
                        self.hips_base_path,
                        self.config.file_extension,
                        self.config.array_type,
                    )

        # Finally, bin the level 8 hpx to 256x256 (1/4 order 7) to save to the butler.
        # This makes smaller arrays to load, and saves the binning operation in the joint phase.
        # Same exact box-average prefilter as the tile pyramid above.
        zoomed = _area_downsample(output_array_hpx, (256, 256))

        return Struct(output_hpx=zoomed)

    def _footprint_bbox(
        self, src_wcs: SkyWcs, src_box: Box2I, dest_wcs: SkyWcs, max_box: Box2I, samples: int = 64
    ) -> Box2I:
        """Conservative enclosing box, in ``dest_wcs`` parent coordinates, of a
        source footprint, clipped to ``max_box``.

        afwMath computes the warp destination domain from the source bbox;
        because the sky projection is nonlinear and the grids rotate relative
        to each other, its extent can reach slightly past a plain transform
        of the four bbox corners. Sampling the entire perimeter densely and
        adding a one-pixel margin therefore always covers (and may slightly
        overestimate) the afw destination domain. Overestimation is harmless:
        map coordinates outside the source array yield NaN (the missing-data
        marker), which the blending block skips, exactly as it skips the NaN
        pixels afwMath leaves outside the true footprint.

        Parameters
        ----------
        src_wcs : `lsst.afw.geom.SkyWcs`
            WCS of the source image.
        src_box : `lsst.geom.Box2I`
            Source bbox in ``src_wcs`` parent coordinates.
        dest_wcs : `lsst.afw.geom.SkyWcs`
            WCS of the destination grid.
        max_box : `lsst.geom.Box2I`
            Maximum allowed destination box (e.g. the tile); the result is
            intersected with it.
        samples : `int`, optional
            Number of perimeter samples per edge. Default is 64.

        Returns
        -------
        box : `lsst.geom.Box2I`
            The (possibly empty) covering box in ``dest_wcs`` parent
            coordinates, clipped to ``max_box``.
        """
        x0 = float(src_box.getMinX())
        y0 = float(src_box.getMinY())
        x1 = float(src_box.getMaxX())
        y1 = float(src_box.getMaxY())
        n = max(int(samples), 2)
        t = np.linspace(0.0, 1.0, n)
        xs = np.concatenate([x0 + (x1 - x0) * t, np.full(n, x1), x1 - (x1 - x0) * t, np.full(n, x0)])
        ys = np.concatenate([np.full(n, y0), y0 + (y1 - y0) * t, np.full(n, y1), y1 - (y1 - y0) * t])
        px = np.empty(xs.size)
        py = np.empty(ys.size)
        for index, (sx, sy) in enumerate(zip(xs, ys, strict=True)):
            p = dest_wcs.skyToPixel(src_wcs.pixelToSky(Point2D(sx, sy)))
            px[index] = p.getX()
            py[index] = p.getY()
        if not (np.isfinite(px).any() and np.isfinite(py).any()):
            return Box2I()  # fully unreachable footprint
        box = Box2I(
            Point2I(int(np.floor(np.nanmin(px))) - 1, int(np.floor(np.nanmin(py))) - 1),
            Point2I(int(np.ceil(np.nanmax(px))) + 1, int(np.ceil(np.nanmax(py))) + 1),
        )
        # Manual intersection (lsst.geom.Box2I has no intersectedTo).
        x0 = max(box.getMinX(), max_box.getMinX())
        y0 = max(box.getMinY(), max_box.getMinY())
        x1 = min(box.getMaxX(), max_box.getMaxX())
        y1 = min(box.getMaxY(), max_box.getMaxY())
        if x1 < x0 or y1 < y0:
            return Box2I()
        return Box2I(Point2I(x0, y0), Point2I(x1, y1))

    def _rust_direct_warp(
        self,
        in_wcs: SkyWcs,
        in_box: Box2I,
        target_wcs: SkyWcs,
        healpix_id: int,
    ) -> DirectWarp | None:
        """Build (and memoize) the "rust" backend's DIRECT per-pixel warp
        for one (source image, tile) pair, or None when either side needs
        the fitted-maps fallback ladder (T8.6, D18/D19).

        The destination side is built by the shared HiPS header gate
        (:func:`~lsst.pipe.tasks._warpDirect.hpx_direct_wcs_from_afw`:
        exact ``RA---HPX``/``DEC--HPX`` cards, inert PV2_1=4/PV2_2=3
        treated as absent, crval dec 0 within construction noise); the
        source side by the shared pure-FITS-TAN gate probed over
        ``in_box`` — the post-acalexn production warps qualify. Both
        present yields a :class:`stellaRGB.warp.DirectWarp`, whose engine
        render evaluates the HPX-on-TAN coordinate pair per destination
        pixel in f64: no control-grid fit, no coordinate maps, and (no D7
        containment rule) no dependence on the footprint box beyond the
        box actually requested. Any gate decline or constructor
        exception fails CLOSED to None (the caller's maps ladder is
        untouched).

        The result (including the None verdict, so a declined source is
        not re-gated on a task retry) is cached on the task keyed by
        content (healpix id, source WCS parameters, source box) — one
        gate pass per (image, tile). Exception: a source whose origin
        triple answers NaN keys the cache on NaN, which never compares
        equal to itself, so that (fail-closed) verdict re-gates per
        image — correct, but correct-but-not-cached.

        Parameters
        ----------
        in_wcs : `lsst.afw.geom.SkyWcs`
            WCS of the source image (parent pixel frame).
        in_box : `lsst.geom.Box2I`
            Source bbox in ``in_wcs`` parent coordinates (the box the
            pure-TAN gate probes).
        target_wcs : `lsst.afw.geom.SkyWcs`
            The HiPS tile WCS (this quantum's order-8 tile, oversampled).
        healpix_id : `int`
            The order-8 healpix id of the tile (cache key component).

        Returns
        -------
        warp : `stellaRGB.warp.DirectWarp` | `None`
            The direct warp, or None if either side was declined (the
            caller must serve that image from the maps fallback ladder).
        """
        cd = np.asarray(in_wcs.getCdMatrix()).ravel()
        origin = in_wcs.getPixelOrigin()
        sky = in_wcs.getSkyOrigin()
        key = (
            healpix_id,
            float(cd[0]), float(cd[1]), float(cd[2]), float(cd[3]),
            float(origin.getX()), float(origin.getY()),
            sky.getRa().asDegrees(), sky.getDec().asDegrees(),
            (in_box.getMinX(), in_box.getMinY(), in_box.getMaxX(), in_box.getMaxY()),
        )
        if key not in self._rust_direct_warps:
            warp = None
            dest = hpx_direct_wcs_from_afw(target_wcs)
            if dest is not None and pure_fits_tan(in_wcs, in_box):
                try:
                    warp = DirectWarp(dest=dest, src=tan_wcs_from_afw(in_wcs))
                except Exception:
                    # Field validation at the engine boundary or a future
                    # constructor contract rejecting the pair: fail closed
                    # to the maps ladder like the gates do.
                    warp = None
            self._rust_direct_warps[key] = warp
        return self._rust_direct_warps[key]

    def _rust_tile_transform(
        self,
        in_wcs: SkyWcs,
        in_box: Box2I,
        target_wcs: SkyWcs,
        warped_box: Box2I,
        healpix_id: int,
    ) -> WarpTransform | None:
        """Fit (and memoize) the "rust" backend FALLBACK transform for one
        (image, tile) pair.

        T8.6: the direct per-pixel engine (:meth:`_rust_direct_warp`) is
        preferred for plain HiPS tiles fed by pure FITS TAN sources; this
        ladder serves only what those gates decline. HiPS tiles are
        curvilinear, so the fit will (correctly) return maps mode with
        densified control nodes; affine mode is taken only for a
        residual-flat pair. The fit is control-grid only — the dense
        per-pixel WCS map build of the cv2 path never happens here.

        The fit source box is grown by 2 px so an affine-mode footprint
        (the corner-image rectangle of the source box, whose fit guard
        requires it to contain the requested dest box) cannot be tripped by
        the perimeter sampling's 1-px conservative margin in
        ``_footprint_bbox`` at the tile's pixel-scale ratio.

        The result is cached on the task keyed by content (healpix id, WCS
        parameters, both boxes) — one fit per (image, tile), surviving task
        retries; a failed fit caches ``None`` (warned once, never retried)
        and the caller serves that image from the afw Warper.

        Parameters
        ----------
        in_wcs : `lsst.afw.geom.SkyWcs`
            WCS of the source image (parent pixel frame).
        in_box : `lsst.geom.Box2I`
            Source bbox in ``in_wcs`` parent coordinates.
        target_wcs : `lsst.afw.geom.SkyWcs`
            The HiPS tile WCS (this quantum's order-8 tile, oversampled).
        warped_box : `lsst.geom.Box2I`
            Conservative source footprint in ``target_wcs`` parent
            coordinates (the transform's dest box).
        healpix_id : `int`
            The order-8 healpix id of the tile (cache key component).

        Returns
        -------
        transform : `stellaRGB.warp.WarpTransform` | `None`
            The fit-verified transform, or None if the fit failed (a warning
            naming the image has been logged; the caller must fall back to
            the afw Warper for this image).
        """
        cd = np.asarray(in_wcs.getCdMatrix()).ravel()
        origin = in_wcs.getPixelOrigin()
        sky = in_wcs.getSkyOrigin()
        key = (
            healpix_id,
            float(cd[0]), float(cd[1]), float(cd[2]), float(cd[3]),
            float(origin.getX()), float(origin.getY()),
            sky.getRa().asDegrees(), sky.getDec().asDegrees(),
            (in_box.getMinX(), in_box.getMinY(), in_box.getMaxX(), in_box.getMaxY()),
            (warped_box.getMinX(), warped_box.getMinY(),
             warped_box.getMaxX(), warped_box.getMaxY()),
        )
        if key not in self._rust_transforms:
            src_box = Box2I(in_box)
            src_box.grow(2)
            try:
                transform = fit_transform(
                    SkyProjection.from_legacy(in_wcs, GeneralFrame(unit="pixel")),
                    Box.from_legacy(src_box),
                    SkyProjection.from_legacy(target_wcs, GeneralFrame(unit="pixel")),
                    Box.from_legacy(warped_box),
                )
            except Exception as exc:
                self.log.warning(
                    "WarpTransform fit failed for tile %s image %s (%r); "
                    "falling back to the afw Warper for this image.",
                    healpix_id,
                    in_box,
                    exc,
                )
                transform = None
            self._rust_transforms[key] = transform
        return self._rust_transforms[key]

    def _assemble_sub_region(
        self, tract_patch: dict[int, Iterable[tuple[DeferredDatasetHandle, SkyWcs, Box2I]]], patch_grow: int
    ) -> list[tuple[NDArray, SkyWcs, Box2I]]:
        """Assemble all the patches in each tract into images.

        This function takes in an input keyed by tract, with values
        corresponding the patches in that tract that overlap the quatum's
        healpix value. It assembles each of these into a single image such
        that the return values is a list of images (and metadata) one element
        for each input tract.

        Parameters
        ----------
        tract_patch : `dict` of `int` to `iterable` of `tuple` of
                      `DeferredDatasetHandle`, `SkyWcs` and `Box2I`
            Input images and metadata organized into corresponding tracts.
        patch_grow : `int`
            Amount to grow patches by

        Returns
        -------
        output_list : `list` of `tuple` of `NDArray` `SkyWcs` and `Box2I`
            List of assembled images and metadata, one element for each tract

        """

        boxes = []
        for _, iterable in tract_patch.items():
            mosaic_maker = FeatheredMosaicCreator(patch_grow)
            new_box = Box2I()
            for _, _, bbox in iterable:
                new_box.include(bbox)
            # allocate tmp array
            new_array = np.zeros((new_box.getHeight(), new_box.getWidth(), 3), dtype=np.float32)
            for handle, skyWcs, box in iterable:
                # Make a new box of the same size, but with the origin centered
                # on the lowest corner were there is data.
                localOrigin = box.getBegin() - new_box.getBegin()
                localOrigin = Point2I(
                    x=int(np.floor(localOrigin.x)),
                    y=int(np.floor(localOrigin.y)),
                )

                localExtent = Extent2I(
                    x=int(np.floor(box.getWidth())),
                    y=int(np.floor(box.getHeight())),
                )
                tmpBox = Box2I(localOrigin, localExtent)
                tmp_new_box = Box2I(Point2I(x=0, y=0), Extent2I(x=new_box.getWidth(), y=new_box.getHeight()))

                image = handle.get()
                if getattr(image, "metadata", None) and image.metadata.get("is_hdr"):
                    raise ValueError(
                        "rgb2hips received an is_hdr ColorImage, but HiPS delivery assumes SDR [0, 1] "
                        "data and would destroy the HDR signal. Deliver an SDR rendering to this task."
                    )
                mosaic_maker.add_to_image(
                    new_array,
                    image.array,
                    Box.from_legacy(tmp_new_box),
                    Box.from_legacy(tmpBox),
                    reverse=False,
                )
            # skywcs is the same for all patches in a tract, share the appending here
            boxes.append((new_array, skyWcs, new_box))
        return boxes

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        # First get what healpix pixel this task is working on
        healpix_id = butlerQC.quantum.dataId["healpix8"]

        # grab the skymap
        skymap: BaseSkyMap = butlerQC.get(inputRefs.skymap)

        # Iterate over the input image refs, to get the corresponding bbox
        # and assemble into container for run
        inputs_by_tract = {}
        for input_image_ref in inputRefs.input_images:
            tract = input_image_ref.dataId["tract"]
            patch = input_image_ref.dataId["patch"]
            # All boxes in a given skymap will have the same inner dimensions
            # for x and y and will be the same for all patches
            imageWcs = skymap[tract][patch].getWcs()
            box = skymap[tract][patch].getOuterBBox()
            patch_grow = skymap[tract][patch].getCellInnerDimensions().getX()
            imageHandle = butlerQC.get(input_image_ref)
            container = inputs_by_tract.setdefault(tract, list())
            container.append((imageHandle, imageWcs, box))

        input_images = self._assemble_sub_region(inputs_by_tract, patch_grow)

        outputs = self.run(input_images, healpix_id)
        butlerQC.put(outputs, outputRefs)
