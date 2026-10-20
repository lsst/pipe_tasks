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

__all__ = (
    "MetaMosaicTask",
    "MetaMosaicConnections",
    "MetaMosaicConfig",
)

import copy
import tempfile
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

import colour
import cv2
import numpy as np

from lsst.afw.geom import makeSkyWcs
from lsst.afw.image import ImageF
from lsst.afw.math import Warper
from lsst.daf.butler import DeferredDatasetHandle
from lsst.geom import Box2I, Extent2I, Point2D, Point2I
from lsst.images import Box, SkyProjection, TractFrame
from lsst.pex.config import ChoiceField, ConfigField, Field
from lsst.pipe.base import (
    InputQuantizedConnection,
    NoWorkFound,
    OutputQuantizedConnection,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    QuantumContext,
    Struct,
)
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.skymap import BaseSkyMap
from stellaRGB.types import ColorImage
from stellaRGB.warp import WarpTransform, fit_transform, warp_into

from ._utils import FeatheredMosaicCreator, probe_warp_scale, warp_maps

# The "rust" backend's resampling kernel. The afw reference uses lanczos5
# (MetaMosaicConfig.setDefaults) and the rust engine has no lanczos5; "lanczos4"
# mirrors the cv2 backend's effective kernel (cv2.INTER_LANCZOS4). The rust
# engine applies no pixel-area scaling (unlike the afw Warper the cv2 path
# mirrors with probe_warp_scale); skymap tracts share the reference tract's CD
# matrix, so the transform's pixel-space Jacobian has |det| = 1 and none is
# needed.
_RUST_KERNEL = "lanczos4"

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from lsst.afw.geom import SkyWcs
    from lsst.skymap import TractInfo, PatchInfo


class MetaMosaicConnections(PipelineTaskConnections, dimensions=("skymap",)):
    """Connections for the cross-tract meta mosaic task.

    ``dimensions=("skymap",)`` (NOT ``tract``) because a single quantum must see
    every tract's patches in order to size the mosaic and pick a reference tract.
    No mask is produced; this is pure assembly.
    """

    inputRGB = Input(
        doc="RGB images that are to go into the mosaic",
        name="rgb_picture",
        storageClass="ColorImage",
        dimensions=("tract", "patch", "skymap"),
        multiple=True,
        deferLoad=True,
    )

    skyMap = Input(
        doc="The skymap which the data has been mapped onto",
        storageClass="SkyMap",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        dimensions=("skymap",),
    )

    outputRGBMosaic = Output(
        doc="A cross-tract RGB mosaic created from the input data",
        name="rgb_meta_mosaic",
        storageClass="ColorImage",
        dimensions=("skymap",),
    )


class MetaMosaicConfig(PipelineTaskConfig, pipelineConnections=MetaMosaicConnections):
    """Configuration for the cross-tract meta mosaic task."""

    bin_factor = Field[int](doc="The factor to bin by when producing the mosaic", default=1)
    do_dci_d65_convert = Field[bool](
        doc="Force the output to be converted from display P3 to DCI-D65 colorspace.", default=False
    )
    use_local_temp = Field[bool](
        doc="Use the current directory when creating local temp files.", default=False
    )
    reference_tract = Field[int](
        doc="Tract whose WCS/pixel grid defines the output mosaic; None = auto-select.",
        optional=True,
        default=None,
    )
    crop_mode = ChoiceField(
        doc="How to crop the output raster.",
        dtype=str,
        allowed={
            "none": "Keep the full union rectangle, empty margins included.",
            "full_coverage": "Crop to the largest rectangle of fully covered virtual-patch "
            "cells; cells containing holes are excluded; outlier clusters outside it are "
            "dropped. The default.",
            "data_extent": "Crop to the tightest rectangle containing every input pixel: "
            "only outer rows/columns that are entirely empty are trimmed, so partial rows "
            "and interior holes survive as black. Intended for inspecting mosaic coverage "
            "/ what data is missing.",
        },
        default="full_coverage",
    )
    warp = ConfigField[Warper.ConfigClass](doc="Warper configuration")
    warp_backend = ChoiceField(
        doc="Backend used to resample foreign-tract patches onto the reference grid.",
        dtype=str,
        allowed={
            "afw": "afwMath Warper (default, reference)",
            "cv2": "precomputed-map cv2.remap (Lanczos4)",
            "rust": "stellaRGB.warp Rust engine (Lanczos4; per-tract fitted "
            "WarpTransform, no dense coordinate maps)",
        },
        default="afw",
    )

    def setDefaults(self):
        self.warp.warpingKernelName = "lanczos5"
        return super().setDefaults()


class MetaMosaicTask(PipelineTask):
    """Assembles RGB patches from one or more tracts onto a common yx grid.

    A virtual tract is defined in memory: it shares the reference tract's WCS and
    origin but has sky extents large enough to enclose all input data, and carries
    a patch grid identical in geometry (pitch, size, borders) to the reference
    tract, so its patches overlap just like ordinary LSST patches.

    Every reference-tract patch is placed 1:1 into its matching virtual patch
    exactly (no resampling). Every foreign-tract patch is decoded once, warped
    once onto the box enclosing the virtual patches it overlaps, and each
    cell-sized slice of that warp is added separately. The warp backend is
    ``config.warp_backend``: "afw" uses the afwMath Warper (reference);
    "cv2" warps with ``cv2.remap`` over per-foreign-tract precomputed
    coordinate maps (see ``warp_maps``), which is much faster and validated
    to agree with the afw path; "rust" warps with the ``stellaRGB.warp``
    engine over a per-foreign-tract ``WarpTransform`` (a control-grid fit of
    the projection pair verified to ``fit_transform``'s tolerance — no dense
    coordinate maps are ever built; see :meth:`_fit_tract_transform`; a tract
    whose fit fails (off-domain or noninvertible projection pair) warns and
    falls back to the afw Warper for that tract alone). Every
    contributor is then feather-weighted directly into a single mosaic
    accumulation pass by the backend-independent ``FeatheredMosaicCreator``
    (the single holder of the masked-add/NaN-reweight contract) against a
    per-pixel weight accumulator, so a NaN in one contributor does not
    corrupt the overlap region; there is no per-cell pre-average.

    Missing data is left as NaN during assembly and replaced with zero only at the
    very end. No mask and no plugins.

    The output raster is cropped according to ``crop_mode``:
    "full_coverage" (default) crops to the largest rectangle of fully-covered
    virtual-patch cells (every pixel carries data); cells containing holes are
    excluded and outlier clusters outside the rectangle are dropped.
    "data_extent" crops to the tightest rectangle containing every input
    pixel, trimming only entirely empty outer rows/columns, so partial rows
    and interior holes between rotated footprints survive as black (useful
    for inspecting coverage). "none" keeps the full union rectangle.
    """

    _DefaultName = "metaMosaic"
    ConfigClass = MetaMosaicConfig

    config: ConfigClass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.warper = Warper.fromConfig(self.config.warp)

    @staticmethod
    def _to_float32(rgb: NDArray) -> NDArray:
        """Normalize an RGB array to float32 in [0, 1] for consistent blending."""
        match rgb.dtype:
            case np.uint8:
                return rgb.astype(np.float32) / 255.0
            case np.uint16:
                return rgb.astype(np.float32) / 65535.0
            case np.float16:
                return rgb.astype(np.float32)
            case _:
                return rgb.astype(np.float32)

    def _select_reference_tract(self, inputs_by_tract: dict[int, list]) -> int:
        """Return the tract id to use as the reference grid.

        Uses ``config.reference_tract`` if set, else the tract with the most
        input patches (ties broken by the lower tract id).

        Parameters
        ----------
        inputs_by_tract : `dict` of `int` to `list`
            Input handles grouped by tract id.

        Returns
        -------
        ref_tract : `int`
        """
        if self.config.reference_tract is not None:
            return self.config.reference_tract
        return max(inputs_by_tract, key=lambda t: (len(inputs_by_tract[t]), -t))

    def _ref_projection(self, ref_tract: int, ref_info: TractInfo, skymap_name: str) -> SkyProjection:
        """Build the reference tract's SkyProjection for the output ColorImage."""
        return SkyProjection.from_legacy(
            ref_info.getWcs(),
            TractFrame(
                skymap=skymap_name,
                tract=ref_tract,
                bbox=Box.from_legacy(ref_info.bbox),
            ),
        )

    def _scaled_projection(
        self,
        ref_wcs,
        ref_tract: int,
        skymap_name: str,
        full_origin: Point2D,
        bin_factor: int,
        out_box: Box2I,
    ) -> SkyProjection:
        """Build a SkyProjection for a ``bin_factor``-binned output image.

        The binned pixel grid is anchored so that binned pixel ``(i, j)`` maps to
        the same sky as full-resolution reference pixel
        ``full_origin + (i * bin_factor, j * bin_factor)``.

        Parameters
        ----------
        ref_wcs : `lsst.afw.geom.SkyWcs`
            The full-resolution reference tract WCS.
        ref_tract : `int`
            The reference tract id.
        skymap_name : `str`
            The skymap name.
        full_origin : `lsst.geom.Point2D`
            Sky-anchored pixel origin of the full-resolution mosaic in reference
            coordinates.
        bin_factor : `int`
            The binning factor.
        out_box : `lsst.geom.Box2I`
            The binned output bounding box.

        Returns
        -------
        proj : `lsst.images.SkyProjection`
        """
        crpix = ref_wcs.getPixelOrigin()
        crval = ref_wcs.getSkyOrigin()
        cd = ref_wcs.getCdMatrix()
        ctype = ref_wcs.getFitsMetadata().get("CTYPE1")
        projection = ctype.split("-")[-1] if ctype else "TAN"
        new_crpix = Point2D(
            (crpix.getX() - full_origin.getX()) / bin_factor,
            (crpix.getY() - full_origin.getY()) / bin_factor,
        )
        new_wcs = makeSkyWcs(crpix=new_crpix, crval=crval, cdMatrix=cd * bin_factor, projection=projection)
        return SkyProjection.from_legacy(
            new_wcs,
            TractFrame(skymap=skymap_name, tract=ref_tract, bbox=Box.from_legacy(out_box)),
        )

    def _map_patch_box_to_ref(
        self, patch_proj: SkyProjection, patch_bbox: Box2I, ref_proj: SkyProjection
    ) -> Box2I:
        """Map a foreign patch's outer bbox into reference-tract pixel coordinates.

        The four corners of ``patch_bbox`` are transformed patch->sky->ref pixels
        and enclosed in a box.

        Parameters
        ----------
        patch_proj : `lsst.images.SkyProjection`
            SkyProjection of the patch's tract.
        patch_bbox : `lsst.geom.Box2I`
            Outer bounding box of the patch.
        ref_proj : `lsst.images.SkyProjection`
            SkyProjection of the reference tract.

        Returns
        -------
        box : `lsst.geom.Box2I`
            The patch's location in the reference pixel grid.
        """
        xs = np.array(
            [
                patch_bbox.getBeginX(),
                patch_bbox.getEndX(),
                patch_bbox.getBeginX(),
                patch_bbox.getEndX(),
            ]
        )
        ys = np.array(
            [
                patch_bbox.getBeginY(),
                patch_bbox.getBeginY(),
                patch_bbox.getEndY(),
                patch_bbox.getEndY(),
            ]
        )
        sky = patch_proj.pixel_to_sky(x=xs, y=ys)
        pixels = ref_proj.sky_to_pixel(sky)
        px = np.asarray(pixels.x)
        py = np.asarray(pixels.y)
        return Box.from_float_bounds(
            x_min=float(px.min()),
            x_max=float(px.max()),
            y_min=float(py.min()),
            y_max=float(py.max()),
        ).to_legacy()

    def _fit_tract_transform(
        self,
        tract_info: TractInfo,
        tract_id: int,
        skymap_name: str,
        ref_proj: SkyProjection,
        union_box: Box2I,
        grow: int,
    ) -> WarpTransform:
        """Fit the "rust" backend's per-tract :class:`WarpTransform`.

        The fit samples the projection pair on a control grid only (the
        ``fit_transform`` adaptive-densification contract) — the anti-pattern
        this backend exists to eliminate is the dense per-tract coordinate-map
        build of the "cv2" path (measured ~303 s/tract at production scale).
        The returned transform is cached per foreign tract and reused by
        every patch of that tract; per-patch rendering happens at the
        patch's own dest box via ``warp_into``'s sub-box support.

        Parameters
        ----------
        tract_info : `lsst.skymap.TractInfo`
            The foreign tract.
        tract_id : `int`
            The foreign tract id (frame label only).
        skymap_name : `str`
            The skymap name (frame label only).
        ref_proj : `lsst.images.SkyProjection`
            Reference-tract projection defining the destination pixel frame
            (the mosaic's virtual-tract grid coordinates).
        union_box : `lsst.geom.Box2I`
            Union of the virtual-patch cells claimed by this tract's patches,
            in reference pixel coordinates (the transform's destination box).
        grow : `int`
            Pixels to grow the source bbox by, so an affine-mode fit cannot
            trip its D7 containment guard: the transform's affine footprint
            is the corner-image rectangle of the source bbox, and claimed
            cell boxes snap to the virtual-patch grid, which can overshoot
            the mapped tract footprint by (at most) one cell width plus the
            fit tolerance times the pixel-space Jacobian (~1 for skymap
            tracts). ``grow`` is set to at least one cell width plus slack.

        Returns
        -------
        transform : `stellaRGB.warp.WarpTransform`
            Maps- or affine-mode transform (per the fit's honest residual
            report) whose ``dest_bbox`` contains ``union_box`` (D7).
        """
        tract_bbox = tract_info.bbox
        src_proj = SkyProjection.from_legacy(
            tract_info.getWcs(),
            TractFrame(skymap=skymap_name, tract=tract_id, bbox=Box.from_legacy(tract_bbox)),
        )
        src_box = Box2I(tract_bbox)
        src_box.grow(grow)
        return fit_transform(
            src_proj,
            Box.from_legacy(src_box),
            ref_proj,
            Box.from_legacy(union_box),
        )

    def _warp_patch_onto_cell(
        self, rgb: NDArray, patch_wcs, ref_wcs, cell_box: Box2I, patch_bbox: Box2I
    ) -> NDArray:
        """Warp a single patch RGB image onto a cell's grid, per channel.

        The output is exactly ``cell_box`` in size; pixels the source does not
        cover are left as NaN (the missing-data marker).

        Parameters
        ----------
        rgb : `numpy.ndarray`
            Float32 RGB patch array in [0, 1].
        patch_wcs : `lsst.afw.geom.SkyWcs`
            WCS of the patch's tract.
        ref_wcs : `lsst.afw.geom.SkyWcs`
            WCS of the reference tract.
        cell_box : `lsst.geom.Box2I`
            The output box (a reference-patch-sized cell) in reference pixel coords.
        patch_bbox : `lsst.geom.Box2I`
            Bounding box of the patch (bounds for the source image).

        Returns
        -------
        warped : `numpy.ndarray`
            Cell-sized RGB array (height, width, 3) with NaN where there is no data.
        """
        channels = []
        for channel_index in range(3):
            tmp = ImageF(patch_bbox)
            tmp.array[:, :] = rgb[..., channel_index]
            warped = self.warper.warpImage(ref_wcs, tmp, patch_wcs, destBBox=cell_box)
            channels.append(warped.array)
        return np.stack(channels, axis=-1)

    def _data_bbox(
        self,
        mosaic_shape: tuple[int, int],
        weight: np.ndarray,
        mode: str = "full_coverage",
        band_rows: int = 4096,
    ) -> Box2I | None:
        """Find the bounding box of the mosaic pixels that received data.

        The single-channel ``weight`` array is scanned in row bands so no
        full-array boolean temporary is materialized when it is memmap-backed.
        A pixel counts as having data when its accumulated weight is strictly
        positive (the same validity rule :meth:`FeatheredMosaicCreator.finalize`
        normalizes on); a row or column survives the reduction when it contains
        at least one such pixel. This implements the "data_extent" crop mode;
        "full_coverage" is computed by :meth:`_full_coverage_bbox`.

        Parameters
        ----------
        mosaic_shape : `tuple` of `int`
            The full-resolution mosaic shape, ``(height, width)``.
        weight : `numpy.ndarray`
            Two-dimensional ``(height, width)`` weight accumulator.
        mode : `str`, optional
            The crop mode (``config.crop_mode``), carried through for
            provenance.
        band_rows : `int`, optional
            Number of rows processed per band. Default is 4096.

        Returns
        -------
        bbox : `lsst.geom.Box2I` | `None`
            The enclosing box in mosaic-local pixel coordinates (inclusive), or
            `None` if no pixel has weight anywhere (a fully empty mosaic);
            crop modes treat that as a fatal empty input (`ValueError`).
        """
        n_rows, n_cols = mosaic_shape
        min_row: int | None = None
        max_row: int | None = None
        min_col: int | None = None
        max_col: int | None = None
        for y0 in range(0, n_rows, band_rows):
            band = np.asarray(weight[y0 : min(y0 + band_rows, n_rows)])
            valid = band > 0
            rows = np.flatnonzero(valid.any(axis=1))
            if rows.size == 0:
                continue
            if min_row is None:
                min_row = y0 + int(rows[0])
            max_row = y0 + int(rows[-1])
            cols = np.flatnonzero(valid.any(axis=0))
            if min_col is None:
                min_col = int(cols[0])
                max_col = int(cols[-1])
            else:
                min_col = min(min_col, int(cols[0]))
                max_col = max(max_col, int(cols[-1]))
        if min_row is None:
            self.log.warning("Mosaic has no weighted pixels; skipping the output crop.")
            return None
        assert min_col is not None and max_col is not None
        return Box2I(
            Point2I(min_col, min_row),
            Extent2I(max_col - min_col + 1, max_row - min_row + 1),
        )

    def _full_coverage_bbox(
        self,
        mosaic_shape: tuple[int, int],
        weight: np.ndarray,
        *,
        pitch_x: int,
        pitch_y: int,
        cell_w: int,
        cell_h: int,
        ncx: int,
        ncy: int,
        ref_cell: tuple[int, int] | None = None,
    ) -> Box2I | None:
        """Find the largest cell-aligned rectangle of fully covered mosaic cells.

        A virtual-patch cell with canvas-local index ``(ci, cj)`` is fully
        covered when every pixel of its canvas-clipped window, origin
        ``(ci * pitch_x, cj * pitch_y)`` and size ``(cell_w, cell_h)``, has
        strictly positive weight, i.e. received data from at least one
        contributor. A cell whose window contains any uncovered (NaN-hole)
        pixel is excluded by construction, so the returned rectangle is
        guaranteed NaN-free at cell granularity; the border is deliberately
        not refined to the pixel (conservative, cell-aligned by design).

        The largest all-valid rectangle on the ``ncy x ncx`` boolean cell grid
        is found exactly with the standard histogram-of-column-heights plus
        monotone-stack algorithm (the grid is small; plain Python is fine).
        Rectangles of equal maximal area are tie-broken by preferring one
        that contains ``ref_cell``, then smaller minimum ``j``, then smaller
        minimum ``i`` (top-left).

        Parameters
        ----------
        mosaic_shape : `tuple` of `int`
            The full-resolution mosaic shape, ``(height, width)``.
        weight : `numpy.ndarray`
            Two-dimensional ``(height, width)`` weight accumulator.
        pitch_x : `int`
            Horizontal cell pitch.
        pitch_y : `int`
            Vertical cell pitch.
        cell_w : `int`
            Cell outer width.
        cell_h : `int`
            Cell outer height.
        ncx : `int`
            Number of virtual-patch cell columns on the canvas.
        ncy : `int`
            Number of virtual-patch cell rows on the canvas.
        ref_cell : `tuple` of `int`, optional
            Canvas-local ``(i, j)`` cell index of a reference-tract patch,
            used to tie-break equal-area rectangles. Default is None.

        Returns
        -------
        bbox : `lsst.geom.Box2I` | `None`
            The cell-aligned box in mosaic-local pixel coordinates (inclusive
            corners, matching :meth:`_data_bbox`), or `None` if no cell is
            fully valid; crop modes treat that as a fatal empty input
            (`ValueError`).
        """
        n_rows, n_cols = mosaic_shape
        valid_cells = np.zeros((ncy, ncx), dtype=bool)
        for cj in range(ncy):
            y0 = cj * pitch_y
            y1 = min(y0 + cell_h, n_rows)
            for ci in range(ncx):
                x0 = ci * pitch_x
                x1 = min(x0 + cell_w, n_cols)
                valid_cells[cj, ci] = bool((weight[y0:y1, x0:x1] > 0).all())
        if not valid_cells.any():
            self.log.warning("Mosaic has no fully covered cell; skipping the output crop.")
            return None

        def _contains(rect: tuple[int, int, int, int], ci: int, cj: int) -> bool:
            ri0, ri1, rj0, rj1 = rect
            return ri0 <= ci <= ri1 and rj0 <= cj <= rj1

        def _better(candidate: tuple[int, int, int, int], incumbent: tuple[int, int, int, int]) -> bool:
            """Apply the documented maximal-area / tie-break ordering."""
            ci0, ci1, cj0, cj1 = candidate
            bi0, bi1, bj0, bj1 = incumbent
            area = (ci1 - ci0 + 1) * (cj1 - cj0 + 1)
            best_area = (bi1 - bi0 + 1) * (bj1 - bj0 + 1)
            if area != best_area:
                return area > best_area
            if ref_cell is not None:
                has_ref = _contains(candidate, *ref_cell)
                best_has_ref = _contains(incumbent, *ref_cell)
                if has_ref != best_has_ref:
                    return has_ref
            if cj0 != bj0:
                return cj0 < bj0
            return ci0 < bi0

        best: tuple[int, int, int, int] | None = None

        def _consider(ci0: int, ci1: int, cj0: int, cj1: int) -> None:
            nonlocal best
            candidate = (ci0, ci1, cj0, cj1)
            if best is None or _better(candidate, best):
                best = candidate

        # Exact largest-rectangle-in-a-boolean-grid via column heights and a
        # monotone stack (histogram bars = consecutive valid cells per column).
        heights = np.zeros(ncx, dtype=int)
        for cj in range(ncy):
            heights += 1
            heights[~valid_cells[cj]] = 0
            stack: list[tuple[int, int]] = []  # (start column, bar height)
            for ci in range(ncx + 1):
                bar = int(heights[ci]) if ci < ncx else 0
                start = ci
                while stack and stack[-1][1] > bar:
                    s_start, s_height = stack.pop()
                    _consider(s_start, ci - 1, cj - s_height + 1, cj)
                    start = s_start
                if not stack or stack[-1][1] < bar:
                    stack.append((start, bar))
        assert best is not None
        bi0, bi1, bj0, bj1 = best
        x0 = bi0 * pitch_x
        y0 = bj0 * pitch_y
        x1 = min(bi1 * pitch_x + cell_w, n_cols)
        y1 = min(bj1 * pitch_y + cell_h, n_rows)
        return Box2I(Point2I(x0, y0), Extent2I(x1 - x0, y1 - y0))

    def run(
        self,
        inputRGB: Iterable[DeferredDatasetHandle],
        skyMap: BaseSkyMap,
    ) -> Struct:
        """Assemble patches from one or more tracts into one RGB mosaic.

        Parameters
        ----------
        inputRGB : `Iterable` of `~lsst.daf.butler.DeferredDatasetHandle`
            Deferred handles to the per-patch RGB images.
        skyMap : `BaseSkyMap`
            The skymap that defines the relative position of the inputs.

        Returns
        -------
        result : `Struct`
            The `Struct` with attribute ``outputRGBMosaic``.
        """
        inputRGB = list(inputRGB)
        if not inputRGB:
            raise NoWorkFound("No RGB images to mosaic")

        # Group inputs by tract and capture the skymap name.
        inputs_by_tract: dict[int, list] = {}
        skymap_name: str | None = None
        for handle in inputRGB:
            tract = handle.dataId["tract"]
            if skymap_name is None:
                skymap_name = handle.dataId["skymap"]
            inputs_by_tract.setdefault(tract, []).append(handle)
        assert skymap_name is not None

        ref_tract = self._select_reference_tract(inputs_by_tract)
        ref_info: TractInfo = skyMap[ref_tract]
        ref_wcs = ref_info.getWcs()
        ref_proj = self._ref_projection(ref_tract, ref_info, skymap_name)

        # Virtual tract geometry: identical to the reference tract's patch grid,
        # so each reference patch maps 1:1 to one virtual patch.
        ref_patch_bbox = ref_info[0].getOuterBBox()
        cell_w = ref_patch_bbox.getWidth()
        cell_h = ref_patch_bbox.getHeight()
        grid_origin = ref_patch_bbox.getBegin()
        nx = ref_info.num_patches.x
        ny = ref_info.num_patches.y
        pitch_x = ref_info[1].getOuterBBox().getBeginX() - grid_origin.getX() if nx >= 2 else cell_w
        pitch_y = ref_info[nx].getOuterBBox().getBeginY() - grid_origin.getY() if ny >= 2 else cell_h
        patch_grow = ref_info[0].getCellInnerDimensions().getX()

        # Map every reference-tract patch to its matching virtual patch position.
        ref_content: dict[tuple[int, int], DeferredDatasetHandle] = {}
        for handle in inputs_by_tract.get(ref_tract, []):
            bbox = ref_info[handle.dataId["patch"]].getOuterBBox()
            i = round((bbox.getBeginX() - grid_origin.getX()) / pitch_x)
            j = round((bbox.getBeginY() - grid_origin.getY()) / pitch_y)
            ref_content[(i, j)] = handle

        # For each foreign tract patch, collect the virtual patches (cells) it
        # overlaps that have no reference-tract content. Each patch is kept as one
        # record so it is later decoded and warped exactly once for all of its
        # cells.
        # One record per foreign patch: tract id, handle, patch info/WCS, the
        # virtual patch cells it claims, the reference-frame cell boxes and
        # their enclosing box (both warp backends warp onto that union box).
        foreign_records: list[tuple[int, DeferredDatasetHandle, PatchInfo, SkyWcs, list, list, Box2I]] = []
        for tract, handles in inputs_by_tract.items():
            if tract == ref_tract:
                continue
            tract_info: TractInfo = skyMap[tract]
            for handle in handles:
                patch = handle.dataId["patch"]
                patch_info: PatchInfo = tract_info[patch]
                patch_wcs = patch_info.getWcs()
                patch_bbox = patch_info.getOuterBBox()
                patch_proj = SkyProjection.from_legacy(
                    patch_wcs,
                    TractFrame(
                        skymap=skymap_name,
                        tract=tract,
                        bbox=Box.from_legacy(tract_info.bbox),
                    ),
                )
                mapped = self._map_patch_box_to_ref(patch_proj, patch_bbox, ref_proj)
                i0 = int(np.floor((mapped.getBeginX() - grid_origin.getX() - cell_w) / pitch_x)) + 1
                i1 = int(np.floor((mapped.getEndX() - 1 - grid_origin.getX()) / pitch_x))
                j0 = int(np.floor((mapped.getBeginY() - grid_origin.getY() - cell_h) / pitch_y)) + 1
                j1 = int(np.floor((mapped.getEndY() - 1 - grid_origin.getY()) / pitch_y))
                cells = [
                    (i, j)
                    for i in range(i0, i1 + 1)
                    for j in range(j0, j1 + 1)
                    if (i, j) not in ref_content
                ]
                if cells:
                    cell_boxes = [
                        Box2I(
                            Point2I(grid_origin.getX() + i * pitch_x, grid_origin.getY() + j * pitch_y),
                            Extent2I(cell_w, cell_h),
                        )
                        for (i, j) in cells
                    ]
                    # Box2I begin/end are INCLUSIVE; include() grows the box to cover.
                    dest_box = Box2I(cell_boxes[0])
                    for cbox in cell_boxes[1:]:
                        dest_box.include(cbox)
                    foreign_records.append(
                        (tract, handle, patch_info, patch_wcs, cells, cell_boxes, dest_box)
                    )

        positions = set(ref_content) | {
            cell for _, _, _, _, cells, _, _ in foreign_records for cell in cells
        }
        if not positions:
            raise NoWorkFound("No valid RGB images to mosaic")

        min_i = min(i for i, j in positions)
        max_i = max(i for i, j in positions)
        min_j = min(j for i, j in positions)
        max_j = max(j for i, j in positions)
        ncx = max_i - min_i + 1
        ncy = max_j - min_j + 1
        mosaic_w = (ncx - 1) * pitch_x + cell_w
        mosaic_h = (ncy - 1) * pitch_y + cell_h

        # Allocate the weighted-sum mosaic and its weight accumulator on disk.
        self.imageHandle = tempfile.NamedTemporaryFile(dir="." if self.config.use_local_temp else None)
        mosaic = np.memmap(
            self.imageHandle.name,
            mode="w+",
            shape=(mosaic_h, mosaic_w, 3),
            dtype=np.float32,
        )
        mosaic[...] = 0.0
        self.weightHandle = tempfile.NamedTemporaryFile(dir="." if self.config.use_local_temp else None)
        weight = np.memmap(
            self.weightHandle.name,
            mode="w+",
            shape=(mosaic_h, mosaic_w),
            dtype=np.float32,
        )
        weight[...] = 0.0
        self.log.info("Mosaic %d x %d virtual patches (%d x %d px)", ncx, ncy, mosaic_w, mosaic_h)

        # Phase timings (wall clock, accumulated over all inputs) and, for the
        # "cv2" warp backend, the per-foreign-tract coordinate-map cache, or
        # for the "rust" backend the per-foreign-tract WarpTransform cache,
        # both built over the union of each tract's claimed cell dest boxes.
        # The cache is local to this run(), so it is cleared at the start of
        # every run. For "rust", "map-build" times the control-grid fits
        # (one per foreign tract, amortized ~0 vs the cv2 dense-map build);
        # warp+feather-accumulate stay separate columns: warp_into times into
        # "warp=", the backend-independent FeatheredMosaicCreator.add_to_image
        # feathered masked-adds (the D12 contract holder) time into
        # "accumulate=" exactly as for the other backends.
        use_cv2 = self.config.warp_backend == "cv2"
        use_rust = self.config.warp_backend == "rust"
        t_decode = t_warp = t_map = t_add = t_finalize = t_crop = 0.0
        n_decode = n_warp = n_map = n_add = 0
        tract_maps: dict[int, tuple[NDArray, NDArray, Box2I, Point2I, float]] = {}
        tract_transforms: dict[int, WarpTransform] = {}
        tract_union: dict[int, Box2I] = {}
        if use_cv2 or use_rust:
            for tract, _, _, _, _, _, dest_box in foreign_records:
                if tract in tract_union:
                    tract_union[tract].include(dest_box)
                else:
                    tract_union[tract] = Box2I(dest_box)

        # Assembly happens in a try block so the temp-file handles are closed on
        # failure (the output memmap must stay open on success).
        try:
            # Setup color space conversion in case it is used.
            if self.config.do_dci_d65_convert:
                d65 = copy.deepcopy(colour.models.RGB_COLOURSPACE_DCI_P3)
                dp3 = copy.deepcopy(colour.models.RGB_COLOURSPACE_DISPLAY_P3)
                d65.whitepoint = dp3.whitepoint
                d65.whitepoint_name = dp3.whitepoint_name

            input_metadata: dict = {}
            is_hdr = False
            dci_skip_warned = False
            # Feather at full resolution: binning is applied only as the final
            # whole-mosaic resize below, so pass bin_factor=1 to the featherer.
            mosaic_maker = FeatheredMosaicCreator(patch_grow)
            full_box = Box.from_legacy(Box2I(Point2I(0, 0), Extent2I(mosaic_w, mosaic_h)))

            # Reference-tract patches are placed 1:1 into their matching virtual
            # patch (no resampling) and feather-weighted into the mosaic.
            for (i, j), handle in ref_content.items():
                local_x = (i - min_i) * pitch_x
                local_y = (j - min_j) * pitch_y
                box_local = Box.from_legacy(
                    Box2I(Point2I(local_x, local_y), Extent2I(cell_w, cell_h))
                )
                t0 = time.perf_counter()
                rgb_image = handle.get()
                input_metadata = dict(getattr(rgb_image, "metadata", None) or {})
                is_hdr = bool(input_metadata.get("is_hdr"))
                content = self._to_float32(rgb_image.array)
                if self.config.do_dci_d65_convert and not is_hdr:
                    content = colour.RGB_to_RGB(np.clip(content, 0, 1), dp3, d65)
                elif self.config.do_dci_d65_convert and not dci_skip_warned:
                    self.log.warning("Skipping DCI-D65 conversion for HDR (is_hdr) input.")
                    dci_skip_warned = True
                t_decode += time.perf_counter() - t0
                n_decode += 1
                t0 = time.perf_counter()
                mosaic_maker.add_to_image(mosaic, content, full_box, box_local, reverse=False, weight=weight)
                t_add += time.perf_counter() - t0
                n_add += 1

            # Each foreign patch is decoded and warped exactly once onto the box
            # enclosing all of its cells, then each cell's slice of that single
            # warp is feather-weighted into the mosaic. The warp never lives
            # longer than one foreign patch; the mosaic memmap is the only
            # accumulator.
            for tract, handle, patch_info, patch_wcs, cells, cell_boxes, dest_box in foreign_records:
                t0 = time.perf_counter()
                rgb_image = handle.get()
                input_metadata = dict(getattr(rgb_image, "metadata", None) or {})
                is_hdr = bool(input_metadata.get("is_hdr"))
                rgb = self._to_float32(rgb_image.array)
                if self.config.do_dci_d65_convert and not is_hdr:
                    rgb = colour.RGB_to_RGB(np.clip(rgb, 0, 1), dp3, d65)
                elif self.config.do_dci_d65_convert and not dci_skip_warned:
                    self.log.warning("Skipping DCI-D65 conversion for HDR (is_hdr) input.")
                    dci_skip_warned = True
                t_decode += time.perf_counter() - t0
                n_decode += 1
                patch_bbox = patch_info.getOuterBBox()
                if use_rust and tract not in tract_transforms:
                    # One fit per foreign tract; a failed fit is cached as the
                    # None sentinel so later patches of the tract neither retry
                    # the fit nor re-warn. Fit failures (noninvertible fits,
                    # off-domain projections, containment shortfalls) are
                    # robustness events, not fatal ones: warn once per tract
                    # and serve the tract from the afw Warper below.
                    tm0 = time.perf_counter()
                    try:
                        transform = self._fit_tract_transform(
                            skyMap[tract],
                            tract,
                            skymap_name,
                            ref_proj,
                            tract_union[tract],
                            max(cell_w, cell_h) + 4,
                        )
                        n_map += 1
                    except Exception as exc:
                        self.log.warning(
                            "WarpTransform fit failed for foreign tract %s (%r); "
                            "falling back to the afw Warper for this tract.",
                            tract,
                            exc,
                        )
                        transform = None
                    t_map += time.perf_counter() - tm0
                    tract_transforms[tract] = transform
                rust_transform = tract_transforms.get(tract)
                if rust_transform is not None:
                    t0 = time.perf_counter()
                    # The source is passed as a ColorImage so the transform's
                    # tract-pixel source frame sees the patch at its own bbox
                    # (a bare ndarray would be declared at origin (0,0)).
                    # warp_into renders exactly this patch's dest box (the
                    # transform's dest_bbox is the tract-wide union box, so
                    # this is a contained sub-box render; no dense maps and
                    # no canvas-sized temporaries — the per-patch buffer is
                    # dest_box-sized, the same profile as the cv2 branch).
                    warped = warp_into(
                        ColorImage(rgb, bbox=Box.from_legacy(patch_bbox)),
                        rust_transform,
                        kernel=_RUST_KERNEL,
                        dest_bbox=Box.from_legacy(dest_box),
                    )
                    t_warp += time.perf_counter() - t0
                    n_warp += 1
                elif use_cv2:
                    if tract not in tract_maps:
                        tm0 = time.perf_counter()
                        tract_bbox = skyMap[tract].bbox
                        union_box = tract_union[tract]
                        map_x, map_y = warp_maps(patch_wcs, tract_bbox, ref_wcs, union_box)
                        # Mirror the (kernel-independent) pixel-area scaling
                        # afwMath's Warper applies so both backends agree.
                        scale = probe_warp_scale(self.warper, patch_wcs, ref_wcs, patch_bbox)
                        tract_maps[tract] = (map_x, map_y, union_box, tract_bbox.getBegin(), scale)
                        t_map += time.perf_counter() - tm0
                        n_map += 1
                    map_x, map_y, union_box, tract_origin, scale = tract_maps[tract]
                    # Slice the cached tract-wide map (dest coordinates over the
                    # union of the tract's cell boxes) down to this patch's
                    # dest box, then shift from the tract parent frame into
                    # this patch's array coordinates (cv2 indexes from the
                    # source array origin, i.e. patch_bbox.getMin()).
                    sx0 = dest_box.getMinX() - union_box.getMinX()
                    sy0 = dest_box.getMinY() - union_box.getMinY()
                    sx1 = sx0 + dest_box.getWidth()
                    sy1 = sy0 + dest_box.getHeight()
                    off_x = patch_bbox.getMinX() - tract_origin.getX()
                    off_y = patch_bbox.getMinY() - tract_origin.getY()
                    mx = map_x[sy0:sy1, sx0:sx1] - off_x
                    my = map_y[sy0:sy1, sx0:sx1] - off_y
                    t0 = time.perf_counter()
                    warped = np.empty((dest_box.getHeight(), dest_box.getWidth(), 3), dtype=np.float32)
                    for channel_index in range(3):
                        warped[..., channel_index] = scale * cv2.remap(
                            np.ascontiguousarray(rgb[..., channel_index]),
                            mx,
                            my,
                            cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=np.nan,
                        )
                    t_warp += time.perf_counter() - t0
                    n_warp += 1
                else:
                    t0 = time.perf_counter()
                    warped = self._warp_patch_onto_cell(rgb, patch_wcs, ref_wcs, dest_box, patch_bbox)
                    t_warp += time.perf_counter() - t0
                    n_warp += 1
                for (i, j), cbox in zip(cells, cell_boxes, strict=True):
                    x0 = cbox.getMinX() - dest_box.getMinX()
                    y0 = cbox.getMinY() - dest_box.getMinY()
                    # Box2I max is INCLUSIVE: slice end is max+1.
                    content = warped[y0 : y0 + cbox.getHeight(), x0 : x0 + cbox.getWidth()]
                    # Skip cells the warp did not cover at all (the slice is
                    # RAM-resident): they would add no data and no weight, only
                    # inflating the mosaic extent with zero-weight (black) cells.
                    if not np.isfinite(content).any():
                        continue
                    box_local = Box.from_legacy(
                        Box2I(
                            Point2I((i - min_i) * pitch_x, (j - min_j) * pitch_y),
                            Extent2I(cell_w, cell_h),
                        )
                    )
                    t0 = time.perf_counter()
                    mosaic_maker.add_to_image(
                        mosaic, content, full_box, box_local, reverse=False, weight=weight
                    )
                    t_add += time.perf_counter() - t0
                    n_add += 1

            t0 = time.perf_counter()
            FeatheredMosaicCreator.finalize(mosaic, weight)
            t_finalize += time.perf_counter() - t0
            # The data extent must be measured from the weight accumulator while
            # it is still valid, i.e. before the NaN-to-zero pass below.
            t0 = time.perf_counter()
            crop_mode = self.config.crop_mode
            crop_box = None
            if crop_mode == "data_extent":
                crop_box = self._data_bbox((mosaic_h, mosaic_w), weight, crop_mode)
            elif crop_mode == "full_coverage":
                # Anchor the tie-break on the first reference-tract patch's
                # canvas-local cell (reference content may be empty when a
                # non-empty reference tract was forced via config).
                ref_cell = None
                if ref_content:
                    ref_i, ref_j = next(iter(ref_content))
                    ref_cell = (ref_i - min_i, ref_j - min_j)
                crop_box = self._full_coverage_bbox(
                    (mosaic_h, mosaic_w),
                    weight,
                    pitch_x=pitch_x,
                    pitch_y=pitch_y,
                    cell_w=cell_w,
                    cell_h=cell_h,
                    ncx=ncx,
                    ncy=ncy,
                    ref_cell=ref_cell,
                )
            if crop_mode != "none" and crop_box is None:
                raise ValueError(
                    f"Crop mode '{crop_mode}' cannot crop a mosaic with no pixels "
                    "that received data."
                )
            # The crop box is inclusive in full-resolution mosaic coordinates;
            # without one the full rectangle is used unchanged.
            if crop_box is not None:
                cx0, cy0 = crop_box.getMinX(), crop_box.getMinY()
                cx1, cy1 = crop_box.getMaxX(), crop_box.getMaxY()
            else:
                cx0 = cy0 = 0
                cx1 = mosaic_w - 1
                cy1 = mosaic_h - 1
            crop_w = cx1 - cx0 + 1
            crop_h = cy1 - cy0 + 1
            if crop_w != mosaic_w or crop_h != mosaic_h:
                self.log.info("Cropped output to %dx%d of %dx%d px", crop_w, crop_h, mosaic_w, mosaic_h)
                # Fraction of cropped-region pixels with no finite contributor,
                # counted in bands over the weight slice (no full-canvas float).
                empty_pixels = 0
                for y0 in range(cy0, cy1 + 1, 4096):
                    band = np.asarray(weight[y0 : min(y0 + 4096, cy1 + 1), cx0 : cx1 + 1])
                    empty_pixels += int(np.count_nonzero(band == 0.0))
                self.log.info(
                    "Crop mode %s: mosaic cropped to %dx%d px; retaining %.1f%% empty pixels",
                    crop_mode,
                    crop_w,
                    crop_h,
                    100.0 * empty_pixels / (crop_w * crop_h),
                )
            # Replace the NaN missing-data marker with zero in row bands (views
            # write through to the memmap; no full-array boolean temporary).
            # Only the rows kept by the crop need conversion; rows outside the
            # crop window are discarded anyway. Every pixel of the output view
            # lies within these rows, so interior holes still become zero.
            for y0 in range(cy0, cy1 + 1, 4096):
                np.nan_to_num(mosaic[y0 : min(y0 + 4096, cy1 + 1)], copy=False, nan=0.0)
            mosaic.flush()
            t_crop += time.perf_counter() - t0
            self.weightHandle.close()
            del weight

            # Assemble the output (optionally cropped and binned) and its WCS
            # metadata.
            bin_factor = self.config.bin_factor
            if bin_factor > 1:
                new_w = int(np.floor(crop_w / bin_factor))
                new_h = int(np.floor(crop_h / bin_factor))
                if new_w < 1 or new_h < 1:
                    raise ValueError(
                        f"bin_factor {bin_factor} exceeds cropped output dimensions "
                        f"{crop_w} x {crop_h} px; the binned output would be empty."
                    )
                # Crop first, then resize the cropped view.
                cropped = mosaic[cy0 : cy1 + 1, cx0 : cx1 + 1]
                out_array = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
                out_origin = Point2I(0, 0)
                out_box = Box2I(out_origin, Extent2I(new_w, new_h))
                full_origin = Point2D(
                    grid_origin.getX() + min_i * pitch_x + cx0,
                    grid_origin.getY() + min_j * pitch_y + cy0,
                )
                sky_proj = self._scaled_projection(
                    ref_wcs, ref_tract, skymap_name, full_origin, bin_factor, out_box
                )
            else:
                # A memmap view of the mosaic; same lifetime contract as before
                # (the image handle stays open until runQuantum.put completes).
                out_array = mosaic[cy0 : cy1 + 1, cx0 : cx1 + 1]
                out_origin = Point2I(
                    grid_origin.getX() + min_i * pitch_x + cx0,
                    grid_origin.getY() + min_j * pitch_y + cy0,
                )
                out_box = Box2I(out_origin, Extent2I(out_array.shape[1], out_array.shape[0]))
                # The projection is the reference WCS; only the array bbox shifts,
                # and SkyProjection.from_legacy's TractFrame bbox describes the
                # tract, not the output array.
                sky_proj = ref_proj
            result = Struct(
                outputRGBMosaic=ColorImage(
                    out_array,
                    bbox=Box.from_legacy(out_box),
                    sky_projection=sky_proj,
                    metadata=input_metadata,
                )
            )
        except BaseException:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(self, attr):
                    getattr(self, attr).close()
            raise

        if hasattr(self, "weightHandle"):
            self.weightHandle.close()
        self.log.info(
            "Timings: decode=%.1fs (n=%d), warp=%.1fs (n=%d), map-build=%.1fs (n=%d), "
            "accumulate=%.1fs (n=%d), finalize=%.1fs, crop=%.1fs",
            t_decode,
            n_decode,
            t_warp,
            n_warp,
            t_map,
            n_map,
            t_add,
            n_add,
            t_finalize,
            t_crop,
        )
        return result

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)
        if hasattr(self, "imageHandle"):
            self.imageHandle.close()
        if hasattr(self, "weightHandle"):
            self.weightHandle.close()
