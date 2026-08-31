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
import warnings
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
from lsst.images import Box, ColorImage, SkyProjection, TractFrame
from lsst.pex.config import ConfigField, Field
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

from ._utils import FeatheredMosaicCreator

if TYPE_CHECKING:
    from numpy.typing import NDArray
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
    warp = ConfigField[Warper.ConfigClass](doc="Warper configuration")

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
    exactly (no resampling) and feathered against its neighbours at the mosaic
    level. Every other virtual patch gathers the foreign-tract patches that
    overlap it, warps each onto that patch's grid, and combines them with a
    nan-mean coaddition. All virtual patches are then blended into the output
    mosaic with a NaN-reweighted feather (``FeatheredMosaicCreator``) so that a
    NaN in one contributor does not corrupt the overlap region.

    Missing data is left as NaN during assembly and replaced with zero only at the
    very end. No mask and no plugins.
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

        # For each "other" virtual patch, collect the foreign patches overlapping it.
        foreign_contributors: dict[tuple[int, int], list] = {}
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
                for i in range(i0, i1 + 1):
                    for j in range(j0, j1 + 1):
                        if (i, j) not in ref_content:
                            foreign_contributors.setdefault((i, j), []).append((handle, patch_info))

        positions = set(ref_content) | set(foreign_contributors)
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
            shape=(mosaic_h, mosaic_w, 3),
            dtype=np.float32,
        )
        weight[...] = 0.0
        self.log.info("Mosaic %d x %d virtual patches (%d x %d px)", ncx, ncy, mosaic_w, mosaic_h)

        # Setup color space conversion in case it is used.
        if self.config.do_dci_d65_convert:
            d65 = copy.deepcopy(colour.models.RGB_COLOURSPACE_DCI_P3)
            dp3 = copy.deepcopy(colour.models.RGB_COLOURSPACE_DISPLAY_P3)
            d65.whitepoint = dp3.whitepoint
            d65.whitepoint_name = dp3.whitepoint_name

        # Assembly happens in a try block so the temp-file handles are closed on
        # failure (the output memmap must stay open on success).
        try:
            # Setup color space conversion in case it is used.
            if self.config.do_dci_d65_convert:
                d65 = copy.deepcopy(colour.models.RGB_COLOURSPACE_DCI_P3)
                dp3 = copy.deepcopy(colour.models.RGB_COLOURSPACE_DISPLAY_P3)
                d65.whitepoint = dp3.whitepoint
                d65.whitepoint_name = dp3.whitepoint_name

            # Feather at full resolution: binning is applied only as the final
            # whole-mosaic resize below, so pass bin_factor=1 to the featherer.
            mosaic_maker = FeatheredMosaicCreator(patch_grow)
            full_box = Box.from_legacy(Box2I(Point2I(0, 0), Extent2I(mosaic_w, mosaic_h)))
            for i in range(min_i, max_i + 1):
                for j in range(min_j, max_j + 1):
                    if (i, j) not in positions:
                        continue
                    local_x = (i - min_i) * pitch_x
                    local_y = (j - min_j) * pitch_y
                    box_local = Box.from_legacy(
                        Box2I(Point2I(local_x, local_y), Extent2I(cell_w, cell_h))
                    )

                    if (i, j) in ref_content:
                        content = self._to_float32(ref_content[(i, j)].get().array)
                        if self.config.do_dci_d65_convert:
                            content = colour.RGB_to_RGB(np.clip(content, 0, 1), dp3, d65)
                    else:
                        cell_box = Box2I(
                            Point2I(grid_origin.getX() + i * pitch_x, grid_origin.getY() + j * pitch_y),
                            Extent2I(cell_w, cell_h),
                        )
                        stack = []
                        for handle, patch_info in foreign_contributors[(i, j)]:
                            rgb = self._to_float32(handle.get().array)
                            if self.config.do_dci_d65_convert:
                                rgb = colour.RGB_to_RGB(np.clip(rgb, 0, 1), dp3, d65)
                            warped = self._warp_patch_onto_cell(
                                rgb, patch_info.getWcs(), ref_wcs, cell_box, patch_info.getOuterBBox()
                            )
                            stack.append(warped)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            content = np.nanmean(np.stack(stack, axis=0), axis=0)

                    mosaic_maker.add_to_image(
                        mosaic, content, full_box, box_local, reverse=False, weight=weight
                    )

            FeatheredMosaicCreator.finalize(mosaic, weight)
            mosaic[np.isnan(mosaic)] = 0.0
            mosaic.flush()
            self.weightHandle.close()
            del weight

            # Assemble the output (optionally binned) and its WCS metadata.
            bin_factor = self.config.bin_factor
            if bin_factor > 1:
                new_w = int(np.floor(mosaic_w / bin_factor))
                new_h = int(np.floor(mosaic_h / bin_factor))
                out_array = cv2.resize(mosaic, (new_w, new_h), interpolation=cv2.INTER_AREA)
                out_origin = Point2I(0, 0)
                out_box = Box2I(out_origin, Extent2I(new_w, new_h))
                full_origin = Point2D(
                    grid_origin.getX() + min_i * pitch_x, grid_origin.getY() + min_j * pitch_y
                )
                sky_proj = self._scaled_projection(
                    ref_wcs, ref_tract, skymap_name, full_origin, bin_factor, out_box
                )
            else:
                out_array = mosaic
                out_origin = Point2I(
                    grid_origin.getX() + min_i * pitch_x, grid_origin.getY() + min_j * pitch_y
                )
                out_box = Box2I(out_origin, Extent2I(out_array.shape[1], out_array.shape[0]))
                sky_proj = ref_proj
            result = Struct(
                outputRGBMosaic=ColorImage(
                    out_array, bbox=Box.from_legacy(out_box), sky_projection=sky_proj
                )
            )
        except BaseException:
            for attr in ("imageHandle", "weightHandle"):
                if hasattr(self, attr):
                    getattr(self, attr).close()
            raise

        if hasattr(self, "weightHandle"):
            self.weightHandle.close()
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
