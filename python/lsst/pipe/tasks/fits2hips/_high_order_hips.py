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

__all__ = ("HighOrderFitsHipsTaskConnections", "HighOrderFitsHipsTaskConfig", "HighOrderFitsHipsTask")

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
from lsst.geom import Box2I, Point2I, Extent2I
from lsst.afw.math import Warper
from lsst.daf.butler import DeferredDatasetHandle
from lsst.afw.image import ImageF
from lsst.resources import ResourcePath

from collections.abc import Iterable
from lsst.sphgeom import RangeSet

import cv2

from ..rgb2hips._utils import _write_hips_image
from ..prettyPictureMaker import FeatheredMosaicCreator


class HighOrderFitsHipsTaskConnections(PipelineTaskConnections, dimensions=("healpix8", "band")):
    input_images = Input(
        doc="Fits images which are to be turned into hips tiles",
        name="",
        storageClass="ImageF",
        dimensions=("tract", "patch", "skymap", "band"),
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
        name="fits_picture_hips8",
        storageClass="NumpyArray",
        dimensions=("healpix8", "band"),
    )


class HighOrderFitsHipsTaskConfig(PipelineTaskConfig, pipelineConnections=HighOrderFitsHipsTaskConnections):
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

    def setDefaults(self):
        self.warp.warpingKernelName = "lanczos5"


class HighOrderFitsHipsTask(PipelineTask):
    """Pipeline task that generates high-order HealPix tiles from Fits images.

    Of Note; This task has special dispensation to write "out-of-tree" to a
    location not within the butler. DO NOT model other tasks on this one.

    This task takes in Fits images generated on a tract patch grid. It assembles
    them into a 4096 x 4096 image aligned with the wcs coordinates of hips
    order 8 pixels. This is then divided up into an 8x8 grid to produce 512x512
    images at hips order 11. The images is then resampled using lanczos order 4
    such that the image is half the size. The original image is then divided
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

    _DefaultName = "highOrderFitsHipsTask"
    ConfigClass = HighOrderFitsHipsTaskConfig

    config: ConfigClass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.warper = Warper.fromConfig(self.config.warp)

        # Set the base resource path that will be used for all outputs
        self.hips_base_path = ResourcePath(self.config.hips_base_uri, forceDirectory=True)

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

        self.log.info("Warping input exposures and populating hpx8 super tile.")
        # Need to loop over input arrays
        # Warp and combine input images into the HealPix tile
        for input_image, in_wcs, in_box in input_images:
            tmp_image = ImageF(in_box)
            in_image: NDArray = input_image

            # construct an Exposure object from one channel in the array
            tmp_image.array[:, :] = in_image

            # Warp the image to the target WCS
            warpped = self.warper.warpImage(target_wcs, tmp_image, in_wcs, maxBBox=exp_bbox)
            warpped_box_slices = warpped.getBBox().slices

            # Update the output array with valid (non-NaN) values
            are_warpped = np.isfinite(warpped.array)
            output_array_hpx[warpped_box_slices][are_warpped] = warpped.array[are_warpped]

        # Replace any remaining NaN values with zeros
        output_array_hpx[np.isnan(output_array_hpx)] = 0

        # Flip the y-axis to match HealPix indexing
        output_array_hpx = output_array_hpx[::-1, :, :]

        # Generate tiles for different HealPix orders using Lanczos resampling instead of binning.
        # This handles how intensities should change as the hips level changes.
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
                binned_array = cv2.resize(output_array_hpx, (size, size), interpolation=cv2.INTER_LANCZOS4)
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
                    sub_pixel = binned_array[i * 512 : i * 512 + 512, j * 512 : j * 512 + 512]
                    self.log.info(f"writing sub_pixel {pixel_id}")
                    _write_hips_image(sub_pixel, pixel_id, hips_level, self.hips_base_path, "fitsfloat")

        # Finally, bin the level 8 hpx to 256x256 (1/4 order 7) to save to the butler.
        # This makes smaller arrays to load, and saves the binning operation in the joint phase.
        zoomed = cv2.resize(output_array_hpx, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        return Struct(output_hpx=zoomed)

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
            new_array = np.zeros((new_box.getHeight(), new_box.getWidth()), dtype=np.float32)
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
                mosaic_maker.add_to_image(new_array, image.image.array, tmp_new_box, tmpBox, reverse=False)
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
            band.add(input_image_ref.dataId["band"])
            # All boxes in a given skymap will have the same inner dimensions
            # for x and y and will be the same for all patches
            imageWcs = skymap[tract][patch].getWcs()
            box = skymap[tract][patch].getOuterBBox()
            patch_grow = skymap[tract][patch].getCellInnerDimensions().getX()
            imageHandle = butlerQC.get(input_image_ref)
            container = inputs_by_tract.setdefault(tract, list())
            container.append((imageHandle, imageWcs, box))

        band_name = butlerQC.quantum.dataId["band"]

        self.hips_base_path = self.hips_base_path.join(f"band_{band_name}", forceDirectory=True)

        input_images = self._assemble_sub_region(inputs_by_tract, patch_grow)

        outputs = self.run(input_images, healpix_id)
        butlerQC.put(outputs, outputRefs)
