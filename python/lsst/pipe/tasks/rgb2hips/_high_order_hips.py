from __future__ import annotations

__all__ = ("HighOrderHipsTaskConnections", "HighOrderHipsTaskConfig", "HighOrderHipsTask")

import numpy as np
from enum import Enum
from numpy.typing import NDArray

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
from numpy.lib.stride_tricks import as_strided
from lsst.sphgeom import RangeSet

from skimage.transform import resize
from PIL import Image

from ._hipsWcsMaker import makeHpxWcs
from ._utils import _write_hips_image

Image.MAX_IMAGE_PIXELS = None


class ColorChannel(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


class HighOrderHipsTaskConnections(PipelineTaskConnections, dimensions=("healpix8",)):
    input_images = Input(
        doc="Color images which are to be turned into hips tiles",
        name="rgb_picture_array",
        storageClass="NumpyArray",
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
    hips_order = 8
    warp = ConfigField[Warper.ConfigClass](
        doc="Warper configuration",
    )
    patchGrow = Field[int](
        doc="Dilate the patches by this much when determining the positions, normally size of overlap",
        default=150,
    )
    hips_base_uri = Field[str](
        doc="URI to HiPS base for output.",
        optional=False,
    )
    color_ordering = Field[str](doc="The bands used to construct the input images", optional=False)
    file_extension = ChoiceField[str](
        doc="Extension for the presisted image, must be png or webp",
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

    def setDefaults(self):
        self.warp.warpingKernelName = "lanczos5"


class HighOrderHipsTask(PipelineTask):
    _DefaultName = "highOrderHipsTask"
    ConfigClass = HighOrderHipsTaskConfig

    config: ConfigClass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.warper = Warper.fromConfig(self.config.warp)
        self.hips_base_path = ResourcePath(self.config.hips_base_uri, forceDirectory=True)
        self.hips_base_path = self.hips_base_path.join(
            f"color_{self.config.color_ordering}", forceDirectory=True
        )

    def run(self, input_images: Iterable[tuple[DeferredDatasetHandle, SkyWcs, Box2I]], healpix_id) -> Struct:
        # Make the WCS for the transform, intentionally over-sampled to shift order 12.
        # This creates as 4096 x 4096 image that can be broken appart to form the higher
        # orders, binning each as needed
        target_wcs = makeHpxWcs(8, healpix_id, 12)

        # construct a bounding box that holds the warping results for each channel
        exp_bbox = Box2I(corner=Point2I(0, 0), dimensions=Extent2I(2**12, 2**12))

        output_array_hpx = np.zeros((4096, 4096, 3), dtype=np.float32)
        output_array_hpx[:, :, :] = np.nan

        # Need to loop over input arrays then channel
        for input_image, in_wcs, in_box in input_images:
            tmp_image = ImageF(in_box)
            # Flip the Y axis, because things are reversed
            in_image: NDArray = input_image.get()[::-1, :, :]
            # Need to cast images if they are saved in various formats
            match in_image.dtype:
                case np.uint8:
                    in_image = in_image.astype(np.float32) / 255.0
                case np.uint16:
                    in_image = in_image.astype(np.float32) / 65535
                case np.float16:
                    in_image = in_image.astype(np.float32)
            for channel in ColorChannel:
                # existing data
                existing = output_array_hpx[..., channel.value]

                # construct an Exposure object from one channel in the array
                channel_array = in_image[..., channel.value]
                tmp_image.array[:, :] = channel_array
                warpped = self.warper.warpImage(target_wcs, tmp_image, in_wcs, maxBBox=exp_bbox)

                # determine the mask for nan values, meaning they have not been set yet
                existing_nan_mask = np.isnan(existing)
                existing_filled_mask = ~existing_nan_mask

                # determine what values in the array are set
                are_warpped = np.isfinite(warpped.image.array)

                # Values that are in the new warp, but not in existing can be assigned
                new_values_mask = existing_nan_mask * are_warpped
                existing[new_values_mask] = warpped.image.array[new_values_mask]

                # Values that are set in existing and new warp should be averaged
                both_set_mask = existing_filled_mask * are_warpped
                existing[both_set_mask] = (existing[both_set_mask] + warpped.image.array[both_set_mask]) / 2.0
            # The healpix is populated with all data available
        # Fill in nans with zeros
        output_array_hpx[np.isnan(output_array_hpx)] = 0

        # now it is time to start making the tiles for higher orders
        #
        # Construct a basic RangeSet for the pixel for this quanta
        quanta_range_set = RangeSet([healpix_id])

        for zoom, hpx_level, factor in zip((0, 2, 4, 8), (11, 10, 9, 8), (3, 2, 1, 0)):
            if zoom:
                size = 4096 // zoom
                binned_array = resize(output_array_hpx, (size, size))
            else:
                binned_array = output_array_hpx
            # always create blocks of 512x512 as that is native size
            view = self._make_block(binned_array, (512, 512))
            hpx_start, hpx_stop = quanta_range_set.scaled(4**factor).ranges()[0]
            hpx_id_array = np.arange(hpx_start, hpx_stop).reshape(view.shape[:2])
            for i in range(view.shape[0]):
                for j in range(view.shape[1]):
                    pixel_id = hpx_id_array[i, j]
                    sub_pixel = view[i, j]
                    _write_hips_image(
                        sub_pixel,
                        pixel_id,
                        hpx_level,
                        self.hips_base_path,
                        self.config.file_extension,
                        self.config.array_type,
                    )

        # Finally, zoom the level 8 hpx to 256x256 to save to the buter.
        # This makes smaller arrays to load, and save the binning
        # operation in the joint phase.
        zoomed = resize(output_array_hpx, (256, 256))

        return Struct(output_hpx=zoomed)

    @staticmethod
    def _make_block(array, block):
        shape = (array.shape[0] // block[0], array.shape[1] // block[1]) + block
        strides = (block[0] * array.strides[0], block[1] * array.strides[1]) + array.strides
        return as_strided(array, shape=shape, strides=strides)

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
        input_images = []
        for inputImageRef in inputRefs.inputImages:
            tract = inputImageRef["tract"]
            patch = inputImageRef["patch"]
            imageWcs = skymap[tract][patch].getWcs()
            box = skymap[tract][patch].getInnerBBox()
            box.dilatedBy(self.config.patchGrow)
            imageHandle = butlerQC.get(inputImageRef)
            input_images.append((imageHandle, imageWcs, box))

        outputs = self.run(input_images, healpix_id)
        butlerQC.put(outputs, outputRefs)
