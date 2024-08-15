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
    "PrettyPictureTask",
    "PrettyPictureConnections",
    "PrettyPictureConfig",
    "PrettyMosaicTask",
    "PrettyMosaicConnections",
    "PrettyMosaicConfig",
)

from collections.abc import Iterable, Mapping
import numpy as np
from typing import TYPE_CHECKING, cast, Any
from lsst.skymap import BaseSkyMap

from lsst.daf.butler import Butler, DeferredDatasetHandle
from lsst.daf.butler import DatasetRef
from lsst.pex.config import Field, Config, ConfigDictField, ConfigField, ListField, ChoiceField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
    InMemoryDatasetHandle,
)
import cv2

from lsst.pipe.base.connectionTypes import Input, Output
from lsst.geom import Box2I, Point2I, Extent2I
from lsst.afw.image import Exposure, Mask

from ._plugins import plugins
from ._gp_standin.interpolate_over import InterpolateOverDefectGaussianProcess
from ._colorMapper import lsstRGB

import tempfile


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from lsst.pipe.base import QuantumContext, InputQuantizedConnection, OutputQuantizedConnection
    from lsst.skymap import TractInfo, PatchInfo


class PrettyPictureConnections(
    PipelineTaskConnections,
    dimensions={"tract", "patch", "skymap"},
    defaultTemplates={"coaddTypeName": "deep"},
):
    inputCoadds = Input(
        doc=(
            "Model of the static sky, used to find temporal artifacts. Typically a PSF-Matched, "
            "sigma-clipped coadd. Written if and only if assembleStaticSkyModel.doWrite=True"
        ),
        name="{coaddTypeName}CoaddPsfMatched",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
        multiple=True,
    )

    outputRGB = Output(
        doc="A RGB image created from the input data stored as a 3d array",
        name="rgbPicture_array",
        storageClass="NumpyArray",
        dimensions=("tract", "patch", "skymap"),
    )

    outputRGBMask = Output(
        doc="A Mask corresponding to the fused masks of the input channels",
        name="rgbPicture_mask",
        storageClass="Mask",
        dimensions=("tract", "patch", "skymap"),
    )


class ChannelRGBConfig(Config):
    """This describes the rgb values of a given input channel.

    For instance if this channel is red the values would be self.r = 1,
    self.g = 0, self.b = 0. If the channel was cyan the values would be
    self.r = 0, self.g = 1, self.b = 1.
    """

    r = Field[float](doc="The amount of red contained in this channel")
    g = Field[float](doc="The amount of green contained in this channel")
    b = Field[float](doc="The amount of blue contained in this channel")

    def validate(self):
        for f in (self.r, self.g, self.b):
            if f < 0 or f > 1:
                raise ValueError(f"Field {f} can not have a value less than 0 or greater than one")
        return super().validate()


class LumConfig(Config):
    """Configurations to control how luminance is mapped in the rgb code"""

    stretch = Field[float](doc="The stretch of the luminance in asinh", default=50)
    max = Field[float](doc="The maximum allowed luminance on a 0 to 100 scale", default=85)
    A = Field[float](doc="A scaling factor to apply post asinh stretching", default=0.9)
    b0 = Field[float](doc="A linear offset to apply post asinh stretching", default=0.05)
    minimum = Field[float](
        doc="The minimum intensity value after stretch, values lower will be set to zero", default=0
    )
    floor = Field[float](doc="A scaling factor to apply to the luminance before asinh scaling", default=0.0)
    Q = Field[float](doc="softening parameter", default=8)


class LocalContrastConfig(Config):
    """Configuration to control local contrast enhancement of the luminance
    channel."""

    doLocalContrast = Field[bool](
        doc="Apply local contrast enhancements to the luminance channel", default=True
    )
    highlights = Field[float](doc="Adjustment factor for the highlights", default=-1.5)
    shadows = Field[float](doc="Adjustment factor for the shadows", default=0.4)
    clarity = Field[float](doc="Amount of clarity to apply to contrast modification", default=0.2)
    sigma = Field[float](
        doc="The scale size of what is considered local in the contrast enhancement", default=20
    )
    maxLevel = Field[int](
        doc="The maximum number of scales the contrast should be enhanced over, if None then all",
        optional=True,
    )


class ScaleColorConfig(Config):
    """Controls how colors are scaled in the rgb generation process."""

    saturation = Field[float](
        doc=(
            "The overall saturation factor with the scaled luminance between zero and one. "
            "A value of one is not recommended as it makes bright pixels very saturated"
        ),
        default=0.5,
    )
    maxChroma = Field[float](
        doc=(
            "The maximum chromaticity in the CIELCh color space, large "
            "values will cause bright pixels to fall outside the RGB gamut."
        ),
        default=50.0,
    )


class RemapBoundsConfig(Config):
    """Remaps input images to a known range of values.

    Often input images are not mapped to any defined range of values
    (for instance if they are in count units). This controls how the units of
    and image will be mapped to a zero to one range by determining an upper
    bound.
    """

    quant = Field[float](
        doc=(
            "The maximum values of each of the three channels will be multiplied by this factor to "
            "determine the maximum flux of the image, values larger than this quantity will be clipped."
        ),
        default=0.8,
    )
    absMax = Field[float](
        doc="Instead of determining the maximum value from the image, use this fixed value instead",
        optional=True,
    )


class PrettyPictureConfig(PipelineTaskConfig, pipelineConnections=PrettyPictureConnections):
    channelConfig = ConfigDictField(
        doc="A dictionary that maps band names to their rgb channel configurations",
        keytype=str,
        itemtype=ChannelRGBConfig,
        default={},
    )
    imageRemappingConfig = ConfigField[RemapBoundsConfig](
        doc="Configuration controlling channel normalization process"
    )
    luminanceConfig = ConfigField[LumConfig](
        doc="Configuration for the luminance scaling when making an RGB image"
    )
    localContrastConfig = ConfigField[LocalContrastConfig](
        doc="Configuration controlling the local contrast correction in RGB image production"
    )
    colorConfig = ConfigField[ScaleColorConfig](
        doc="Configuration to control the color scaling process in RGB image production"
    )
    cieWhitePoint = ListField[float](
        doc="The white point of the input arrays in ciexz coordinates", maxLength=2, default=[0.28, 0.28]
    )
    arrayType = ChoiceField[str](
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
        self.channelConfig["i"] = ChannelRGBConfig(r=1, g=0, b=0)
        self.channelConfig["r"] = ChannelRGBConfig(r=0, g=1, b=0)
        self.channelConfig["g"] = ChannelRGBConfig(r=0, g=0, b=1)
        return super().setDefaults()


class PrettyPictureTask(PipelineTask):
    _DefaultName = "prettyPictureTask"
    ConfigClass = PrettyPictureConfig

    config: ConfigClass

    def run(self, images: Mapping[str, Exposure]) -> Struct:
        channels = {}
        shape = (0, 0)
        jointMask: None | NDArray = None
        maskDict: Mapping[str, int] = {}
        for channel, imageExposure in images.items():
            # This is a hack until gp is in the stack natively, or until a
            # plugin is made with a GP interpolation based on numpy arrays.
            GP = InterpolateOverDefectGaussianProcess(
                imageExposure.maskedImage,
                defects=["SAT", "NO_DATA"],
                fwhm=15,
                block_size=40,
                solver="treegp",
                method="spanset",
                bin_spacing=15,
                use_binning=True,
            )
            # This modifies the MaskedImage in place
            GP.interpolate_over_defects()
            maskedImage = GP.maskedImage
            imageArray = maskedImage.image.array
            # run all the plugins designed for array based interaction
            for plug in plugins.channel():
                imageArray = plug(
                    imageArray, imageExposure.mask.array, imageExposure.mask.getMaskPlaneDict()
                ).astype(np.float32)
            channels[channel] = imageArray
            # This will get done each loop, but they are trivial lookups so it
            # does not matter
            shape = imageArray.shape
            maskDict = imageExposure.mask.getMaskPlaneDict()
            if jointMask is None:
                jointMask = np.zeros(shape, dtype=imageExposure.mask.dtype)
                jointMask |= imageExposure.mask.array

        # mix the images to rgb
        imageRArray = np.zeros(shape, dtype=np.float32)
        imageGArray = np.zeros(shape, dtype=np.float32)
        imageBArray = np.zeros(shape, dtype=np.float32)

        for band, image in channels.items():
            mix = self.config.channelConfig[band]
            # channelSum = mix.r + mix.g + mix.b
            channelSum = 1
            if mix.r:
                imageRArray += mix.r / channelSum * image
            if mix.g:
                imageGArray += mix.g / channelSum * image
            if mix.b:
                imageBArray += mix.b / channelSum * image

        exposure = next(iter(images.values()))
        box: Box2I = exposure.getBBox()
        boxCenter = box.getCenter()
        psf = exposure.psf.computeImage(boxCenter).array
        # Ignore type because Exposures do in fact have a bbox, but it is c++
        # and not typed.
        colorImage = lsstRGB(
            imageRArray,
            imageGArray,
            imageBArray,
            scaleLumKWargs=self.config.luminanceConfig.toDict(),
            remapBoundsKwargs=self.config.imageRemappingConfig.toDict(),
            scaleColorKWargs=self.config.colorConfig.toDict(),
            **(self.config.localContrastConfig.toDict()),
            cieWhitePoint=tuple(self.config.cieWhitePoint),  # type: ignore
            psf=psf,
        )

        # Find the dataset type and thus the maximum values as well
        maxVal: int | float
        match self.config.arrayType:
            case "uint8":
                dtype = np.uint8
                maxVal = 255
            case "uint16":
                dtype = np.uint16
                maxVal = 65535
            case "half":
                dtype = np.half
                maxVal = 1.0
            case "float":
                dtype = np.float32
                maxVal = 1.0
            case _:
                assert True, "This code path should be unreachable"

        # lsstRGB returns an image in 0-1 scale it to the maximum value
        colorImage *= maxVal  # type: ignore

        # assert for typing reasons
        assert jointMask is not None
        # Run any image level correction plugins
        for plug in plugins.partial():
            colorImage = plug(colorImage, jointMask, maskDict)

        # pack the joint mask back into a mask object
        lsstMask = Mask(
            width=jointMask.shape[1], height=jointMask.shape[0], planeDefs=maskDict
        )  # type: ignore
        lsstMask.array = jointMask  # type: ignore
        return Struct(outputRGB=colorImage.astype(dtype), outputRGBMask=lsstMask)  # type: ignore

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        imageRefs: list[DatasetRef] = inputRefs.inputCoadds
        sortedImages = self.makeInputsFromRefs(imageRefs, butlerQC)
        outputs = self.run(sortedImages)
        butlerQC.put(outputs, outputRefs)

    def makeInputsFromRefs(
        self, refs: Iterable[DatasetRef], butler: Butler | QuantumContext
    ) -> dict[str, Exposure]:
        sortedImages: dict[str, Exposure] = {}
        for ref in refs:
            key: str = cast(str, ref.dataId["band"])
            image = butler.get(ref)
            sortedImages[key] = image
        return sortedImages

    def makeInputsFromArrays(self, **kwargs) -> dict[int, DeferredDatasetHandle]:
        # ignore type because there are not proper stubs for afw
        temp = {}
        for key, array in kwargs.items():
            temp[key] = Exposure(
                Box2I(Point2I(0, 0), Extent2I(*array.shape)), dtype=array.dtype
            )  # type: ignore
            temp[key].image.array[:] = array

        return self.makeInputsFromExposures(**temp)

    def makeInputsFromExposures(self, **kwargs) -> dict[int, DeferredDatasetHandle]:
        sortedImages = {}
        for key, value in kwargs.items():
            sortedImages[key] = value
        return sortedImages


class PrettyMosaicConnections(PipelineTaskConnections, dimensions=("tract", "skymap")):
    inputRGB = Input(
        doc="Individual RGB images that are to go into the mosaic",
        name="rgbPicture_array",
        storageClass="NumpyArray",
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

    inputRGBMask = Input(
        doc="Individual RGB images that are to go into the mosaic",
        name="rgbPicture_mask",
        storageClass="Mask",
        dimensions=("tract", "patch", "skymap"),
        multiple=True,
        deferLoad=True,
    )

    outputRGBMosaic = Output(
        doc="A RGB mosaic created from the input data stored as a 3d array",
        name="rgbMosaic_array",
        storageClass="NumpyArray",
        dimensions=("tract", "skymap"),
    )


class PrettyMosaicConfig(PipelineTaskConfig, pipelineConnections=PrettyMosaicConnections):
    binFactor = Field[int](doc="The factor to bin by when producing the mosaic")


class PrettyMosaicTask(PipelineTask):
    _DefaultName = "prettyMosaicTask"
    ConfigClass = PrettyMosaicConfig

    config: ConfigClass

    def run(
        self,
        inputRGB: Iterable[DeferredDatasetHandle],
        skyMap: BaseSkyMap,
        inputRGBMask: Iterable[DeferredDatasetHandle],
    ) -> Struct:
        # create the bounding region
        newBox = Box2I()
        # store the bounds as they are retrieved from the skymap
        boxes = []
        for handle in inputRGB:
            dataId = handle.dataId
            tractInfo: TractInfo = skyMap[dataId["tract"]]
            patchInfo: PatchInfo = tractInfo[dataId["patch"]]
            bbox = patchInfo.getOuterBBox()
            boxes.append(bbox)
            newBox.include(bbox)

        # fixup the boxes to be smaller if needed, and put the origin at zero,
        # this must be done after constructing the complete outer box
        modifiedBoxes = []
        origin = newBox.getBegin()
        for iterBox in boxes:
            localOrigin = iterBox.getBegin() - origin
            localOrigin = Point2I(
                x=int(np.floor(localOrigin.x / self.config.binFactor)),
                y=int(np.floor(localOrigin.y / self.config.binFactor)),
            )
            localExtent = Extent2I(
                x=int(np.floor(iterBox.getWidth() / self.config.binFactor)),
                y=int(np.floor(iterBox.getHeight() / self.config.binFactor)),
            )
            tmpBox = Box2I(localOrigin, localExtent)
            modifiedBoxes.append(tmpBox)
        boxes = modifiedBoxes

        # scale the container box
        newBoxOrigin = Point2I(0, 0)
        newBoxExtent = Extent2I(
            x=int(np.floor(newBox.getWidth() / self.config.binFactor)),
            y=int(np.floor(newBox.getHeight() / self.config.binFactor)),
        )
        newBox = Box2I(newBoxOrigin, newBoxExtent)

        # Allocate storage for the mosaic
        self.imageHandle = tempfile.NamedTemporaryFile()
        self.maskHandle = tempfile.NamedTemporaryFile()
        consolidatedImage = None
        consolidatedMask = None

        # Actually assemble the mosaic
        maskDict = {}
        for box, handle, handleMask in zip(boxes, inputRGB, inputRGBMask):
            rgb = handle.get()
            rgbMask = handleMask.get()
            maskDict = rgbMask.getMaskPlaneDict()
            # allocate the memory for the mosaic
            if consolidatedImage is None:
                consolidatedImage = np.memmap(
                    self.imageHandle.name,
                    mode="w+",
                    shape=(newBox.getHeight(), newBox.getWidth(), 3),
                    dtype=rgb.dtype,
                )
            if consolidatedMask is None:
                consolidatedMask = np.memmap(
                    self.maskHandle.name,
                    mode="w+",
                    shape=(newBox.getHeight(), newBox.getWidth()),
                    dtype=rgbMask.array.dtype,
                )

            if self.config.binFactor > 1:
                # opencv wants things in x, y dimensions
                shape = tuple(box.getDimensions())[::-1]
                rgb = cv2.resize(
                    rgb,
                    dst=consolidatedImage[*box.slices],
                    dsize=shape,
                    fx=shape[0] / self.config.binFactor,
                    fy=shape[1] / self.config.binFactor,
                )
                rgbMask = cv2.resize(
                    rgbMask.array.astype(np.float32),
                    dst=consolidatedMask[*box.slices],
                    dsize=shape,
                    fx=shape[0] / self.config.binFactor,
                    fy=shape[1] / self.config.binFactor,
                )
            else:
                consolidatedImage[*box.slices] = rgb
                consolidatedMask[*box.slices] = rgbMask.array

        for plugin in plugins.full():
            if consolidatedImage is not None and consolidatedMask is not None:
                consolidatedImage = plugin(consolidatedImage, consolidatedMask, maskDict)
        # If consolidated image still None, that means there was no work to do.
        # Return an empty image instead of letting this task fail.
        if consolidatedImage is None:
            consolidatedImage = np.zeros((0, 0, 0), dtype=np.uint8)

        return Struct(outputRGBMosaic=consolidatedImage)

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
        if hasattr(self, "maskHandle"):
            self.maskHandle.close()

    def makeInputsFromArrays(
        self, inputs: Iterable[tuple[Mapping[str, Any], NDArray]]
    ) -> Iterable[DeferredDatasetHandle]:
        structuredInputs = []
        for dataId, array in inputs:
            structuredInputs.append(InMemoryDatasetHandle(inMemoryDataset=array, **dataId))

        return structuredInputs


#  ## old needs translated to two tasks
old = '''
    doMemeMap = Field[bool](doc="Use a memory map when constructing a mosaic", default=True)

    def run(self, inputImages: dict[int, _InputImageContainer]) -> Struct:
        # get outer bounding region
        newBox = Box2I()
        for container in inputImages.values():
            newBox.include(container.getBbox())

        # create the outer container
        self.imageHandle = tempfile.NamedTemporaryFile()
        self.maskHandle = tempfile.NamedTemporaryFile()
        consolidatedImage = np.memmap(self.imageHandle.name, mode='w+', shape=(newBox.getHeight(), newBox.getWidth(), 3), dtype=np.uint8)
        consolidatedMask = np.memmap(self.maskHandle.name, mode='w+', shape=(newBox.getHeight(), newBox.getWidth()), dtype=np.float32)
            #consolidatedImage = np.zeros((newBox.getHeight(), newBox.getWidth(), 3), dtype=np.uint8)
            #consolidatedMask = np.zeros((newBox.getHeight(), newBox.getWidth()))

        # assemble sub regions
        for region, container in inputImages.items():
            box, rgbImage, mask = container.realizeRGBImage(
                minimum=self.config.minimum,
                stretch=self.config.stretch,
                impact=self.config.impact,
                maxLum=self.config.lumMax
            )
            box = box.shiftedBy(-1*ExtentI(newBox.getBegin()))
            consolidatedImage[*box.slices, :] = rgbImage.astype(np.uint8)
            consolidatedMask[*box.slices] = mask

            for plugin in plugins.full():
                consolidatedImage = plugin(consolidatedImage, consolidatedMask)
        return Struct(image=consolidatedImage)

    def _makeIdentifierNumber(self, ref: DatasetRef) -> int:
        """This function is responsible for making unique identifiers to track
        sub regions in an image.

        For coadds this is simply patch, single frame likely would use packed
        visitIds.
        """
        return cast(int, ref.dataId["patch"])

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        # TODO: this needs to be made generic for image type
        imageRefs: list[DatasetRef] = inputRefs.inputCoadds
        sortedImages = self.makeInputsFromRefs(imageRefs, butlerQC)
        self.run(sortedImages)

    def makeInputsFromRefs(
        self, refs: Iterable[DatasetRef], butler: Butler | QuantumContext
    ) -> dict[int, _InputImageContainer]:
        sortedImages: dict[int, _InputImageContainer] = {}
        for ref in refs:
            key: int = self._makeIdentifierNumber(ref)
            container = sortedImages.setdefault(
                key, _InputImageContainer(key, list(), self.config.channelConfig)
            )
            proxyImage = (
                butler.get(ref)
                if isinstance(butler, QuantumContext)
                else DeferredDatasetHandle(butler, ref, None)
            )
            container.images.append((ref.dataId, proxyImage))
        return sortedImages

    def makeInputsFromArrays(
        self, *args, channelRGBs=((1, 0, 0), (0, 1, 0), (0, 0, 1))
    ) -> dict[int, _InputImageContainer]:
        if len(args) != len(channelRGBs):
            raise ValueError("There must be the same number of input arrays as channels")
        if args[0] is
        # ignore type because there are not proper stubs for afw
        rExp = Exposure(Box2I(PointI(0, 0), ExtentI(*rArray.shape)), dtype=rArray.dtype)  # type: ignore
        rExp.image.array[:] = rArray
        gExp = Exposure(Box2I(PointI(0, 0), ExtentI(*gArray.shape)), dtype=gArray.dtype)  # type: ignore
        gExp.image.array[:] = gArray
        bExp = Exposure(Box2I(PointI(0, 0), ExtentI(*bArray.shape)), dtype=bArray.dtype)  # type: ignore
        bExp.image.array[:] = bArray
        return self.makeInputsFromExposures([rExp], [gExp], [bExp])

    def makeInputsFromExposures(
        self, rExposures: list[Exposure], gExposures: list[Exposure], bExposures: list[Exposure]
    ) -> dict[int, _InputImageContainer]:
        if len(rExposures) != len(gExposures) or len(gExposures) != len(bExposures):
            raise AttributeError("Lengths of all inputs must be equal")
        sortedImages: dict[int, _InputImageContainer] = {}
        for key in range(len(rExposures)):
            container = sortedImages.setdefault(key, _InputImageContainer(key))
            rHandle = InMemoryDatasetHandle(rExposures[key])
            container.r = (cast(DataCoordinate, rHandle.dataId), rHandle)

            gHandle = InMemoryDatasetHandle(gExposures[key])
            container.g = (cast(DataCoordinate, gHandle.dataId), gHandle)

            bHandle = InMemoryDatasetHandle(bExposures[key])
            container.b = (cast(DataCoordinate, bHandle.dataId), bHandle)
        return sortedImages
'''
