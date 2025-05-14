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
    "ChannelRGBConfig",
    "PrettyPictureTask",
    "PrettyPictureConnections",
    "PrettyPictureConfig",
    "PrettyMosaicTask",
    "PrettyMosaicConnections",
    "PrettyMosaicConfig",
    "PrettyPictureBackgroundFixerConfig",
    "PrettyPictureBackgroundFixerTask",
    "PrettyPictureStarFixerConfig",
    "PrettyPictureStarFixerTask",
)

from collections.abc import Iterable, Mapping
from lsst.afw.image import ExposureF
import numpy as np
from typing import TYPE_CHECKING, cast, Any
from lsst.skymap import BaseSkyMap

from scipy.stats import norm
from scipy.ndimage import binary_dilation, label
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator
from skimage.restoration import inpaint_biharmonic

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
from ._colorMapper import lsstRGB, lumScale, latLum

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
        name="rgb_picture_array",
        storageClass="NumpyArray",
        dimensions=("tract", "patch", "skymap"),
    )

    outputRGBMask = Output(
        doc="A Mask corresponding to the fused masks of the input channels",
        name="rgb_picture_mask",
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


class LumConfigV2(Config):
    highlight = Field[float](
        doc="The value of highlights in scaling factor applied to post asinh streaching", default=1.0
    )
    shadow = Field[float](
        doc="The value of shadows in scaling factor applied to post asinh streaching", default=0.0
    )
    midtone = Field[float](
        doc="The value of midtone in scaling factor applied to post asinh streaching", default=0.5
    )
    equalizer_levels = ListField[float](
        doc=(
            "A list of factors to modify the constrast in a scale dependent way. "
            "One coefficient for each spatial scale, starting from the largest. "
            "Values large than 1 increase contrast, while less than 1 decreases. "
            "This adjustment is multaplicative."
        ),
        optional=True,
    )
    tone_adjustment = ListField[float](
        doc=(
            "A list of length 10 that adjusts the brightness of the image ranging "
            "from dark regions to light. These 10 values represent control points along "
            "the lumanance interval 0-1, but the actual adjustments made are continuous "
            "and are calculated from these control points."
        ),
        optional=True,
    )
    tone_width = Field[float](
        doc=(
            "This parameters controls how each tone control point affect the adjustment "
            "of the values in between. Increase the value to have a more continuous "
            "change between control points, decrease to make the control shaprer. Value "
            "must be greater than zero."
        ),
        default=0.07,
    )
    max = Field[float](doc="The maximum allowed luminance on a 0 to 1 scale", default=1)
    stretch = Field[float]("Streach of the lum", default=400)
    floor = Field[float](doc="A scaling factor to apply to the luminance before asinh scaling", default=0.0)


class LumConfig(Config):
    """Configurations to control how luminance is mapped in the rgb code"""

    stretch = Field[float](doc="The stretch of the luminance in asinh", default=400)
    max = Field[float](doc="The maximum allowed luminance on a 0 to 1 scale", default=1)
    floor = Field[float](doc="A scaling factor to apply to the luminance before asinh scaling", default=0.0)
    Q = Field[float](doc="softening parameter", default=0.7)
    highlight = Field[float](
        doc="The value of highlights in scaling factor applied to post asinh streaching", default=1.0
    )
    shadow = Field[float](
        doc="The value of shadows in scaling factor applied to post asinh streaching", default=0.0
    )
    midtone = Field[float](
        doc="The value of midtone in scaling factor applied to post asinh streaching", default=0.5
    )
    equalizer_levels = ListField[float](
        doc=(
            "A list of factors to modify the constrast in a scale dependent way. "
            "One coefficient for each spatial scale, starting from the largest. "
            "Values large than 1 increase contrast, while less than 1 decreases"
        ),
        optional=True,
    )


class LocalContrastConfig(Config):
    """Configuration to control local contrast enhancement of the luminance
    channel."""

    doLocalContrast = Field[bool](
        doc="Apply local contrast enhancements to the luminance channel", default=True
    )
    highlights = Field[float](doc="Adjustment factor for the highlights", default=-0.9)
    shadows = Field[float](doc="Adjustment factor for the shadows", default=0.5)
    clarity = Field[float](doc="Amount of clarity to apply to contrast modification", default=0.1)
    sigma = Field[float](
        doc="The scale size of what is considered local in the contrast enhancement", default=30
    )
    maxLevel = Field[int](
        doc="The maximum number of scales the contrast should be enhanced over, if None then all",
        default=4,
        optional=True,
    )


class ScaleColorConfig(Config):
    """Controls color scaling in the RGB generation process."""

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
    equalizer_levels = ListField[float](
        doc=(
            "A list of factors to modify the color constrast in a scale dependent way. "
            "One coefficient for each spatial scale, starting from the largest. "
            "Values large than 1 increase contrast, while less than 1 decreases"
        ),
        optional=True,
    )


class RemapBoundsConfig(Config):
    """Remaps input images to a known range of values.

    Often input images are not mapped to any defined range of values
    (for instance if they are in count units). This controls how the units of
    and image are mapped to a zero to one range by determining an upper
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
        default=220,
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
    luminanceConfig = ConfigField[LumConfigV2](
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
    doPSFDeconcovlve = Field[bool](
        doc="Use the PSF in a richardson lucy deconvolution on the luminance channel.", default=True
    )
    exposureBrackets = ListField[float](
        doc=(
            "Exposure scaling factors used in creating multiple exposures with different scalings which will "
            "then be fused into a final image"
        ),
        optional=True,
        default=[1.25, 1, 0.75],
    )
    doRemapGamut = Field[bool](
        doc="Apply a color correction to unrepresentable colors, if false they will clip", default=True
    )
    gamutMethod = ChoiceField[str](
        doc="If doRemapGamut is True this determines the method",
        default="inpaint",
        allowed={
            "mapping": "Use a mapping function",
            "inpaint": "Use surrounding pixels to determine likely value",
        },
    )

    def setDefaults(self):
        self.channelConfig["i"] = ChannelRGBConfig(r=1, g=0, b=0)
        self.channelConfig["r"] = ChannelRGBConfig(r=0, g=1, b=0)
        self.channelConfig["g"] = ChannelRGBConfig(r=0, g=0, b=1)
        return super().setDefaults()


class PrettyPictureTask(PipelineTask):
    """Turns inputs into an RGB image."""

    _DefaultName = "prettyPictureTask"
    ConfigClass = PrettyPictureConfig

    config: ConfigClass

    def run(self, images: Mapping[str, Exposure]) -> Struct:
        """Turns the input arguments in arguments into an RGB array.

        Parameters
        ----------
        images : `Mapping` of `str` to `Exposure`
            A mapping of input images and the band they correspond to.

        Returns
        -------
        result : `Struct`
            A struct with the corresponding RGB image, and mask used in
            RGB image construction. The struct will have the attributes
            outputRGBImage and outputRGBMask. Each of the outputs will
            be a `NDarray` object.

        Notes
        -----
        Construction of input images are made easier by use of the
        makeInputsFrom* methods.
        """
        channels = {}
        shape = (0, 0)
        jointMask: None | NDArray = None
        maskDict: Mapping[str, int] = {}
        doJointMaskInit = False
        if jointMask is None:
            doJointMask = True
            doJointMaskInit = True
        for channel, imageExposure in images.items():
            imageArray = imageExposure.image.array
            # run all the plugins designed for array based interaction
            for plug in plugins.channel():
                imageArray = plug(
                    imageArray, imageExposure.mask.array, imageExposure.mask.getMaskPlaneDict(), self.config
                ).astype(np.float32)
            channels[channel] = imageArray
            # These operations are trivial look-ups and don't matter if they
            # happen in each loop.
            shape = imageArray.shape
            maskDict = imageExposure.mask.getMaskPlaneDict()
            if doJointMaskInit:
                jointMask = np.zeros(shape, dtype=imageExposure.mask.dtype)
                doJointMaskInit = False
            if doJointMask:
                jointMask |= imageExposure.mask.array

        # mix the images to RGB
        imageRArray = np.zeros(shape, dtype=np.float32)
        imageGArray = np.zeros(shape, dtype=np.float32)
        imageBArray = np.zeros(shape, dtype=np.float32)

        for band, image in channels.items():
            mix = self.config.channelConfig[band]
            if mix.r:
                imageRArray += mix.r * image
            if mix.g:
                imageGArray += mix.g * image
            if mix.b:
                imageBArray += mix.b * image

        exposure = next(iter(images.values()))
        box: Box2I = exposure.getBBox()
        boxCenter = box.getCenter()
        try:
            psf = exposure.psf.computeImage(boxCenter).array
        except Exception:
            psf = None

        # assert for typing reasons
        assert jointMask is not None
        # Run any image level correction plugins
        colorImage = np.zeros((*imageRArray.shape, 3))
        colorImage[:, :, 0] = imageRArray
        colorImage[:, :, 1] = imageGArray
        colorImage[:, :, 2] = imageBArray
        for plug in plugins.partial():
            colorImage = plug(colorImage, jointMask, maskDict, self.config)

        # match self.config.luminanceConfig:
        #     case LumConfig():
        #         lum_func = lumScale
        #     case LumConfigV2():
        #         lum_func = lumScale
        lum_func = lumScale

        # Ignore type because Exposures do in fact have a bbox, but it's c++
        # and not typed.
        colorImage = lsstRGB(
            colorImage[:, :, 0],
            colorImage[:, :, 1],
            colorImage[:, :, 2],
            scaleLum=lum_func,
            scaleLumKWargs=self.config.luminanceConfig.toDict(),
            remapBoundsKwargs=self.config.imageRemappingConfig.toDict(),
            scaleColorKWargs=self.config.colorConfig.toDict(),
            **(self.config.localContrastConfig.toDict()),
            cieWhitePoint=tuple(self.config.cieWhitePoint),  # type: ignore
            psf=psf if self.config.doPSFDeconcovlve else None,
            brackets=list(self.config.exposureBrackets) if self.config.exposureBrackets else None,
            doRemapGamut=self.config.doRemapGamut,
            gamutMethod=self.config.gamutMethod,
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

        # pack the joint mask back into a mask object
        lsstMask = Mask(width=jointMask.shape[1], height=jointMask.shape[0], planeDefs=maskDict)
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
        r"""Make valid inputs for the run method from butler references.

        Parameters
        ----------
        refs : `Iterable` of `DatasetRef`
            Some `Iterable` container of `Butler` `DatasetRef`\ s
        butler : `Butler` or `QuantumContext`
            This is the object that fetches the input data.

        Returns
        -------
        sortedImages : `dict` of `str` to `Exposure`
            A dictionary of `Exposure`\ s that keyed by the band they
            correspond to.
        """
        sortedImages: dict[str, Exposure] = {}
        for ref in refs:
            key: str = cast(str, ref.dataId["band"])
            image = butler.get(ref)
            sortedImages[key] = image
        return sortedImages

    def makeInputsFromArrays(self, **kwargs) -> dict[int, DeferredDatasetHandle]:
        r"""Make valid inputs for the run method from numpy arrays.

        Parameters
        ----------
        kwargs : `NDArray`
            This is standard python kwargs where the left side of the equals
            is the data band, and the right side is the corresponding `NDArray`
            array.

        Returns
        -------
        sortedImages : `dict` of `str` to `Exposure`
            A dictionary of `Exposure`\ s that keyed by the band they
            correspond to.
        """
        # ignore type because there aren't proper stubs for afw
        temp = {}
        for key, array in kwargs.items():
            temp[key] = Exposure(Box2I(Point2I(0, 0), Extent2I(*array.shape)), dtype=array.dtype)
            temp[key].image.array[:] = array

        return self.makeInputsFromExposures(**temp)

    def makeInputsFromExposures(self, **kwargs) -> dict[int, DeferredDatasetHandle]:
        r"""Make valid inputs for the run method from `Exposure` objects.

        Parameters
        ----------
        kwargs : `Exposure`
            This is standard python kwargs where the left side of the equals
            is the data band, and the right side is the corresponding
            `Exposure`.

        Returns
        -------
        sortedImages : `dict` of `str` to `Exposure`
            A dictionary of `Exposure`\ s that keyed by the band they
            correspond to.
        """
        sortedImages = {}
        for key, value in kwargs.items():
            sortedImages[key] = value
        return sortedImages


class PrettyPictureBackgroundFixerConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap", "band"),
    defaultTemplates={"coaddTypeName": "deep"},
):
    inputCoadd = Input(
        doc=("Input coadd for which the background is to be removed"),
        name="{coaddTypeName}CoaddPsfMatched",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
    )
    outputCoadd = Output(
        doc="The coadd with the background fixed and subtracted",
        name="pretty_picture_coadd_bg_subtracted",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
    )


class PrettyPictureBackgroundFixerConfig(
    PipelineTaskConfig, pipelineConnections=PrettyPictureBackgroundFixerConnections
):
    brightnessThresh = Field[float](doc="Ignore any pixels above this value", default=50)


class PrettyPictureBackgroundFixerTask(PipelineTask):
    """Empirically flatten an images background.

    Many astrophysical images have backgrounds with imperfections in them.
    This Task attempts to determine control points which are considered
    background values, and fits a radial basis function model to those
    points. This model is then subtracted off the image.

    """

    _DefaultName = "prettyPictureBackgroundFixerTask"
    ConfigClass = PrettyPictureBackgroundFixerConfig

    config: ConfigClass

    def _neg_log_likelihood(self, params, x):
        """Calculate the negative log-likelihood for a Gaussian distribution.

        This function computes the negative log-likelihood of a set of data `x`
        given a Gaussian distribution with parameters `mu` and `sigma`.  It's
        designed to be used as the objective function for a minimization routine
        to find the best-fit Gaussian parameters.

        Parameters
        ----------
        params : `tuple`
            A tuple containing the mean (`mu`) and standard deviation (`sigma`)
            of the Gaussian distribution.
        x : `NDArray`
            The data samples for which to calculate the log-likelihood.

        Returns
        -------
        float
            The negative log-likelihood of the data given the Gaussian parameters.
            Returns infinity if sigma is non-positive or if the mean is less than
            the maximum value in x (to enforce the constraint that the Gaussian
            only models the lower tail of the distribution).
        """
        mu, sigma = params
        if sigma <= 0:
            return np.inf
        M = np.max(x)
        if mu < M - 1e-8:  # Allow for floating point precision issues
            return np.inf
        z = (x - mu) / sigma
        term = np.log(2) - np.log(sigma) + norm.logpdf(z)
        loglikelihood = np.sum(term)
        return -loglikelihood

    def _tile_slices(self, arr, R, C):
        """Generate slices for tiling an array.

        This function divides an array into a grid of tiles and returns a list of
        slice objects representing each tile.  It handles cases where the array
        dimensions are not evenly divisible by the number of tiles in each
        dimension, distributing the remainder among the tiles.

        Parameters
        ----------
        arr : `NDArray`
           The input array to be tiled. Used only to determine the array's shape.
        R : `int`
           The number of tiles in the row dimension.
        C : `int`
           The number of tiles in the column dimension.

        Returns
        -------
        slices : `list` of `tuple`
           A list of tuples, where each tuple contains two `slice` objects
           representing the row and column slices for a single tile.
        """
        M = arr.shape[0]
        N = arr.shape[1]

        # Function to compute slices for a given dimension size and number of divisions
        def get_slices(total_size, num_divisions):
            base = total_size // num_divisions
            remainder = total_size % num_divisions
            slices = []
            start = 0
            for i in range(num_divisions):
                end = start + base
                if i < remainder:
                    end += 1
                slices.append((start, end))
                start = end
            return slices

        # Get row and column slices
        row_slices = get_slices(M, R)
        col_slices = get_slices(N, C)

        # Generate all possible tile combinations of row and column slices
        tiles = []
        for rs in row_slices:
            r_start, r_end = rs
            for cs in col_slices:
                c_start, c_end = cs
                tile_slice = (slice(r_start, r_end), slice(c_start, c_end))
                tiles.append(tile_slice)

        return tiles

    def fixBackground(self, image):
        """Estimate and subtract the background from an image.

        This function estimates the background level in an image using a median-based
        approach combined with Gaussian fitting and radial basis function interpolation.
        It aims to provide a more accurate background estimation than a simple median
        filter, especially in images with varying background levels.

        Parameters
        ----------
        image : `NDArray`
            The input image as a NumPy array.

        Returns
        -------
        numpy.ndarray
            An array representing the estimated background level across the image.
        """
        # Find the median value in the image, which is likely to be
        # close to average background. Note this doesn't work well
        # in fields with high density or diffuse flux.
        initial_mask = image < self.config.brightnessThresh
        maxLikely = np.median(image[initial_mask], axis=None)

        # find all the pixels that are fainter than this
        # and find the std. This is just used as an initialization
        # parameter and doesn't need to be accurate.
        # choose a really large initial std
        initial_std = np.std(image[initial_mask], axis=None)
        # Do this 3 times for outlier rejection
        for _ in range(3):
            mask = image < maxLikely
            mask *= image > maxLikely - 5 * initial_std
            initial_std = (image[mask] - maxLikely).std()

        # Don't do anything if there are no pixels to check
        if np.any(mask):
            # use a minimizer to determine best mu and sigma for a Gaussian
            # given only samples below the mean of the Gaussian.
            result = minimize(
                self._neg_log_likelihood,
                (maxLikely, initial_std),
                args=(image[mask]),
                bounds=((maxLikely, None), (1e-8, None)),
            )
            mu_hat, sigma_hat = result.x
        else:
            mu_hat, sigma_hat = 0, 0

        # create a new masking threshold that is the determined
        # mean plus std from the fit
        threshhold = mu_hat + sigma_hat
        image_mask = (image < threshhold) * (image > mu_hat - 5 * sigma_hat)

        # create python slices that tile the image.
        tiles = self._tile_slices(image, 25, 25)

        yloc = []
        xloc = []
        values = []

        # for each box find the middle position and the median background
        # value in the window.
        for xslice, yslice in tiles:
            ypos = (yslice.stop - yslice.start) / 2 + yslice.start
            xpos = (xslice.stop - xslice.start) / 2 + xslice.start
            yloc.append(ypos)
            xloc.append(xpos)
            window = image[yslice, xslice][image_mask[yslice, xslice]]
            if window.size > 0:
                value = np.median(window)
            else:
                value = 0
            values.append(value)

        positions = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
        # create an interpolant for the background and interpolate over the image.
        inter = RBFInterpolator(
            np.vstack((yloc, xloc)).T, values, kernel="thin_plate_spline", degree=4, smoothing=0.05
        )
        backgrounds = inter(np.array(positions)[::-1].reshape(2, -1).T).reshape(image.shape)
        fixed = image - backgrounds
        backgrounds += np.median(fixed[(fixed > -3 * sigma_hat) * (fixed < 3 * sigma_hat)])

        return backgrounds

    def run(self, inputCoadd: Exposure):
        """Estimate a background for an input Exposure and remove it.

        Parameters
        ----------
        inputCoadd : `Exposure`
            The exposure the background will be removed from.

        Returns
        -------
        result : `Struct`
            A `Struct` that contains the exposure with the background removed.
            This `Struct` will have an attribute named ``outputCoadd``.

        """
        background = self.fixBackground(inputCoadd.image.array)
        # create a copy to mutate
        output = ExposureF(inputCoadd, deep=True)
        output.image.array -= background
        return Struct(outputCoadd=output)


class PrettyPictureStarFixerConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap"),
):
    inputCoadd = Input(
        doc=("Input coadd for which the background is to be removed"),
        name="pretty_picture_coadd_bg_subtracted",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
        multiple=True,
    )
    outputCoadd = Output(
        doc="The coadd with the background fixed and subtracted",
        name="pretty_picture_coadd_fixed_stars",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
        multiple=True,
    )


class PrettyPictureStarFixerConfig(PipelineTaskConfig, pipelineConnections=PrettyPictureStarFixerConnections):
    brightnessThresh = Field[float](
        doc="The flux value below which pixels with SAT or NO_DATA bits will be ignored"
    )
    maxFixSize = Field[int](
        doc="Any contiguous region with more than this number of pixels will not be fixed", default=3000
    )


class PrettyPictureStarFixerTask(PipelineTask):
    """This class fixes up regions in an image where there is no, or bad data.

    The fixes done by this task are overwhelmingly comprised of the cores of
    bright stars for which there is no data.
    """

    _DefaultName = "prettyPictureStarFixerTask"
    ConfigClass = PrettyPictureStarFixerConfig

    config: ConfigClass

    def run(self, inputs: Mapping[str, ExposureF]) -> Struct:
        """Fix areas in an image where this is no data, most likely to be
        the cores of bright stars.

        Because we want to have consistent fixes accross bands, this method
        relies on supplying all bands and fixing pixels that are marked
        as having a defect in any band even if within one band there  is
        no issue.

        Parameters
        ----------
        inputs : `Mapping` of `str` to `ExposureF`
            This mapping has keys of band as a `str` and the corresponding
            ExposureF as a value.

        Returns
        -------
        results : `Struct` of `Mapping` of `str` to `ExposureF`
            A `Struct` that has a mapping of band to `ExposureF`. The `Struct`
            has an attribute named ``results``.

        """
        # make the joint mask of all the channels
        doJointMaskInit = True
        for imageExposure in inputs.values():
            maskDict = imageExposure.mask.getMaskPlaneDict()
            if doJointMaskInit:
                jointMask = np.zeros(imageExposure.mask.array.shape, dtype=imageExposure.mask.array.dtype)
                doJointMaskInit = False
            jointMask |= imageExposure.mask.array

        sat_bit = maskDict["SAT"]
        no_data_bit = maskDict["NO_DATA"]
        together = (jointMask & 2**sat_bit).astype(bool) | (jointMask & 2**no_data_bit).astype(bool)

        # use the last imageExposure as it is likely close enough across all bands
        bright_mask = imageExposure.image.array > self.config.brightnessThresh

        # dilate the mask a bit, this helps get a bit fainter mask without starting
        # to include pixels in an irregular shape, as only the star cores should be
        # fixed.
        both = together & bright_mask

        # filter out extremely large areas
        labels, num_features = label(both)
        for num in range(1, num_features + 1):
            label_mask = labels == num
            amount = np.sum(label_mask)
            if amount < self.config.maxFixSize:
                continue
            else:
                both[label_mask] = 0

        struct = np.array(((0, 1, 0), (1, 1, 1), (0, 1, 0)), dtype=bool)
        both = binary_dilation(both, struct, iterations=4).astype(bool)

        # do the actual fixing of values
        results = {}
        for band, imageExposure in inputs.items():
            if np.sum(both) > 0:
                inpainted = inpaint_biharmonic(imageExposure.image.array, both, split_into_regions=True)
                imageExposure.image.array[both] = inpainted[both]
            results[band] = imageExposure
        return Struct(results=results)

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        refs = inputRefs.inputCoadd
        sortedImages: dict[str, Exposure] = {}
        for ref in refs:
            key: str = cast(str, ref.dataId["band"])
            image = butlerQC.get(ref)
            sortedImages[key] = image

        outputs = self.run(sortedImages).results
        sortedOutputs = {}
        for ref in outputRefs.outputCoadd:
            sortedOutputs[ref.dataId["band"]] = ref

        for band, data in outputs.items():
            butlerQC.put(data, sortedOutputs[band])


class PrettyMosaicConnections(PipelineTaskConnections, dimensions=("tract", "skymap")):
    inputRGB = Input(
        doc="Individual RGB images that are to go into the mosaic",
        name="rgb_picture_array",
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
        name="rgb_picture_mask",
        storageClass="Mask",
        dimensions=("tract", "patch", "skymap"),
        multiple=True,
        deferLoad=True,
    )

    outputRGBMosaic = Output(
        doc="A RGB mosaic created from the input data stored as a 3d array",
        name="rgb_mosaic_array",
        storageClass="NumpyArray",
        dimensions=("tract", "skymap"),
    )


class PrettyMosaicConfig(PipelineTaskConfig, pipelineConnections=PrettyMosaicConnections):
    binFactor = Field[int](doc="The factor to bin by when producing the mosaic")


class PrettyMosaicTask(PipelineTask):
    """Combines multiple RGB arrays into one mosaic."""

    _DefaultName = "prettyMosaicTask"
    ConfigClass = PrettyMosaicConfig

    config: ConfigClass

    def run(
        self,
        inputRGB: Iterable[DeferredDatasetHandle],
        skyMap: BaseSkyMap,
        inputRGBMask: Iterable[DeferredDatasetHandle],
    ) -> Struct:
        r"""Assemble individual `NDArrays` into a mosaic.

        Each input is a `DeferredDatasetHandle` because they're loaded in one
        at a time to be placed into the mosaic to save memory.

        Parameters
        ----------
        inputRGB : `Iterable` of `DeferredDatasetHandle`
            `DeferredDatasetHandle`\ s pointing to RGB `NDArrays`.
        skyMap : `BaseSkyMap`
            The skymap that defines the relative position of each of the input
            images.
        inputRGBMask : `Iterable` of `DeferredDatasetHandle`
            `DeferredDatasetHandle`\ s pointing to masks for each of the
            corresponding images.

        Returns
        -------
        result : `Struct`
            The `Struct` containing the combined mosaic. The `Struct` has
            and attribute named ``outputRGBMosaic``.
        """
        # create the bounding region
        newBox = Box2I()
        # store the bounds as they are retrieved from the skymap
        boxes = []
        tractMaps = []
        for handle in inputRGB:
            dataId = handle.dataId
            tractInfo: TractInfo = skyMap[dataId["tract"]]
            patchInfo: PatchInfo = tractInfo[dataId["patch"]]
            bbox = patchInfo.getOuterBBox()
            boxes.append(bbox)
            newBox.include(bbox)
            tractMaps.append(tractInfo)

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
        tmpImg = None
        for box, handle, handleMask, tractInfo in zip(boxes, inputRGB, inputRGBMask, tractMaps):
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
                    dst=None,
                    dsize=shape,
                    fx=shape[0] / self.config.binFactor,
                    fy=shape[1] / self.config.binFactor,
                )
                rgbMask = cv2.resize(
                    rgbMask.array.astype(np.float32),
                    dst=None,
                    dsize=shape,
                    fx=shape[0] / self.config.binFactor,
                    fy=shape[1] / self.config.binFactor,
                )
            existing = ~np.all(consolidatedImage[*box.slices] == 0, axis=2)
            if tmpImg is None or tmpImg.shape != rgb.shape:
                ramp = np.linspace(0, 1, tractInfo.patch_border * 2)
                tmpImg = np.zeros(rgb.shape[:2])
                tmpImg[: tractInfo.patch_border * 2, :] = np.repeat(
                    np.expand_dims(ramp, 1), tmpImg.shape[1], axis=1
                )

                tmpImg[-1 * tractInfo.patch_border * 2 :, :] = np.repeat(  # noqa: E203
                    np.expand_dims(1 - ramp, 1), tmpImg.shape[1], axis=1
                )
                tmpImg[:, : tractInfo.patch_border * 2] = np.repeat(
                    np.expand_dims(ramp, 0), tmpImg.shape[0], axis=0
                )

                tmpImg[:, -1 * tractInfo.patch_border * 2 :] = np.repeat(  # noqa: E203
                    np.expand_dims(1 - ramp, 0), tmpImg.shape[0], axis=0
                )
                tmpImg = np.repeat(np.expand_dims(tmpImg, 2), 3, axis=2)

            consolidatedImage[*box.slices][~existing, :] = rgb[~existing, :]
            consolidatedImage[*box.slices][existing, :] = (
                tmpImg[existing] * rgb[existing]
                + (1 - tmpImg[existing]) * consolidatedImage[*box.slices][existing, :]
            )

            tmpMask = np.zeros_like(rgbMask.array)
            tmpMask[existing] = np.bitwise_or(
                rgbMask.array[existing], consolidatedMask[*box.slices][existing]
            )
            tmpMask[~existing] = rgbMask.array[~existing]
            consolidatedMask[*box.slices] = tmpMask

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
        r"""Make valid inputs for the run method from numpy arrays.

        Parameters
        ----------
        inputs : `Iterable` of `tuple` of `Mapping` and `NDArray`
            An iterable where each element is a tuble with the first
            element is a mapping that corresponds to an arrays dataId,
            and the second is an `NDArray`.

        Returns
        -------
        sortedImages : `dict` of `str` to `Exposure`
            A dictionary of `Exposure`\ s that keyed by the band they
            correspond to.
        """
        structuredInputs = []
        for dataId, array in inputs:
            structuredInputs.append(InMemoryDatasetHandle(inMemoryDataset=array, **dataId))

        return structuredInputs
