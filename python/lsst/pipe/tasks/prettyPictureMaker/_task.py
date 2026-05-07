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

import colour
import copy
import itertools
from collections.abc import Iterable, Mapping
from lsst.afw.image import ExposureF
import numpy as np
from typing import TYPE_CHECKING, cast, Any
from lsst.skymap import BaseSkyMap

from scipy.stats import halfnorm, mode
from scipy.ndimage import binary_dilation
from scipy.interpolate import RBFInterpolator
from skimage.restoration import inpaint_biharmonic

from lsst.daf.butler import Butler, DataCoordinate, DeferredDatasetHandle
from lsst.daf.butler import DatasetRef
from lsst.images import ColorImage, Projection, Box, TractFrame
from lsst.pex.config import Field, Config, ConfigDictField, ListField, ChoiceField
from lsst.pex.config.configurableActions import ConfigurableActionField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
    InMemoryDatasetHandle,
    NoWorkFound,
    QuantaAdjuster,
)
from lsst.rubinoxide import rbf_interpolator
import cv2

from lsst.pipe.base.connectionTypes import Input, Output
from lsst.geom import Box2I, Point2I, Extent2I
from lsst.afw.image import Exposure, Mask
from lsst.skymap import Index2D

from ._plugins import plugins
from ._colorMapper import lsstRGB
from ._utils import FeatheredMosaicCreator
from ._functors import (
    BoundsRemapper,
    ColorScaler,
    LumCompressor,
    ExposureBracketer,
    GamutFixer,
    LocalContrastEnhancer,
)

import logging
import tempfile

logger = logging.getLogger(__name__)

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
        name="pretty_coadd",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
        multiple=True,
    )

    skyMap = Input(
        doc="The skymap which the data has been mapped onto",
        storageClass="SkyMap",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        dimensions=("skymap",),
    )

    outputRGB = Output(
        doc="A RGB image created from the input data stored as a 3d array",
        name="rgb_picture",
        storageClass="ColorImage",
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


class PrettyPictureConfig(PipelineTaskConfig, pipelineConnections=PrettyPictureConnections):
    channelConfig = ConfigDictField(
        doc="A dictionary that maps band names to their rgb channel configurations",
        keytype=str,
        itemtype=ChannelRGBConfig,
        default={},
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
    recenterNoise = Field[float](
        doc="Recenter the noise away from zero. Supplied value is in units of sigma",
        optional=True,
        default=None,
    )
    noiseSearchThreshold = Field[float](
        doc=(
            "Flux threshold below which most flux will be considered noise, used to estimate noise properties"
        ),
        default=2,
    )
    maxNoiseImbalance = Field[float](
        doc=(
            "When recentering noise, if the ratio of counts of positive pixels, to negative pixesl passes "
            "this threhsold, consider there to be extened low flux and only estimate noise below zero."
        ),
        default=1.5,
    )
    doPsfDeconvolve = Field[bool](
        doc="Use the PSF in a Richardson-Lucy deconvolution on the luminance channel.", default=False
    )
    doPSFDeconcovlve = Field[bool](
        doc="Use the PSF in a Richardson-Lucy deconvolution on the luminance channel.",
        default=False,
        deprecated="This field will be removed in v32. Use doPsfDeconvolve instead.",
        optional=True,
    )
    doRemapGamut = Field[bool](
        doc="Apply a color correction to unrepresentable colors; if False, clip them.", default=True
    )
    doExposureBrackets = Field[bool](
        doc="Apply exposure bracketing to aid in dynamic range compression", default=True
    )
    doLocalContrast = Field[bool](doc="Apply local contrast optimizations to luminance.", default=True)

    imageRemappingConfig = ConfigurableActionField[BoundsRemapper](
        doc="Action controlling normalization process"
    )
    luminanceConfig = ConfigurableActionField[LumCompressor](
        doc="Action controlling luminance scaling when making an RGB image"
    )
    localContrastConfig = ConfigurableActionField[LocalContrastEnhancer](
        doc="Action controlling the local contrast correction in RGB image production"
    )
    colorConfig = ConfigurableActionField[ColorScaler](
        doc="Action to control the color scaling process in RGB image production"
    )
    exposureBracketerConfig = ConfigurableActionField[ExposureBracketer](
        doc=(
            "Exposure scaling action used in creating multiple exposures with different scalings which will "
            "then be fused into a final image"
        ),
    )
    gamutMapperConfig = ConfigurableActionField[GamutFixer](
        doc="Action to fix pixels which lay outside RGB color gamut"
    )

    exposureBrackets = ListField[float](
        doc=(
            "Exposure scaling factors used in creating multiple exposures with different scalings which will "
            "then be fused into a final image"
        ),
        optional=True,
        default=[1.25, 1, 0.75],
        deprecated=(
            "This field will stop working in v31 and be removed in v32, "
            "please set exposureBracketerConfig.exposureBrackets"
        ),
    )
    gamutMethod = ChoiceField[str](
        doc="If doRemapGamut is True this determines the method",
        default="inpaint",
        allowed={
            "mapping": "Use a mapping function",
            "inpaint": "Use surrounding pixels to determine likely value",
        },
        deprecated="This field will stop working in v31 and be removed in v32, please set gamutMapperConfig",
    )

    def setDefaults(self):
        self.channelConfig["i"] = ChannelRGBConfig(r=1, g=0, b=0)
        self.channelConfig["r"] = ChannelRGBConfig(r=0, g=1, b=0)
        self.channelConfig["g"] = ChannelRGBConfig(r=0, g=0, b=1)
        return super().setDefaults()

    def _handle_deprecated(self):
        """Handle deprecated configuration migration.

        This method migrates deprecated configuration fields to their new
        locations in sub-configurations. It checks the configuration history
        to determine if deprecated fields were explicitly set and updates
        the new configuration locations accordingly.

        Notes
        -----
        The following deprecated fields are migrated:
        - ``gamutMethod`` -> ``gamutMapperConfig.gamutMethod``
        - ``exposureBrackets`` -> ``exposureBracketerConfig.exposureBrackets``
        - ``doLocalContrast`` -> ``localContrastConfig.doLocalContrast``
        - ``doPSFDeconcovlve`` -> ``doPsfDeconvolve``
        """
        # check if gamutMethod is set
        if len(self._history["gamutMethod"]) > 1:
            # This has been set in config, update it in the new location
            self.gamutMapperConfig.gamutMethod = self.gamutMethod

        if len(self._history["exposureBrackets"]) > 1:
            self.exposureBracketerConfig.exposureBrackets = self.exposureBrackets
            if self.exposureBrackets is None:
                self.doExposureBrackets = False

        if len(self.localContrastConfig._history["doLocalContrast"]) > 1:
            self.doLocalContrast = self.localContrastConfig.doLocalContrast

        # Handle doPsfDeconcovlve typo fix
        if len(self._history["doPSFDeconcovlve"]) > 1:
            self.doPsfDeconvolve = self.doPSFDeconcovlve

    def freeze(self):
        # ensure this is not already frozen
        if self._frozen is not True:
            self._handle_deprecated()
        super().freeze()


class PrettyPictureTask(PipelineTask):
    """Turns inputs into an RGB image."""

    _DefaultName = "prettyPicture"
    ConfigClass = PrettyPictureConfig

    config: ConfigClass

    def _find_normal_stats(self, array):
        """Calculate standard deviation from negative values using half-normal distribution.

        Raises
        ------
        ValueError
            Array dimension validation fails.

        Parameters
        ----------
        array : `numpy.array`
            Input array of numerical values.

        Returns
        -------
        mean : `float`
            The central moment of the distribution
        sigma : `float`
            Estimated standard deviation from negative values. Returns np.inf if:
            - No negative values exist in the array
            - Half-normal fitting fails
        """
        # Extract negative values efficiently
        values_noise = array[array < self.config.noiseSearchThreshold]

        # find the mode
        center = mode(np.round(values_noise, 2)).mode

        # extract the negative values
        values_neg = array[array < center]

        # Return infinity if no negative values found
        if values_neg.size == 0:
            return 0, np.inf

        try:
            # Fit half-normal distribution to absolute negative values
            mu, sigma = halfnorm.fit(np.abs(values_neg - center))
            mu += center
        except (ValueError, RuntimeError):
            # Handle fitting failures (e.g., constant data, optimization issues)
            return 0, np.inf

        # examine for excess positive flux, this means there is contaminating signal
        new_cut = array[array < (mu + 3 * sigma)]
        positivity_ratio = np.sum(new_cut > mu) / np.sum(new_cut < mu)

        if positivity_ratio > self.config.maxNoiseImbalance:
            # This means there is an excess flux, possibly diffuse source,
            # only estimate around zero.
            mu, sigma = halfnorm.fit(np.abs(values_noise[values_noise < 0]))

        return mu, sigma

    def _match_sigmas_and_recenter(self, *arrays, factor=1):
        """Scale array values to match minimum standard deviation across arrays
        and recenter noise.

        Adjusts values below each array's sigma by scaling and shifting them to
        align with the minimum sigma value across all input arrays. This operates
        in-place for efficiency.

        Parameters
        ----------
        *arrays : any number of `numpy.array`
            Variable number of input arrays to process.
        factor : float, optional
            Scaling factor for adjustments (default: 1).

        """
        # Calculate standard deviations for all arrays
        sigmas = []
        mus = []
        for arr in arrays:
            m, s = self._find_normal_stats(arr)
            mus.append(m)
            sigmas.append(s)
        mus = np.array(mus)
        sigmas = np.array(sigmas)

        # If no sigmas could be determined, return the original
        # arrays.
        if not np.any(np.isfinite(sigmas)):
            return

        min_sig = np.min(sigmas)

        for mu, sigma, array in zip(mus, sigmas, arrays):
            # Identify values below the array's sigma threshold
            lower_pos = (array - mu) < sigma

            # Skip processing if sigma is invalid
            if not np.isfinite(sigma):
                continue

            # Calculate scaling ratio relative to minimum sigma
            sigma_ratio = min_sig / sigma

            # Apply adjustment to qualifying values
            array[lower_pos] = (array[lower_pos] - mu) * sigma_ratio + min_sig * factor

    def run(
        self,
        images: Mapping[str, Exposure],
        image_wcs: Projection[Any] | None = None,
        image_box: Box | None = None,
    ) -> Struct:
        """Turns the input arguments in arguments into an RGB array.

        Parameters
        ----------
        images : `Mapping` of `str` to `Exposure`
            A mapping of input images and the band they correspond to.
        image_wcs : `~lsst.images.Projection`, optional
            A projection describing the sky coordinate of each pixel.
        image_box : `~lsst.images.Box`, optional
            A box that defines this image as part of a larger region.

        Returns
        -------
        result : `Struct`
            A struct with the corresponding RGB image, and mask used in
            RGB image construction. The struct will have the attributes
            outputRGB and outputRGBMask. Each of the outputs will
            be a `~lsst.images.ColorImage` object.

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
            if band not in self.config.channelConfig:
                logger.info(f"{band} image found but not requested in RGB image, skipping")
                continue
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

        if self.config.recenterNoise:
            self._match_sigmas_and_recenter(
                imageRArray, imageGArray, imageBArray, factor=self.config.recenterNoise
            )

        # assert for typing reasons
        assert jointMask is not None
        # Run any image level correction plugins
        colorImage = np.zeros((*imageRArray.shape, 3))
        colorImage[:, :, 0] = imageRArray
        colorImage[:, :, 1] = imageGArray
        colorImage[:, :, 2] = imageBArray
        for plug in plugins.partial():
            colorImage = plug(colorImage, jointMask, maskDict, self.config)

        # Filter the local contrast parameters for diffusion that are None
        # This is so we only apply key word overrides that are specifically set.
        local_contrast_config = self.config.localContrastConfig.toDict()
        to_remove = []
        for k, v in local_contrast_config["diffusionFunction"].items():
            if v is None:
                to_remove.append(k)
        for item in to_remove:
            local_contrast_config["diffusionControl"].pop(item)

        colorImage = lsstRGB(
            colorImage[:, :, 0],
            colorImage[:, :, 1],
            colorImage[:, :, 2],
            local_contrast=self.config.localContrastConfig if self.config.doLocalContrast else None,
            scale_lum=self.config.luminanceConfig,
            scale_color=self.config.colorConfig,
            remap_bounds=self.config.imageRemappingConfig,
            bracketing_function=(
                self.config.exposureBracketerConfig if self.config.doExposureBrackets else None
            ),
            gamut_remapping_function=self.config.gamutMapperConfig if self.config.doRemapGamut else None,
            cieWhitePoint=tuple(self.config.cieWhitePoint),  # type: ignore
            psf=psf if self.config.doPsfDeconvolve else None,
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
        return Struct(
            outputRGB=ColorImage(colorImage.astype(dtype), bbox=image_box, projection=image_wcs),
            outputRGBMask=lsstMask,
        )  # type: ignore

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        imageRefs: list[DatasetRef] = inputRefs.inputCoadds
        sortedImages = self.makeInputsFromRefs(imageRefs, butlerQC)
        if not sortedImages:
            requested = ", ".join(self.config.channelConfig.keys())
            raise NoWorkFound(f"No input images of band(s) {requested}")

        # get the patch tract bounding box and wcs
        skymap = butlerQC.get(inputRefs.skyMap)
        quantumDataId = butlerQC.quantum.dataId
        tractInfo = skymap[quantumDataId["tract"]]
        patchInfo = tractInfo[quantumDataId["patch"]]
        outputs = self.run(
            images=sortedImages,
            image_wcs=Projection.from_legacy(
                patchInfo.wcs,
                TractFrame(
                    skymap=quantumDataId["skymap"],
                    tract=quantumDataId["tract"],
                    bbox=Box.from_legacy(tractInfo.bbox),
                ),
            ),
            image_box=Box.from_legacy(patchInfo.getOuterBBox()),
        )
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
            A dictionary of `Exposure`\ s keyed by the band they
            correspond to.
        """
        sortedImages: dict[str, Exposure] = {}
        for ref in refs:
            key: str = cast(str, ref.dataId["band"])
            image = butler.get(ref)
            sortedImages[key] = image
        return sortedImages

    def makeInputsFromArrays(self, **kwargs) -> dict[str, DeferredDatasetHandle]:
        r"""Make valid inputs for the run method from numpy arrays.

        Parameters
        ----------
        kwargs : `numpy.ndarray`
            This is standard python kwargs where the left side of the equals
            is the data band, and the right side is the corresponding `numpy.ndarray`
            array.

        Returns
        -------
        sortedImages : `dict` of `str` to \
                `~lsst.daf.butler.DeferredDatasetHandle`
            A dictionary of `~lsst.daf.butlger.DeferredDatasetHandle`\ s keyed
            by the band they correspond to.
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
        sortedImages : `dict` of `int` to \
                `~lsst.daf.butler.DeferredDatasetHandle`
            A dictionary of `~lsst.daf.butler.DeferredDatasetHandle`\ s keyed
            by the band they correspond to.
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
        multiple=True,
    )
    outputCoadd = Output(
        doc="The coadd with the background fixed and subtracted",
        name="pretty_picture_coadd_bg_subtracted",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
    )

    def adjust_all_quanta(self, adjuster: QuantaAdjuster) -> None:
        # At this stage of a QG build, we won't necessarily have pruned
        # quanta that don't have inputs, so instead of a set of quantum data
        # IDs, we want to start from a set of quantum *input* data IDs.  We'll
        # just ignore quanta that don't; they'll get prune later.
        flat_data_ids: set[DataCoordinate] = set()
        for quantum_data_id in adjuster.iter_data_ids():
            flat_data_ids.update(adjuster.get_inputs(quantum_data_id)["inputCoadd"])
        # Reorganize task data IDs into
        #
        # {skymap: {(band, tract): [patches]}}}
        #
        # It's unlikely we'll ever get more than one skymap in a single QG
        # build, of course, but not impossible.
        hierarchical_data_ids: dict[str, dict[tuple[str, int], list[int]]] = {}
        for quantum_data_id in flat_data_ids:
            hierarchical_data_ids.setdefault(quantum_data_id["skymap"], {}).setdefault(
                (quantum_data_id["band"], quantum_data_id["tract"]), []
            ).append(quantum_data_id["patch"])
        for skymap_name, data_ids_for_skymap in hierarchical_data_ids.items():
            # We need to load the skyMap to turn single-int patch_id IDs into
            # grid (x, y) pairs, so we can offset those to find neighbors.
            skyMap = adjuster.butler.get("skyMap", skymap=skymap_name)
            for (band, tract_id), patches in data_ids_for_skymap.items():
                tract_info = skyMap[tract_id]
                base_data_id = DataCoordinate.standardize(
                    skymap=skymap_name,
                    tract=tract_id,
                    band=band,
                    universe=adjuster.butler.dimensions,
                )
                for patch_id in patches:
                    patch_index: Index2D = tract_info.getPatchIndexPair(patch_id)
                    quantum_data_id = DataCoordinate.standardize(base_data_id, patch=patch_id)
                    # Find all adjacent patches (including corner neighbors).
                    for offset_x, offset_y in itertools.product((-1, 0, 1), (-1, 0, 1)):
                        if not (offset_x or offset_y):
                            # Skip the input that matches the quantum data ID;
                            # it's already got an edge to this quantum.
                            continue
                        neighbor_patch_id = tract_info.getSequentialPatchIndexFromPair(
                            Index2D(x=patch_index.x + offset_x, y=patch_index.y + offset_y)
                        )
                        neighbor_data_id = DataCoordinate.standardize(base_data_id, patch=neighbor_patch_id)
                        if neighbor_data_id not in flat_data_ids:
                            # Skip inputs that don't exist, either because
                            # they didn't have data or because they're just a
                            # totally invalid patch index.
                            continue
                        # Finally we can add an edge from the neighboring
                        # input to this quantum.
                        adjuster.add_input(quantum_data_id, "inputCoadd", neighbor_data_id)


class PrettyPictureBackgroundFixerConfig(
    PipelineTaskConfig, pipelineConnections=PrettyPictureBackgroundFixerConnections
):
    use_detection_mask = Field[bool](
        doc="Use the detection mask to determine background instead of empirically finding it in this task",
        default=False,
    )
    num_background_bins = Field[int](
        doc="The number of bins along each axis when determining background", default=5
    )
    min_bin_fraction = Field[float](
        doc="Bins with fewer pixels than this fraction of the total will be ignored", default=0.1
    )

    pos_sigma_multiplier = Field[float](
        doc="How many sigma to consider as background in the positive direction", default=2
    )
    extra_pixel_rad = Field[int](
        doc=(
            "If there are neighboring input images consider control points that are radius of input image"
            ", (x^2 + x^2)^(1/2), plus this many pixels away from the center of the input image as control"
            "points to use."
        ),
        default=2000,
    )
    max_flux_imbalance = Field[float](
        doc="When determining background, if the ratio of counts of positive pixels, to negative pixesl passes "
        "this threhsold, consider there to be extened low flux and only estimate noise below zero.",
        default=1.7,
    )
    max_search_flux = Field[float](
        doc="Pixels above this value should never be considered background", default=3
    )


class PrettyPictureBackgroundFixerTask(PipelineTask):
    """Empirically flatten an images background.

    Many astrophysical images have backgrounds with imperfections in them.
    This Task attempts to determine control points which are considered
    background values, and fits a radial basis function model to those
    points. This model is then subtracted off the image.

    """

    _DefaultName = "prettyPictureBackgroundFixer"
    ConfigClass = PrettyPictureBackgroundFixerConfig

    config: ConfigClass

    def _tile_slices(self, arr, R, C):
        """Generate slices for tiling an array.

        This function divides an array into a grid of tiles and returns a list of
        slice objects representing each tile.  It handles cases where the array
        dimensions are not evenly divisible by the number of tiles in each
        dimension, distributing the remainder among the tiles.

        Parameters
        ----------
        arr : `numyp.ndarray`
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
        def get_slices(total_size: int, num_divisions: int) -> list[tuple[int, int]]:
            """Generate slice ranges for dividing a size into equal parts.

            Parameters
            ----------
            total_size : `int`
                Total size to be divided into slices.
            num_divisions : `int`
                Number of divisions to create.

            Returns
            -------
            `list` of `tuple` of `int`
                List of (start, end) tuples representing each slice.

            Notes
            -----
            This function divides the total_size into num_divisions equal parts.
            If the division is not exact, the remainder is distributed by adding
            1 to the first 'remainder' slices, ensuring balanced distribution.
            """
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

    @staticmethod
    def _findImageStatistics(image, pos_sigma_mult=1, max_flux_imbalance=1.7, max_search_flux=3):
        """Find pixels that are likely to be background based on image statistics.

        This method estimates background pixels by analyzing the distribution of
        pixel values in the image. It uses the median as an estimate of the background
        level and fits a half-normal distribution to values below the median to
        determine the background sigma. Pixels below a threshold (mean + sigma) are
        classified as background.

        Parameters
        ----------
        image : `numpy.ndarray`
            Input image array for which to find background pixels.
        pos_sigma_mult : `float`
            How many sigma to consider as background in the positive direction
        max_flux_imbalance : `float`
            Limit on the ratio of the area below the determined average to that
            above. If the ratio is in excess of this value, then there is
            assumed to be diffuse background flux in the image, and zero is
            assumed to be the background level. Some excess flux is expected as
            there is a real distribution of faint flux.
        max_serach_flux : `float`
            Pixels above this value should never be considered background
        """
        # Find the median value in the image, which is likely to be
        # close to average background. Note this doesn't work well
        # in fields with high density or diffuse flux.
        maxLikely = np.median(image[image < max_search_flux], axis=None)

        # find all the pixels that are fainter than this
        # and find the std. This is just used as an initialization
        # parameter and doesn't need to be accurate.
        mask = image < maxLikely
        initial_std = (image[mask] - maxLikely).std()

        # new estimate
        sub_image = image[image < max(initial_std + 3 * initial_std, max_search_flux)]

        center = mode(np.round(sub_image, 2)).mode

        # Don't do anything if there are no pixels to check
        if np.any(mask):
            # use a minimizer to determine best mu and sigma for a Gaussian
            # given only samples below the mean of the Gaussian.
            mu_hat, sigma_hat = halfnorm.fit(np.abs(image[image < center] - center))
            mu_hat += center
            # mu_hat = maxLikely
        else:
            mu_hat, sigma_hat = (maxLikely, 2 * initial_std)

        new_cut = image[image < (mu_hat + 3 * sigma_hat)]
        positivity_ratio = np.sum(new_cut > mu_hat) / np.sum(new_cut < mu_hat)

        if positivity_ratio > max_flux_imbalance:
            # This means there is an excess flux, possibly diffuse source,
            # assume all flux is diffuse
            return np.inf, -np.inf

        # create a new masking threshold that's the determined
        # mean plus std from the fit
        threshhold_pos = min(mu_hat + pos_sigma_mult * sigma_hat, max_search_flux)
        threshhold_neg = mu_hat - pos_sigma_mult * sigma_hat
        return threshhold_pos, threshhold_neg

    def findBackgroundPixels(self, image, threshold_pair=None):
        """Find pixels that are likely to be background based on image statistics.

        This method estimates background pixels by analyzing the distribution of
        pixel values in the image. It uses the median as an estimate of the background
        level and fits a half-normal distribution to values below the median to
        determine the background sigma. Pixels below a threshold (mean + sigma) are
        classified as background.

        Parameters
        ----------
        image : `numpy.ndarray`
            Input image array for which to find background pixels.
        pos_sigma_mult : `float`
            How many sigma to consider as background in the positive direction
        max_flux_imbalance : `float`
            Limit on the ratio of the area below the determined average to that
            above. If the ratio is in excess of this value, then there is
            assumed to be diffuse background flux in the image, and zero is
            assumed to be the background level. Some excess flux is expected as
            there is a real distribution of faint flux.
        max_serach_flux : `float`
            Pixels above this value should never be considered background

        Returns
        -------
        result : `numpy.ndarray`
            Boolean mask array where True indicates background pixels.

        Notes
        -----
        This method works best for images with relatively uniform background. It may
        not perform well in fields with high density or diffuse flux, as noted in
        the implementation comments.
        """
        if threshold_pair is None:
            threshhold_pos, threshhold_neg = self._findImageStatistics(
                image,
                self.config.pos_sigma_multiplier,
                self.config.max_flux_imbalance,
                self.config.max_search_flux,
            )
        else:
            threshhold_pos, threshhold_neg = threshold_pair
        if np.isinf(threshhold_pos):
            return None
        # mean plus std from the fit
        image_mask = (image < threshhold_pos) * (image > threshhold_neg)
        return image_mask

    def _findControlPoints(
        self,
        exposure: Exposure,
        origin: Point2I,
        use_detection_mask: bool = False,
        threshhold_pair: tuple[float, float] | None = None,
    ):
        if use_detection_mask:
            mask_plane_dict = exposure.mask.getMaskPlaneDict()
            image_mask = ~(exposure.mask.array & 2 ** mask_plane_dict["DETECTED"])
        else:
            image_mask = self.findBackgroundPixels(exposure.image.array, threshold_pair=threshhold_pair)

        yloc = []
        xloc = []
        values = []

        if image_mask is None:
            logger.info("returning early from _findControlPoints")
            return values, yloc, xloc

        tiles = self._tile_slices(
            exposure.image.array, self.config.num_background_bins, self.config.num_background_bins
        )

        # adjust for the offset of the origin
        this_origin: Point2I = exposure.getBBox().getBegin()
        offset: Extent2I = this_origin - origin
        x_offset = offset.getX()
        y_offset = offset.getY()

        # for each box find the middle position and the median background
        # value in the window.
        for i, (xslice, yslice) in enumerate(tiles):
            ypos = (yslice.stop - yslice.start) / 2 + yslice.start
            xpos = (xslice.stop - xslice.start) / 2 + xslice.start
            window = exposure.image.array[yslice, xslice][image_mask[yslice, xslice]]
            # make sure each bin is at least 1% filled
            min_fill = int((yslice.stop - yslice.start) ** 2 * self.config.min_bin_fraction)
            if window.size > min_fill:
                value = np.mean(window)
            else:
                continue
            values.append(value)
            yloc.append(ypos + y_offset)
            xloc.append(xpos + x_offset)

        return values, yloc, xloc

    def fixBackground(self, image, yloc: list[float], xloc: list[float], values: list[float]):
        """Estimate and subtract the background from an image.

        This function estimates the background level in an image using supplied control
        values using Gaussian fitting and radial basis function interpolation.
        It aims to provide a more accurate background estimation than a simple median
        filter, especially in images with varying background levels.

        Parameters
        ----------
        image : `numpy.ndarray`
            The input image as a NumPy array.
        yloc : `list` of `float`
            The list of y control points
        xloc : `list` of `float`
            The list of x control points
        values : `list` of `float`
            The list of the values at the control points

        Returns
        -------
        numpy.ndarray
            An array representing the estimated background level across the image.
        """

        # create an interpolant for the background and interpolate over the image.
        inter = RBFInterpolator(
            np.vstack((yloc, xloc)).T,
            values,
            kernel="thin_plate_spline",
            degree=4,
            smoothing=0.05,
            neighbors=None,
        )

        backgrounds = rbf_interpolator.fast_rbf_interpolation_on_grid(inter, image.shape)

        return backgrounds

    def run(self, inputCoadd: Exposure, neighbors: list[Exposure] | None = None):
        """Estimate a background for an input Exposure and remove it.

        Parameters
        ----------
        inputCoadd : `Exposure`
            The exposure the background will be removed from.
        neighbors : `list` of `Exposure`
            Neighboring `Exposure` objects that can be used to constrain
            backgrounds across boundaries.

        Returns
        -------
        result : `Struct`
            A `Struct` that contains the exposure with the background removed.
            This `Struct` will have an attribute named ``outputCoadd``.

        """
        origin = inputCoadd.getBBox().getBegin()
        input_y_dim, input_x_dim = inputCoadd.image.array.shape
        input_rad = np.sqrt((input_y_dim / 2) ** 2 + (input_x_dim / 2) ** 2)
        inside_rad = input_rad + self.config.extra_pixel_rad
        center_x = input_x_dim / 2
        center_y = input_y_dim / 2

        # calculate joint statistics
        if self.config.use_detection_mask is False:
            background_pixels = inputCoadd.image.array[inputCoadd.image.array < self.config.max_search_flux]
            if neighbors:
                for n_exp in neighbors:
                    background_pixels = np.append(
                        background_pixels, n_exp.image.array[n_exp.image.array < self.config.max_search_flux]
                    )
            joint_thresh = self._findImageStatistics(
                background_pixels,
                self.config.pos_sigma_multiplier,
                self.config.max_flux_imbalance,
                self.config.max_search_flux,
            )
            logger.info(f"####################### {joint_thresh}")
            # There is no background to be found, return early
            if np.isinf(joint_thresh[0]):
                # output = ExposureF(inputCoadd, deep=True)
                # return Struct(outputCoadd=output)
                joint_thresh = None

        else:
            joint_thresh = None

        values, yloc, xloc = self._findControlPoints(
            inputCoadd, origin, self.config.use_detection_mask, joint_thresh
        )

        if len(values) == 0:
            output = ExposureF(inputCoadd, deep=True)
            return Struct(outputCoadd=output)

        if neighbors:
            for n_exp in neighbors:
                tmp_values, tmp_yloc, tmp_xloc = self._findControlPoints(
                    n_exp, origin, self.config.use_detection_mask, joint_thresh
                )
                if len(tmp_values) == 0:
                    output = ExposureF(inputCoadd, deep=True)
                    return Struct(outputCoadd=output)

                for value, y_pos, x_pos in zip(tmp_values, tmp_yloc, tmp_xloc):
                    if np.sqrt((y_pos - center_y) ** 2 + (x_pos - center_x) ** 2) < inside_rad:
                        values.append(value)
                        yloc.append(y_pos)
                        xloc.append(x_pos)

        # At least 15 points are requred for TPS with 4th order polynomial
        if len(yloc) < 15:
            return Struct(outputCoadd=inputCoadd)

        background = self.fixBackground(inputCoadd.image.array, yloc, xloc, values)

        # create a copy to mutate
        output = ExposureF(inputCoadd, deep=True)
        output.image.array -= background
        return Struct(outputCoadd=output)

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        quantum_patch = butlerQC.quantum.dataId["patch"]
        primary_ref = None
        neighbor_refs = []
        for ref in inputRefs.inputCoadd:
            if quantum_patch == ref.dataId["patch"]:
                primary_ref = ref
            else:
                neighbor_refs.append(ref)
        if primary_ref is None:
            # This should be unreachable
            raise RuntimeError("There is a major problem input ref associated with quantum cant be found")
        inputCoadd = butlerQC.get(primary_ref)
        neighbors = []
        for n_ref in neighbor_refs:
            neighbors.append(butlerQC.get(n_ref))
        results = self.run(inputCoadd, neighbors=neighbors if neighbors else None)
        butlerQC.put(results, outputRefs)


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


class PrettyPictureStarFixerTask(PipelineTask):
    """This class fixes up regions in an image where there is no, or bad data.

    The fixes done by this task are overwhelmingly comprised of the cores of
    bright stars for which there is no data.
    """

    _DefaultName = "prettyPictureStarFixer"
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
        name="rgb_mosaic",
        storageClass="ColorImage",
        dimensions=("tract", "skymap"),
    )


class PrettyMosaicConfig(PipelineTaskConfig, pipelineConnections=PrettyMosaicConnections):
    binFactor = Field[int](doc="The factor to bin by when producing the mosaic", default=1)
    doDCID65Convert = Field[bool](
        "Force the output to be converted from display p3 to DCI-D65 colorspace.", default=False
    )
    useLocalTemp = Field[bool](doc="Use the current directory when creating local temp files.", default=False)


class PrettyMosaicTask(PipelineTask):
    """Combines multiple RGB arrays into one mosaic."""

    _DefaultName = "prettyMosaic"
    ConfigClass = PrettyMosaicConfig

    config: ConfigClass

    def run(
        self,
        inputRGB: Iterable[DeferredDatasetHandle],
        skyMap: BaseSkyMap,
        inputRGBMask: Iterable[DeferredDatasetHandle],
    ) -> Struct:
        r"""Assemble individual `numpy.ndarrays` into a mosaic.

        Each input is a `~lsst.daf.butler.DeferredDatasetHandle` because
        they're loaded in one at a time to be placed into the mosaic to save
        memory.

        Parameters
        ----------
        inputRGB : `Iterable` of `~lsst.daf.butler.DeferredDatasetHandle`
            `~lsst.daf.butler.DeferredDatasetHandle`\ s pointing to RGB
            `numpy.ndarrays`.
        skyMap : `BaseSkyMap`
            The skymap that defines the relative position of each of the input
            images.
        inputRGBMask : `Iterable` of `~lsst.daf.butler.DeferredDatasetHandle`
            `~lsst.daf.butler.DeferredDatasetHandle`\ s pointing to masks for
            each of the corresponding images.

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
            # This will be overwritten in the loop, but that is ok, because
            # it is the same for each patch.
            patch_grow: int = patchInfo.getCellInnerDimensions().getX()

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
        self.imageHandle = tempfile.NamedTemporaryFile(dir="." if self.config.useLocalTemp else None)
        self.maskHandle = tempfile.NamedTemporaryFile(dir="." if self.config.useLocalTemp else None)
        consolidatedImage = None
        consolidatedMask = None

        # Setup color space conversion in case they are used.
        d65 = copy.deepcopy(colour.models.RGB_COLOURSPACE_DCI_P3)
        dp3 = copy.deepcopy(colour.models.RGB_COLOURSPACE_DISPLAY_P3)
        d65.whitepoint = dp3.whitepoint
        d65.whitepoint_name = dp3.whitepoint_name

        # Actually assemble the mosaic
        maskDict = {}
        mosaic_maker = FeatheredMosaicCreator(patch_grow, self.config.binFactor)
        for box, handle, handleMask, tractInfo in zip(boxes, inputRGB, inputRGBMask, tractMaps):
            rgb = handle.get().array
            # convert to the dci-d65 colorspace
            if self.config.doDCID65Convert:
                rgb = colour.RGB_to_RGB(np.clip(rgb, 0, 1), dp3, d65)
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
                mask_array = rgbMask.array[:: self.config.binFactor, :: self.config.binFactor]
                rgbMask = Mask(*(mask_array.shape[::-1]))
            mosaic_maker.add_to_image(consolidatedImage, rgb, newBox, box, reverse=False)

            consolidatedMask[*box.slices] = np.bitwise_or(consolidatedMask[*box.slices], rgbMask.array)

        for plugin in plugins.full():
            if consolidatedImage is not None and consolidatedMask is not None:
                consolidatedImage = plugin(consolidatedImage, consolidatedMask, maskDict)
        # If consolidated image still None, that means there was no work to do.
        # Return an empty image instead of letting this task fail.
        if consolidatedImage is None:
            consolidatedImage = np.zeros((0, 0, 0), dtype=np.uint8)

        return Struct(outputRGBMosaic=ColorImage(consolidatedImage))

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
        inputs : `Iterable` of `tuple` of `Mapping` and `numpy.ndarray`
            An iterable where each element is a tuple with the first
            element is a mapping that corresponds to an arrays dataId,
            and the second is an `numpy.ndarray`.

        Returns
        -------
        sortedImages : `Iterable` of `~lsst.daf.butler.DeferredDatasetHandle`
            An iterable of `~lsst.daf.butler.DeferredDatasetHandle`\ s
            containing the input data.
        """
        structuredInputs = []
        for dataId, array in inputs:
            structuredInputs.append(InMemoryDatasetHandle(inMemoryDataset=array, **dataId))

        return structuredInputs
