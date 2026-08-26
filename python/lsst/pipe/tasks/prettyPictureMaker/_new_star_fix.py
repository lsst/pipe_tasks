import numpy as np
import logging
from scipy.ndimage import binary_dilation, distance_transform_edt, binary_erosion, binary_closing, label
from typing import cast
from collections.abc import Iterable, Mapping


from lsst.afw.image import Exposure, ExposureF
from lsst.rubinoxide import rgb
from ._task import ChannelRGBConfig
from stellaRGB.functors import ColorScaler, BoundsRemapper
from stellaRGB.types import ScaleColorProtocol, RemapBoundsProtocol

from lsst.pipe.base import (
    PipelineTaskConfig,
    PipelineTaskConnections,
    PipelineTask,
    Struct,
    QuantumContext,
    InputQuantizedConnection,
    OutputQuantizedConnection,
)
from lsst.pex.config import Field, ConfigDictField
from lsst.pex.config.configurableActions import ConfigurableActionField
from lsst.skymap import BaseSkyMap

from lsst.pipe.base.connectionTypes import Input, Output

logger = logging.getLogger(__name__)


def _disk(r):
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def _circular_mask(mask, extent=0.9):
    """One circle per connected component, auto-sized from that component."""
    labl, n = label(mask)
    out = np.zeros_like(mask)
    ys, xs = np.ogrid[: mask.shape[0], : mask.shape[1]]
    for k in range(1, n + 1):
        comp = labl == k
        idx = np.argwhere(comp)
        cy, cx = idx[:, 0].mean(), idx[:, 1].mean()
        d = np.sqrt((idx[:, 0] - cy) ** 2 + (idx[:, 1] - cx) ** 2)
        r = np.percentile(d, 100 * extent)  # auto radius from this blob's reach
        out |= ((ys - cy) ** 2 + (xs - cx) ** 2) <= r * r
    return out


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
        doc="Fluxes above this value will be considered possibly saturated and will be filled, set to None to disable",
        optional=True,
    )
    channel_config = ConfigDictField(
        doc="A dictionary that maps band names to their rgb channel configurations",
        keytype=str,
        itemtype=ChannelRGBConfig,
        default={},
    )
    image_remapping_config = ConfigurableActionField[RemapBoundsProtocol](
        doc="Action controlling normalization process", default=BoundsRemapper
    )

    growth = Field[float](doc="how fast the constructed stelar profile should grow", default=0.02)

    def setDefaults(self):
        self.channel_config["i"] = ChannelRGBConfig(r=1, g=0, b=0)
        self.channel_config["r"] = ChannelRGBConfig(r=0, g=1, b=0)
        self.channel_config["g"] = ChannelRGBConfig(r=0, g=0, b=1)
        return super().setDefaults()


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

        Because we want to have consistent fixes across bands, this method
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

        no_data_bit = maskDict["NO_DATA"]
        no_data_mask = (jointMask & 2**no_data_bit).astype(bool)

        # use the last imageExposure as it is likely close enough across all bands
        # Anything sufficiently bright should be considered no_data
        if self.config.brightnessThresh:
            bright_mask = imageExposure.image.array > self.config.brightnessThresh
        else:
            bright_mask = np.zeros(imageExposure.image.array.shape, dtype=bool)

        both_pre = np.bitwise_or(no_data_mask, bright_mask)

        # dilate the mask a bit, this helps get a bit fainter mask without starting
        # to include pixels in an irregular shape, as only the star cores should be
        # fixed.
        struct = _disk(2).astype(bool)
        both = binary_dilation(both_pre, struct, iterations=2).astype(bool)

        input_rgb = np.zeros((*jointMask.shape, 3), dtype=np.float32)
        imageRArray = input_rgb[..., 0]
        imageGArray = input_rgb[..., 1]
        imageBArray = input_rgb[..., 2]

        for band, image in inputs.items():
            if band not in self.config.channel_config:
                logger.info(f"{band} image found but not requested in RGB image, skipping")
                continue
            mix = self.config.channel_config[band]
            if mix.r:
                imageRArray += mix.r * image.image.array
            if mix.g:
                imageGArray += mix.g * image.image.array
            if mix.b:
                imageBArray += mix.b * image.image.array

        # now need to find the ratio each color contributes
        ratios = {}
        for band, image in inputs.items():
            if band not in self.config.channel_config:
                logger.info(f"{band} image found but not requested in RGB image, skipping")
                continue
            mix = self.config.channel_config[band]
            if mix.r > 0.0:
                rRatio = imageRArray[0, 0] / (mix.r * image.image.array[0, 0])
            else:
                rRatio = 0.0
            if mix.g > 0.0:
                gRatio = imageGArray[0, 0] / (mix.g * image.image.array[0, 0])
            else:
                gRatio = 0.0
            if mix.b > 0.0:
                bRatio = imageBArray[0, 0] / (mix.b * image.image.array[0, 0])
            else:
                bRatio = 0.0
            ratios[band] = (rRatio, gRatio, bRatio)

        remapped = self.config.image_remapping_config(input_rgb)
        avg_scale = np.nanmax(remapped / input_rgb)
        Lab = rgb.RGB_to_Oklab(remapped.astype(np.float64), (0.31, 0.32))

        lum_copy = np.copy(Lab[..., 0])

        # create circularized mask, as most bright stars look like footballs. Only use
        # this mask for the L channel.
        both_lum = _circular_mask(both)

        # Inpaint the luminance channel
        lum_copy = rgb.inpaint_mask(
            lum_copy, both_lum, init_method="radial_rise", peak_amp=self.config.growth, radius=15
        )

        # reconstruct the color (a, b channels) of the saturated star by looking at the suraounding
        # area
        new_a, new_b = rgb.reconstruct_star_color(
            lum_copy,
            Lab[..., 1],
            Lab[..., 2],
            both,
            radius=90.0,
            bg_inner=200.0,
            bg_outer=300.0,
            blend=2.0,
            linear_rgb=remapped.astype(np.float64),
        )

        Lab[..., 0] = lum_copy
        Lab[..., 1] = new_a
        Lab[..., 2] = new_b

        rgb_back = rgb.Oklab_to_RGB(Lab, (0.31, 0.32))

        # Need to set the fluxes back to the original scale of the exposures instead of the normalized
        # units used in Lab conversion

        # scale it up by the average scale
        new_rgb = (rgb_back / avg_scale).astype(np.float32)

        # need to split apart the ratios for the individual arrays
        imageRArray = new_rgb[..., 0]
        imageGArray = new_rgb[..., 1]
        imageBArray = new_rgb[..., 2]
        for band, image in inputs.items():
            if band not in self.config.channel_config:
                logger.info(f"{band} image found but not requested in RGB image, skipping")
                continue
            mix = self.config.channel_config[band]
            match np.argmax(ratios[band]):
                case 0:
                    image.image.array = imageRArray / ratios[band][0] / mix.r
                case 1:
                    image.image.array = imageGArray / ratios[band][1] / mix.g
                case 2:
                    image.image.array = imageBArray / ratios[band][2] / mix.b

        return Struct(results=inputs)

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
