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

"""Retrieve extended PSF model and subtract bright stars at visit level."""

__all__ = ["BrightStarSubtractConnections", "BrightStarSubtractConfig", "BrightStarSubtractTask"]

import logging
from typing import Any
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS, TAN_PIXELS
from lsst.afw.detection import Footprint, FootprintSet, Threshold
from lsst.afw.geom import SkyWcs, SpanSet, makeModifiedWcs
from lsst.afw.geom.transformFactory import makeTransform
from lsst.afw.image import ExposureF, ImageD, ImageF, MaskedImageF
from lsst.afw.math import BackgroundList, FixedKernel, WarpingControl, warpImage
from lsst.daf.butler import DataCoordinate
from lsst.geom import (
    AffineTransform,
    Box2I,
    Extent2D,
    Extent2I,
    Point2D,
    Point2I,
    SpherePoint,
    arcseconds,
    floor,
    radians,
)
from lsst.meas.algorithms import (
    LoadReferenceObjectsConfig,
    ReferenceObjectLoader,
)
from lsst.pex.config import ChoiceField, ConfigField, Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output, PrerequisiteInput
from lsst.utils.timer import timeMethod
from copy import deepcopy

NEIGHBOR_MASK_PLANE = "NEIGHBOR"

logger = logging.getLogger(__name__)


class BrightStarSubtractConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
    defaultTemplates={
        # "outputExposureName": "brightStar_subtracted",
        "outputExposureName": "postISRCCD",
        "outputBackgroundName": "brightStars",
        "badStampsName": "brightStars",
    },
):
    inputCalexp = Input(
        name="calexp",
        storageClass="ExposureF",
        doc="Background-subtracted input exposure from which to extract bright star stamp cutouts.",
        dimensions=("visit", "detector"),
    )
    inputBackground = Input(
        name="calexpBackground",
        storageClass="Background",
        doc="Background model for the input exposure, to be added back on during processing.",
        dimensions=("visit", "detector"),
    )
    inputExposure = Input(
        doc="Input exposure from which to subtract bright star stamps.",
        name="postISRCCD",
        storageClass="Exposure",
        dimensions=(
            "exposure",
            "detector",
        ),
    )
    inputExtendedPsf = Input(
        name="extendedPsf2",  # extendedPsfDetector ???
        storageClass="ImageF",  # MaskedImageF
        doc="Extended PSF model, built from stacking bright star cutouts.",
        dimensions=("band",),
    )
    refCat = PrerequisiteInput(
        doc="Reference catalog that contains bright star positions",
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        multiple=True,
        deferLoad=True,
    )
    # outputBadStamps = Output(
    #     doc="The stamps that are not normalized and consequently not subtracted from the exposure.",
    #     name="{badStampsName}_unsubtracted_stamps",
    #     storageClass="BrightStarStamps",
    #     dimensions=(
    #         "visit",
    #         "detector",
    #     ),
    # )

    outputExposure = Output(
        doc="Exposure with bright stars subtracted.",
        name="{outputExposureName}_subtracted",
        storageClass="ExposureF",
        dimensions=(
            "exposure",
            "detector",
        ),
    )
    outputBackgroundExposure = Output(
        doc="Exposure containing only the modelled bright stars.",
        name="{outputBackgroundName}_background",
        storageClass="ExposureF",
        dimensions=(
            "visit",
            "detector",
        ),
    )
    # scaledModels = Output(
    #     doc="Stamps containing models scaled to the level of stars",
    #     name="scaledModels",
    #     storageClass="BrightStarStamps",
    #     dimensions=(
    #         "visit",
    #         "detector",
    #     ),
    # )


class BrightStarSubtractConfig(PipelineTaskConfig, pipelineConnections=BrightStarSubtractConnections):
    """Configuration parameters for BrightStarSubtractTask"""

    doWriteSubtractor = Field[bool](
        doc="Should an exposure containing all bright star models be written to disk?",
        default=True,
    )
    doWriteSubtractedExposure = Field[bool](
        doc="Should an exposure with bright stars subtracted be written to disk?",
        default=True,
    )
    magLimit = Field[float](
        doc="Magnitude limit, in Gaia G; all stars brighter than this value will be subtracted",
        default=18,
    )
    minValidAnnulusFraction = Field[float](
        doc="Minimum number of valid pixels that must fall within the annulus for the bright star to be "
        "saved for subsequent generation of a PSF.",
        default=0.0,
    )
    numSigmaClip = Field[float](
        doc="Sigma for outlier rejection; ignored if annularFluxStatistic != 'MEANCLIP'.",
        default=4,
    )
    numIter = Field[int](
        doc="Number of iterations of outlier rejection; ignored if annularFluxStatistic != 'MEANCLIP'.",
        default=3,
    )
    warpingKernelName = ChoiceField[str](
        doc="Warping kernel",
        default="lanczos5",
        allowed={
            "bilinear": "bilinear interpolation",
            "lanczos3": "Lanczos kernel of order 3",
            "lanczos4": "Lanczos kernel of order 4",
            "lanczos5": "Lanczos kernel of order 5",
            "lanczos6": "Lanczos kernel of order 6",
            "lanczos7": "Lanczos kernel of order 7",
        },
    )
    scalingType = ChoiceField[str](
        doc="How the model should be scaled to each bright star; implemented options are "
        "`annularFlux` to reuse the annular flux of each stamp, or `leastSquare` to perform "
        "least square fitting on each pixel with no bad mask plane set.",
        default="leastSquare",
        allowed={
            "annularFlux": "reuse BrightStarStamp annular flux measurement",
            "leastSquare": "find least square scaling factor",
        },
    )
    annularFluxStatistic = ChoiceField[str](
        doc="Type of statistic to use to compute annular flux.",
        default="MEANCLIP",
        allowed={
            "MEAN": "mean",
            "MEDIAN": "median",
            "MEANCLIP": "clipped mean",
        },
    )
    badMaskPlanes = ListField[str](
        doc="Mask planes that, if set, lead to associated pixels not being included in the computation of "
        "the scaling factor (`BAD` should always be included). Ignored if scalingType is `annularFlux`, "
        "as the stamps are expected to already be normalized.",
        # Note that `BAD` should always be included, as secondary detected
        # sources (i.e., detected sources other than the primary source of
        # interest) also get set to `BAD`.
        # Lee: find out the value of "BAD" and set the nan values into that number in the mask plane(?)
        default=("BAD", "CR", "CROSSTALK", "EDGE", "NO_DATA", "SAT", "SUSPECT", "UNMASKEDNAN"),
    )
    subtractionBox = ListField[int](
        doc="Size of the stamps to be extracted, in pixels.",
        default=(250, 250),
    )
    subtractionBoxBuffer = Field[float](
        doc=(
            "'Buffer' (multiplicative) factor to be applied to determine the size of the stamp the "
            "processed stars will be saved in. This is also the size of the extended PSF model. The buffer "
            "region is masked and contain no data and subtractionBox determines the region where contains "
            "the data."
        ),
        default=1.1,
    )
    doApplySkyCorr = Field[bool](
        doc="Apply full focal plane sky correction before extracting stars?",
        default=True,
    )
    min_iterations = Field[int](
        doc="Minimum number of iterations to complete before evaluating changes in each iteration.",
        default=3,
    )
    refObjLoader = ConfigField[LoadReferenceObjectsConfig](
        doc="Reference object loader for astrometric calibration.",
    )
    maskWarpingKernelName = ChoiceField[str](
        doc="Warping kernel for mask.",
        default="bilinear",
        allowed={
            "bilinear": "bilinear interpolation",
            "lanczos3": "Lanczos kernel of order 3",
            "lanczos4": "Lanczos kernel of order 4",
            "lanczos5": "Lanczos kernel of order 5",
        },
    )
    # Cutout geometry
    stampSize = ListField[int](
        doc="Size of the stamps to be extracted, in pixels.",
        default=(251, 251),
    )
    stampSizePadding = Field[float](
        doc="Multiplicative factor applied to the cutout stamp size, to guard against post-warp data loss.",
        default=1.1,
    )
     # Star selection
    magRange = ListField[float](
        doc="Magnitude range in Gaia G. Cutouts will be made for all stars in this range.",
        default=[0, 18],
    )
    # excludeArcsecRadius = Field[float](
    #     doc="Stars with a star in the range ``excludeMagRange`` mag in ``excludeArcsecRadius`` are not used.",
    #     default=5,
    # )
    # excludeMagRange = ListField[float](
    #     doc="Stars with a star in the range ``excludeMagRange`` mag in ``excludeArcsecRadius`` are not used.",
    #     default=[0, 20],
    # )
    minAreaFraction = Field[float](
        doc="Minimum fraction of the stamp area, post-masking, that must remain for a cutout to be retained.",
        default=0.1,
    )
    # PSF Fitting
    # useExtendedPsf = Field[bool](
    #     doc="Use the extended PSF model to normalize bright star cutouts.",
    #     default=False,
    # )
    # doFitPsf = Field[bool](
    #     doc="Fit a scaled PSF and a pedestal to each bright star cutout.",
    #     default=True,
    # )
    useMedianVariance = Field[bool](
        doc="Use the median of the variance plane for PSF fitting.",
        default=False,
    )
    psfMaskedFluxFracThreshold = Field[float](
        doc="Maximum allowed fraction of masked PSF flux for PSF fitting to occur.",
        default=0.97,
    )

    # # Misc
    # loadReferenceObjectsConfig = ConfigField[LoadReferenceObjectsConfig](
    #     doc="Reference object loader for astrometric calibration.",
    # )


class BrightStarSubtractTask(PipelineTask):
    """Use an extended PSF model to subtract bright stars from a calibrated
    exposure (i.e. at single-visit level).

    This task uses both a set of bright star stamps produced by
    `~lsst.pipe.tasks.processBrightStars.ProcessBrightStarsTask`
    and an extended PSF model produced by
    `~lsst.pipe.tasks.extended_psf.MeasureExtendedPsfTask`.
    """

    ConfigClass = BrightStarSubtractConfig
    _DefaultName = "subtractBrightStars"

    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        # # Placeholders to set up Statistics if scalingType is leastSquare.
        # self.statsControl, self.statsFlag = None, None
        # # Warping control; only contains shiftingALg provided in config.

        super().__init__(*args, **kwargs)
        stampSize = Extent2D(*self.config.stampSize.list())
        stampRadius = floor(stampSize / 2)
        self.stampBBox = Box2I(corner=Point2I(0, 0), dimensions=Extent2I(1, 1)).dilatedBy(stampRadius)
        paddedStampSize = stampSize #* self.config.stampSizePadding
        self.paddedStampRadius = floor(paddedStampSize / 2)
        self.paddedStampBBox = Box2I(corner=Point2I(0, 0), dimensions=Extent2I(1, 1)).dilatedBy(
            self.paddedStampRadius
        )

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docstring inherited.
        inputs = butlerQC.get(inputRefs)
        dataId = butlerQC.quantum.dataId
        refObjLoader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefs.refCat],
            refCats=inputs.pop("refCat"),
            name=self.config.connections.refCat,
            config=self.config.refObjLoader,
        )
        # TODO: include the un-subtracted star here!
        subtractor = self.run(**inputs, dataId=dataId, refObjLoader=refObjLoader)
        if self.config.doWriteSubtractedExposure:
            outputExposure = inputs["inputExposure"].clone()
            outputExposure.image -= subtractor.image
        else:
            outputExposure = None
        outputBackgroundExposure = ExposureF(subtractor) if self.config.doWriteSubtractor else None
        output = Struct(
            outputExposure=outputExposure,
            outputBackgroundExposure=outputBackgroundExposure,
        )
        butlerQC.put(output, outputRefs)

    @timeMethod
    def run(
        self,
        inputExposure: ExposureF,
        inputCalexp: ExposureF,
        inputBackground: BackgroundList,
        # inputBrightStarStamps, #next plan is to use stamps for pedestal and gradients?
        inputExtendedPsf: ImageF,
        dataId: dict[str, Any] | DataCoordinate,
        # inputBackground: BackgroundList,
        refObjLoader: ReferenceObjectLoader,
    ):
        """Identify bright stars within an exposure using a reference catalog,
        extract stamps around each, warp/shift stamps onto a common frame and
        then optionally fit a PSF plus plane model.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The background-subtracted image to extract bright star stamps.
        inputBackground : `~lsst.afw.math.BackgroundList`
            The background model associated with the input exposure.
        refObjLoader : `~lsst.meas.algorithms.ReferenceObjectLoader`, optional
            Loader to find objects within a reference catalog.
        dataId : `dict` or `~lsst.daf.butler.DataCoordinate`
            The dataId of the exposure that bright stars are extracted from.
            Both 'visit' and 'detector' will be persisted in the output data.

        Returns
        -------
        brightStarResults : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``brightStarStamps``
                (`~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`)
        """
        wcs = inputCalexp.getWcs()
        bbox = inputCalexp.getBBox()
        warpingControl = WarpingControl(self.config.warpingKernelName, self.config.maskWarpingKernelName)

        refCatBright = self._getRefCatBright(refObjLoader, wcs, bbox)
        refCatBright.sort("mag")
        zipRaDec = zip(refCatBright["coord_ra"] * radians, refCatBright["coord_dec"] * radians)
        spherePoints = [SpherePoint(ra, dec) for ra, dec in zipRaDec]
        pixCoords = wcs.skyToPixel(spherePoints)

        inputFixed = inputCalexp.getMaskedImage()
        inputFixed += inputBackground.getImage()
        inputCalexp.mask.addMaskPlane(NEIGHBOR_MASK_PLANE)
        allFootprints, associations = self._associateFootprints(inputCalexp, pixCoords, plane="DETECTED")

        subtractorExp = ExposureF(bbox=bbox)
        templateSubtractor = subtractorExp.maskedImage

        detector = inputCalexp.detector
        pixelScale = wcs.getPixelScale().asArcseconds() * arcseconds
        pixToTan = detector.getTransform(PIXELS, TAN_PIXELS)
        pixToFocalPlaneTan = detector.getTransform(PIXELS, FIELD_ANGLE).then(
            makeTransform(AffineTransform.makeScaling(1 / pixelScale.asRadians()))
        )

        self.warpedDataDict = {}
        removalIndices = []
        for j in range(self.config.min_iterations):
            scaleList = []
            for starIndex, (obj, pixCoord) in enumerate(zip(refCatBright, pixCoords)):  # type: ignore
                inputMI = deepcopy(inputFixed)
                restSubtractor = deepcopy(templateSubtractor)
                myNumber = 0
                for key in self.warpedDataDict.keys():
                    if self.warpedDataDict[key]["subtractor"] is not None and key != obj['id']:
                        restSubtractor.image += self.warpedDataDict[key]["subtractor"].image
                        myNumber += 1
                self.log.debug(f"Number of stars subtracted before finding the scale factor for {obj['id']}: ", myNumber)
                inputMI.image -= restSubtractor.image
        
                footprintIndex = associations.get(starIndex, None)

                if footprintIndex:
                    neighborFootprints = [fp for i, fp in enumerate(allFootprints) if i != footprintIndex]
                    self._setFootprints(inputMI, neighborFootprints, NEIGHBOR_MASK_PLANE)
                else:
                    self._setFootprints(inputMI, allFootprints, NEIGHBOR_MASK_PLANE)
                # Define linear shifting to recenter stamps
                coordFocalPlaneTan = pixToFocalPlaneTan.applyForward(pixCoord)  # center of warped star
                shift = makeTransform(AffineTransform(Point2D(0, 0) - coordFocalPlaneTan))
                angle = np.arctan2(coordFocalPlaneTan.getY(), coordFocalPlaneTan.getX()) * radians
                rotation = makeTransform(AffineTransform.makeRotation(-angle))
                pixToPolar = pixToFocalPlaneTan.then(shift).then(rotation)
                rawStamp= self._getCutout(inputExposure=inputMI, coordPix=pixCoord, stampSize=self.config.stampSize.list())
                if rawStamp is None:
                    self.log.debug(f"No stamp for star with refID {obj['id']}")
                    removalIndices.append(starIndex)
                    continue
                warpedStamp = self._warpRawStamp(obj["id"], obj["mag"], rawStamp, warpingControl, pixToTan, pixCoord)
                warpedModel = ImageF(warpedStamp.getBBox())
                inputExtendedPsfGeneral = deepcopy(inputExtendedPsf)
                good_pixels = warpImage(warpedModel, inputExtendedPsfGeneral, pixToPolar.inverted(), warpingControl)
                self.warpedDataDict[obj["id"]] = {"stamp": warpedStamp, "model": warpedModel, "starIndex": starIndex, "pixCoord": pixCoord}
                if j == 0:
                    self.warpedDataDict[obj["id"]]["scale"] = None
                    self.warpedDataDict[obj["id"]]["subtractor"] = None
                fitPsfResults = {}
                fitPsfResults = self._fitPsf( warpedStamp, warpedModel)
                if fitPsfResults:
                    scaleList.append(fitPsfResults["scale"])
                    self.warpedDataDict[obj["id"]]["scale"] = fitPsfResults["scale"]


                    cond = np.isnan(warpedModel.array)
                    warpedModel.array[cond] = 0
                    warpedModel.array *= fitPsfResults["scale"]
                    overlapBBox = Box2I(warpedStamp.getBBox())
                    overlapBBox.clip(inputCalexp.getBBox())

                    subtractor = deepcopy(templateSubtractor)
                    subtractor[overlapBBox] += warpedModel[overlapBBox]
                    self.warpedDataDict[obj["id"]]["subtractor"] = subtractor


                else:
                    scaleList.append(np.nan)
            if j == 0:
                refCatBright.remove_rows(removalIndices)
                updatedPixCoords = [item for i, item in enumerate(pixCoords) if i not in removalIndices]
                pixCoords = updatedPixCoords
            new_scale_column = Column(scaleList, name=f'scale_0{j}')
            # The following is handy when developing, not sure if we want to do that in the final version!
            refCatBright.add_columns([new_scale_column])

        subtractor = deepcopy(templateSubtractor)
        for key in self.warpedDataDict.keys():
            if self.warpedDataDict[key]["scale"] is not None:
                subtractor.image.array += self.warpedDataDict[key]["subtractor"].image.array
        return subtractor

    def _getRefCatBright(self, refObjLoader: ReferenceObjectLoader, wcs: SkyWcs, bbox: Box2I) -> Table:
        """Get a bright star subset of the reference catalog.

        Trim the reference catalog to only those objects within the exposure
        bounding box dilated by half the bright star stamp size.
        This ensures all stars that overlap the exposure are included.

        Parameters
        ----------
        refObjLoader : `~lsst.meas.algorithms.ReferenceObjectLoader`
            Loader to find objects within a reference catalog.
        wcs : `~lsst.afw.geom.SkyWcs`
            World coordinate system.
        bbox : `~lsst.geom.Box2I`
            Bounding box of the exposure.

        Returns
        -------
        refCatBright : `~astropy.table.Table`
            Bright star subset of the reference catalog.
        """
        dilatedBBox = bbox.dilatedBy(self.paddedStampRadius)
        withinExposure = refObjLoader.loadPixelBox(dilatedBBox, wcs, filterName="phot_g_mean")
        refCatFull = withinExposure.refCat
        fluxField: str = withinExposure.fluxField

        brightFluxRange = sorted(((self.config.magRange * u.ABmag).to(u.nJy)).to_value())

        subsetStars = (refCatFull[fluxField] > brightFluxRange[0]) & (
            refCatFull[fluxField] < brightFluxRange[1]
        )
        refCatSubset = Table(refCatFull.extract("id", "coord_ra", "coord_dec", fluxField, where=subsetStars))
        fluxNanojansky = refCatSubset[fluxField][:] * u.nJy  # type: ignore
        refCatSubset["mag"] = fluxNanojansky.to(u.ABmag).to_value()  # AB magnitudes
        return refCatSubset

    def _associateFootprints(
        self, inputExposure: ExposureF, pixCoords: list[Point2D], plane: str
    ) -> tuple[list[Footprint], dict[int, int]]:
        """Associate footprints from a given mask plane with specific objects.

        Footprints from the given mask plane are associated with objects at the
        coordinates provided, where possible.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The input exposure with a mask plane.
        pixCoords : `list` [`~lsst.geom.Point2D`]
            The pixel coordinates of the objects.
        plane : `str`
            The mask plane used to identify masked pixels.

        Returns
        -------
        footprints : `list` [`~lsst.afw.detection.Footprint`]
            The footprints from the input exposure.
        associations : `dict`[int, int]
            Association indices between objects (key) and footprints (value).
        """
        detThreshold = Threshold(inputExposure.mask.getPlaneBitMask(plane), Threshold.BITMASK)
        footprintSet = FootprintSet(inputExposure.mask, detThreshold)
        footprints = footprintSet.getFootprints()
        associations = {}
        for starIndex, pixCoord in enumerate(pixCoords):
            for footprintIndex, footprint in enumerate(footprints):
                if footprint.contains(Point2I(pixCoord)):
                    associations[starIndex] = footprintIndex
                    break
        self.log.debug(
            "Associated %i of %i star%s to one each of the %i %s footprint%s.",
            len(associations),
            len(pixCoords),
            "" if len(pixCoords) == 1 else "s",
            len(footprints),
            plane,
            "" if len(footprints) == 1 else "s",
        )
        return footprints, associations

    def _setFootprints(self, inputExposure: ExposureF, footprints: list, maskPlane: str):
        """Set footprints in a given mask plane.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The input exposure to modify.
        footprints : `list` [`~lsst.afw.detection.Footprint`]
            The footprints to set in the mask plane.
        maskPlane : `str`
            The mask plane to set the footprints in.

        Notes
        -----
        This method modifies the ``inputExposure`` object in-place.
        """
        detThreshold = Threshold(inputExposure.mask.getPlaneBitMask(maskPlane), Threshold.BITMASK)
        detThresholdValue = int(detThreshold.getValue())
        footprintSet = FootprintSet(inputExposure.mask, detThreshold)

        # Wipe any existing footprints in the mask plane
        inputExposure.mask.clearMaskPlane(int(np.log2(detThresholdValue)))

        # Set the footprints in the mask plane
        footprintSet.setFootprints(footprints)
        footprintSet.setMask(inputExposure.mask, maskPlane)

    def _fitPsf(self, stampMI: MaskedImageF, psfImage: ImageD | ImageF) -> dict[str, Any]:
        """Fit a scaled PSF and a pedestal to each bright star cutout.

        Parameters
        ----------
        stampMI : `~lsst.afw.image.MaskedImageF`
            The masked image of the bright star cutout.
        psfImage : `~lsst.afw.image.ImageD` | `~lsst.afw.image.ImageF`
            The PSF model to fit.

        Returns
        -------
        fitPsfResults : `dict`[`str`, `float`]
            The result of the PSF fitting, with keys:

            ``scale`` : `float`
                The scale factor.
            ``scaleErr`` : `float`
                The error on the scale factor.
            ``pedestal`` : `float`
                The pedestal value.
            ``pedestalErr`` : `float`
                The error on the pedestal value.
            ``pedestalScaleCov`` : `float`
                The covariance between the pedestal and scale factor.
            ``xGradient`` : `float`
                The gradient in the x-direction.
            ``yGradient`` : `float`
                The gradient in the y-direction.
            ``globalReducedChiSquared`` : `float`
                The global reduced chi-squared goodness-of-fit.
            ``globalDegreesOfFreedom`` : `int`
                The global number of degrees of freedom.
            ``psfReducedChiSquared`` : `float`
                The PSF BBox reduced chi-squared goodness-of-fit.
            ``psfDegreesOfFreedom`` : `int`
                The PSF BBox number of degrees of freedom.
            ``psfMaskedFluxFrac`` : `float`
                The fraction of the PSF image flux masked by bad pixels.
        """
        badMaskBitMask = stampMI.mask.getPlaneBitMask(self.config.badMaskPlanes)

        # Calculate the fraction of the PSF image flux masked by bad pixels
        psfMaskedPixels = ImageF(psfImage.getBBox())
        psfMaskedPixels.array[:, :] = (stampMI.mask[psfImage.getBBox()].array & badMaskBitMask).astype(bool)
        # TODO: This is np.float64, else FITS metadata serialization fails
        # Amir: what do we want to do for subtraction? we do not have the luxury of removing the star from the process here!
        psfMaskedFluxFrac = np.dot(psfImage.array.flat, psfMaskedPixels.array.flat).astype(np.float64)
        # if psfMaskedFluxFrac > self.config.psfMaskedFluxFracThreshold:
        #     return {}  # Handle cases where the PSF image is mostly masked

        # Create a padded version of the input constant PSF image
        paddedPsfImage = ImageF(stampMI.getBBox())
        paddedPsfImage[psfImage.getBBox()] = psfImage.convertF()

        # Create consistently masked data
        mask = self.add_psf_mask(psfImage, stampMI)
        badSpans = SpanSet.fromMask(mask, badMaskBitMask)
        goodSpans = SpanSet(stampMI.getBBox()).intersectNot(badSpans)
        varianceData = goodSpans.flatten(stampMI.variance.array, stampMI.getXY0())
        if self.config.useMedianVariance:
            varianceData = np.median(varianceData)
        sigmaData = np.sqrt(varianceData)
        imageData = goodSpans.flatten(stampMI.image.array, stampMI.getXY0())  # B
        imageData /= sigmaData
        psfData = goodSpans.flatten(paddedPsfImage.array, paddedPsfImage.getXY0())
        psfData /= sigmaData

        # Fit the PSF scale factor and global pedestal
        nData = len(imageData)
        coefficientMatrix = np.ones((nData, 4), dtype=float)  # A
        coefficientMatrix[:, 0] = psfData
        coefficientMatrix[:, 1] /= sigmaData
        coefficientMatrix[:, 2:] = goodSpans.indices().T
        coefficientMatrix[:, 2] /= sigmaData
        coefficientMatrix[:, 3] /= sigmaData
        try:
            solutions, sumSquaredResiduals, *_ = np.linalg.lstsq(coefficientMatrix, imageData, rcond=None)
            covarianceMatrix = np.linalg.inv(np.dot(coefficientMatrix.transpose(), coefficientMatrix))  # C
        except np.linalg.LinAlgError:
            return {}  # Handle singular matrix errors
        if sumSquaredResiduals.size == 0:
            return {}
        scale = solutions[0]
        if scale <= 0:
            return {}  # Handle cases where the PSF scale fit has failed
        scaleErr = np.sqrt(covarianceMatrix[0, 0])
        pedestal = solutions[1]
        pedestalErr = np.sqrt(covarianceMatrix[1, 1])
        scalePedestalCov = covarianceMatrix[0, 1]
        xGradient = solutions[3]
        yGradient = solutions[2]

        # Calculate global (whole image) reduced chi-squared
        globalChiSquared = np.sum(sumSquaredResiduals)
        globalDegreesOfFreedom = nData - 4
        globalReducedChiSquared = globalChiSquared / globalDegreesOfFreedom

        # Calculate PSF BBox reduced chi-squared
        psfBBoxGoodSpans = goodSpans.clippedTo(psfImage.getBBox())
        psfBBoxGoodSpansX, psfBBoxGoodSpansY = psfBBoxGoodSpans.indices()
        psfBBoxData = psfBBoxGoodSpans.flatten(stampMI.image.array, stampMI.getXY0())
        psfBBoxModel = (
            psfBBoxGoodSpans.flatten(paddedPsfImage.array, stampMI.getXY0()) * scale
            + pedestal
            + psfBBoxGoodSpansX * xGradient
            + psfBBoxGoodSpansY * yGradient
        )
        psfBBoxVariance = psfBBoxGoodSpans.flatten(stampMI.variance.array, stampMI.getXY0())
        psfBBoxResiduals = (psfBBoxData - psfBBoxModel) ** 2 / psfBBoxVariance
        psfBBoxChiSquared = np.sum(psfBBoxResiduals)
        psfBBoxDegreesOfFreedom = len(psfBBoxData) - 4
        psfBBoxReducedChiSquared = psfBBoxChiSquared / psfBBoxDegreesOfFreedom

        return dict(
            scale=scale,
            scaleErr=scaleErr,
            pedestal=pedestal,
            pedestalErr=pedestalErr,
            xGradient=xGradient,
            yGradient=yGradient,
            pedestalScaleCov=scalePedestalCov,
            globalReducedChiSquared=globalReducedChiSquared,
            globalDegreesOfFreedom=globalDegreesOfFreedom,
            psfReducedChiSquared=psfBBoxReducedChiSquared,
            psfDegreesOfFreedom=psfBBoxDegreesOfFreedom,
            psfMaskedFluxFrac=psfMaskedFluxFrac,
        )

    def add_psf_mask(self, psfImage, stampMI):
        """Add psf frame mask into the stamp's mask.

        Args:
            psfImage (`~lsst.afw.image.ImageF`): PSF data
            stampMI (`~lsst.afw.image.MaskedImageF`): the stamp of the star that being fitted.

        Returns:
            `~lsst.afw.image.MaskX``: the mask frame containing the stamp plus the psf model mask.
        """
        cond = np.isnan(psfImage.array)
        cond |= psfImage.array < 0
        mask = deepcopy(stampMI.mask)
        mask.array[cond] = np.bitwise_or(mask.array[cond], 1)
        return mask


    def _getCutout(self, inputExposure, coordPix: Point2D, stampSize: list[int]):
        """Get a cutout from an input exposure, handling edge cases.

        Generate a cutout from an input exposure centered on a given position
        and with a given size.
        If any part of the cutout is outside the input exposure bounding box,
        the cutout is padded with NaNs.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The image to extract bright star stamps from.
        coordPix : `~lsst.geom.Point2D`
            Center of the cutout in pixel space.
        stampSize : `list` [`int`]
            Size of the cutout, in pixels.

        Returns
        -------
        stamp : `~lsst.afw.image.ExposureF` or `None`
            The cutout, or `None` if the cutout is entirely outside the input
            exposure bounding box.

        Notes
        -----
        This method is a short-term workaround until DM-40042 is implemented.
        At that point, it should be replaced by a call to the Exposure method
        ``getCutout``, which will handle edge cases automatically.
        """
        corner = Point2I(np.array(coordPix) - np.array(stampSize) / 2)
        dimensions = Extent2I(stampSize)
        stampBBox = Box2I(corner, dimensions)
        overlapBBox = Box2I(stampBBox)
        overlapBBox.clip(inputExposure.getBBox())
        if overlapBBox.getArea() > 0:
            # Create full-sized stamp with pixels initially flagged as NO_DATA.
            stamp = ExposureF(bbox=stampBBox)
            stamp.image[:] = np.nan
            stamp.mask.set(inputExposure.mask.getPlaneBitMask("NO_DATA"))
            # # Restore pixels which overlap the input exposure.
            overlap = inputExposure.Factory(inputExposure, overlapBBox)
            stamp.maskedImage[overlapBBox] = overlap
        else:
            stamp = None
        return stamp
    
    def _warpRawStamp(self, obj, mag, rawStamp, warpingControl, pixToTan, pixCoord):
        destImage = MaskedImageF(*self.config.stampSize)
        bottomLeft = Point2D(rawStamp.getXY0())
        newBottomLeft = pixToTan.applyForward(bottomLeft)
        newBottomLeft = Point2I(newBottomLeft)
        destImage.setXY0(newBottomLeft)
        # Define linear shifting to recenter stamps
        newCenter = pixToTan.applyForward(pixCoord)
        self.modelCenter = self.config.stampSize[0] // 2, self.config.stampSize[1] // 2
        shift = (self.modelCenter[0] + newBottomLeft[0] - newCenter[0], self.modelCenter[1] + newBottomLeft[1] - newCenter[1])
        affineShift = AffineTransform(shift)
        shiftTransform = makeTransform(affineShift)

        # Define full transform (warp and shift)
        starWarper = pixToTan.then(shiftTransform)

        # Apply it
        goodPix = warpImage(destImage, rawStamp.getMaskedImage(), starWarper, warpingControl)
        if not goodPix:
            return None
        return destImage

        # # Arbitrarily set origin of shifted star to 0
        # destImage.setXY0(0, 0)