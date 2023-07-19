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

__all__ = ["SubtractBrightStarsConnections", "SubtractBrightStarsConfig", "SubtractBrightStarsTask"]

import logging
from functools import reduce
from operator import ior

import numpy as np
from lsst.afw.image import Exposure, ExposureF, MaskedImageF
from lsst.afw.geom import SpanSet, Stencil
from lsst.afw.math import (
    StatisticsControl,
    WarpingControl,
    makeStatistics,
    rotateImageBy90,
    stringToStatisticsProperty,
    warpImage,
)
from lsst.geom import Box2I, Point2D, Point2I
from lsst.meas.algorithms import LoadReferenceObjectsConfig, ReferenceObjectLoader
from lsst.meas.algorithms.brightStarStamps import BrightStarStamp, BrightStarStamps
from lsst.pex.config import ChoiceField, ConfigField, Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output, PrerequisiteInput
from lsst.pipe.tasks.processBrightStars import ProcessBrightStarsTask

logger = logging.getLogger(__name__)


class SubtractBrightStarsConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
    defaultTemplates={"outputExposureName": "brightStar_subtracted",
                      "outputBackgroundName": "brightStars",
                      "badStampsName": "brightStars"},
):
    inputExposure = Input(
        doc="Input exposure from which to subtract bright star stamps.",
        name="calexp",
        storageClass="ExposureF",
        dimensions=(
            "visit",
            "detector",
        ),
    )
    inputBrightStarStamps = Input(
        doc="Set of preprocessed postage stamps, each centered on a single bright star.",
        name="brightStarStamps",
        storageClass="BrightStarStamps",
        dimensions=(
            "visit",
            "detector",
        ),
    )
    inputExtendedPsf = Input(
        doc="Extended PSF model.",
        name="extended_psf",
        storageClass="ExtendedPsf",
        dimensions=("band",),
    )
    skyCorr = Input(
        doc="Input Sky Correction to be subtracted from the calexp if ``doApplySkyCorr``=True.",
        name="skyCorr",
        storageClass="Background",
        dimensions=(
            "instrument",
            "visit",
            "detector",
        ),
    )
    refCat = PrerequisiteInput(
        doc="Reference catalog that contains bright star positions",
        name="gaia_dr2_20200414",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        multiple=True,
        deferLoad=True,
    )
    outputExposure = Output(
        doc="Exposure with bright stars subtracted.",
        name="{outputExposureName}_calexp",
        storageClass="ExposureF",
        dimensions=(
            "visit",
            "detector",
        ),
    )
    outputBackgroundExposure = Output(
        doc="Exposure containing only the modelled bright stars.",
        name="{outputBackgroundName}_calexp_background",
        storageClass="ExposureF",
        dimensions=(
            "visit",
            "detector",
        ),
    )
    outputBadStamps = Output(
        doc="The stamps that are not normalized and consequently not subtracted from the exposure.",
        name="{badStampsName}_unsubtracted_stapms",
        storageClass="BrightStarStamps",
        dimensions=(
            "visit",
            "detector",
        ),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.doApplySkyCorr:
            self.inputs.remove("skyCorr")


class SubtractBrightStarsConfig(PipelineTaskConfig, pipelineConnections=SubtractBrightStarsConnections):
    """Configuration parameters for SubtractBrightStarsTask"""

    doWriteSubtractor = Field[bool](
        dtype=bool,
        doc="Should an exposure containing all bright star models be written to disk?",
        default=True,
    )
    doWriteSubtractedExposure = Field[bool](
        dtype=bool,
        doc="Should an exposure with bright stars subtracted be written to disk?",
        default=True,
    )
    magLimit = Field[float](
        dtype=float,
        doc="Magnitude limit, in Gaia G; all stars brighter than this value will be subtracted",
        default=18,
    )
    minValidAnnulusFraction = Field(
        dtype=float,
        doc="Minimum number of valid pixels that must fall within the annulus for the bright star to be "
        "saved for subsequent generation of a PSF.",
        default=0.0,
    )
    numSigmaClip = Field(
        dtype=float,
        doc="Sigma for outlier rejection; ignored if annularFluxStatistic != 'MEANCLIP'.",
        default=4,
    )
    numIter = Field(
        dtype=int,
        doc="Number of iterations of outlier rejection; ignored if annularFluxStatistic != 'MEANCLIP'.",
        default=3,
    )
    warpingKernelName = ChoiceField[str](
        dtype=str,
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
        dtype=str,
        doc="How the model should be scaled to each bright star; implemented options are "
        "`annularFlux` to reuse the annular flux of each stamp, or `leastSquare` to perform "
        "least square fitting on each pixel with no bad mask plane set.",
        default="leastSquare",
        allowed={
            "annularFlux": "reuse BrightStarStamp annular flux measurement",
            "leastSquare": "find least square scaling factor",
        },
    )
    annularFluxStatistic = ChoiceField(
        dtype=str,
        doc="Type of statistic to use to compute annular flux.",
        default="MEANCLIP",
        allowed={
            "MEAN": "mean",
            "MEDIAN": "median",
            "MEANCLIP": "clipped mean",
        },
    )
    badMaskPlanes = ListField[str](
        dtype=str,
        doc="Mask planes that, if set, lead to associated pixels not being included in the computation of "
        "the scaling factor (`BAD` should always be included). Ignored if scalingType is `annularFlux`, "
        "as the stamps are expected to already be normalized.",
        # Note that `BAD` should always be included, as secondary detected
        # sources (i.e., detected sources other than the primary source of
        # interest) also get set to `BAD`.
        default=("BAD", "CR", "CROSSTALK", "EDGE", "NO_DATA", "SAT", "SUSPECT", "UNMASKEDNAN"),
    )
    subtractionBox = ListField(
        dtype=int,
        doc="Size of the stamps to be extracted, in pixels.",
        default=(250, 250),
    )
    subtractionBoxBuffer = Field(
        dtype=float,
        doc=(
            "'Buffer' (multiplicative) factor to be applied to determine the size of the stamp the "
            "processed stars will be saved in. This is also the size of the extended PSF model. The buffer "
            "region is masked and contain no data and subtractionBox determines the region where contains "
            "the data."
        ),
        default=1.1,
    )
    doApplySkyCorr = Field[bool](
        dtype=bool,
        doc="Apply full focal plane sky correction before extracting stars?",
        default=True,
    )
    refObjLoader = ConfigField(
        dtype=LoadReferenceObjectsConfig,
        doc="Reference object loader for astrometric calibration.",
    )


class SubtractBrightStarsTask(PipelineTask):
    """Use an extended PSF model to subtract bright stars from a calibrated
    exposure (i.e. at single-visit level).

    This task uses both a set of bright star stamps produced by
    `~lsst.pipe.tasks.processBrightStars.ProcessBrightStarsTask`
    and an extended PSF model produced by
    `~lsst.pipe.tasks.extended_psf.MeasureExtendedPsfTask`.
    """

    ConfigClass = SubtractBrightStarsConfig
    _DefaultName = "subtractBrightStars"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Placeholders to set up Statistics if scalingType is leastSquare.
        self.statsControl, self.statsFlag = None, None
        # warping control; only contains shiftingALg provided in config
        self.warpControl = WarpingControl(self.config.warpingKernelName)

    def _setUpStatistics(self, exampleMask):
        """Configure statistics control and flag, for use if ``scalingType`` is
        `leastSquare`.
        """
        if self.config.scalingType == "leastSquare":
            self.statsControl = StatisticsControl()
            # Set the mask planes which will be ignored.
            andMask = reduce(ior, (exampleMask.getPlaneBitMask(bm) for bm in self.config.badMaskPlanes))
            self.statsControl.setAndMask(andMask)
            self.statsFlag = stringToStatisticsProperty("SUM")

    def applySkyCorr(self, calexp, skyCorr):
        """Apply correction to the sky background level.
        Sky corrections can be generated via the SkyCorrectionTask within the
        pipe_tools module. Because the sky model used by that code extends over
        the entire focal plane, this can produce better sky subtraction.
        The calexp is updated in-place.

        Parameters
        ----------
        calexp : `~lsst.afw.image.Exposure` or `~lsst.afw.image.MaskedImage`
            Calibrated exposure.
        skyCorr : `~lsst.afw.math.backgroundList.BackgroundList`
            Full focal plane sky correction, obtained by running
            `~lsst.pipe.tasks.skyCorrection.SkyCorrectionTask`.
        """
        if isinstance(calexp, Exposure):
            calexp = calexp.getMaskedImage()
        calexp -= skyCorr.getImage()

    def scaleModel(self, model, star, inPlace=True, nb90Rots=0, psf_annular_flux=1.):
        """Compute scaling factor to be applied to the extended PSF so that its
        amplitude matches that of an individual star.

        Parameters
        ----------
        model : `~lsst.afw.image.MaskedImageF`
            The extended PSF model, shifted (and potentially warped) to match
            the bright star's positioning.
        star : `~lsst.meas.algorithms.brightStarStamps.BrightStarStamp`
            A stamp centered on the bright star to be subtracted.
        inPlace : `bool`
            Whether the model should be scaled in place. Default is `True`.
        nb90Rots : `int`
            The number of 90-degrees rotations to apply to the star stamp.
        psf_annular_flux: `float`, optional
            The annular flux of the PSF model at the radius where the flux of the given star is determined.
            This is 1 for stars present in inputBrightStarStamps, but can be different for stars that are
            missing from inputBrightStarStamps.

        Returns
        -------
        scalingFactor : `float`
            The factor by which the model image should be multiplied for it
            to be scaled to the input bright star.
        """
        if self.config.scalingType == "annularFlux":
            scalingFactor = star.annularFlux * psf_annular_flux
        elif self.config.scalingType == "leastSquare":
            if self.statsControl is None:
                self._setUpStatistics(star.stamp_im.mask)
            starIm = star.stamp_im.clone()
            # Rotate the star postage stamp.
            starIm = rotateImageBy90(starIm, nb90Rots)
            # Reverse the prior star flux normalization ("unnormalize").
            starIm *= star.annularFlux
            # The estimator of the scalingFactor (f) that minimizes (Y-fX)^2
            # is E[XY]/E[XX].
            xy = starIm.clone()
            xy.image.array *= model.image.array
            xx = starIm.clone()
            xx.image.array = model.image.array**2
            # Compute the least squares scaling factor.
            xySum = makeStatistics(xy, self.statsFlag, self.statsControl).getValue()
            xxSum = makeStatistics(xx, self.statsFlag, self.statsControl).getValue()
            scalingFactor = xySum / xxSum if xxSum else 1
        if inPlace:
            model.image *= scalingFactor
        return scalingFactor

    def _overrideWarperConfig(self):
        """Override the warper config with the config of this task. This is necessary for the stars that are
        missing from the inputBrightStarStamps but need to be subtracted.
        """
        self.warper.config.minValidAnnulusFraction = self.config.minValidAnnulusFraction
        self.warper.config.numSigmaClip = self.config.numSigmaClip
        self.warper.config.numIter = self.config.numIter
        self.warper.config.annularFluxStatistic = self.config.annularFluxStatistic
        self.warper.config.badMaskPlanes = self.config.badMaskPlanes
        self.warper.config.stampSize = self.config.subtractionBox
        self.warper.modelStampBuffer = self.config.subtractionBoxBuffer
        self.warper.setModelStamp()

    def setMissedStarsStatsControl(self):
        """Configure statistics control for processing missing stars from inputBrightStarStamps.
        """
        self.missedStatsControl = StatisticsControl()
        self.missedStatsControl.setNumSigmaClip(self.warper.config.numSigmaClip)
        self.missedStatsControl.setNumIter(self.warper.config.numIter)
        self.missedStatsFlag = stringToStatisticsProperty(self.warper.config.annularFluxStatistic)

    def setWarpTask(self):
        """Create an instance of ProcessBrightStarsTask that will be used to produce stamps of stars to be
        subtracted.
        """
        self.warper = ProcessBrightStarsTask()
        self._overrideWarperConfig()
        self.warper.modelCenter = self.modelStampSize[0] // 2, self.modelStampSize[1] // 2

    def makeBrightStarList(self, inputBrightStarStamps, inputExposure, refObjLoader):
        """Make a list of bright stars that are missing from inputBrightStarStamps to be subtracted.

        Parameters
        ----------
        inputBrightStarStamps : `~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`
            Set of stamps centered on each bright star to be subtracted, produced by running
            `~lsst.pipe.tasks.processBrightStars.ProcessBrightStarsTask`.
        inputExposure : `~lsst.afw.image.ExposureF`
            The image from which bright stars should be subtracted.
        refObjLoader : `~lsst.meas.algorithms.ReferenceObjectLoader`, optional
            Loader to find objects within a reference catalog.

        Returns
        -------
        brightStarList:
            A list containing `lsst.meas.algorithms.brightStarStamps.BrightStarStamp` of stars to be
            subtracted.
        """
        self.setWarpTask()
        missedStars = self.warper.extractStamps(
            inputExposure, refObjLoader=refObjLoader, inputBrightStarStamps=inputBrightStarStamps
        )
        self.warpOutputs = self.warper.warpStamps(missedStars.starIms, missedStars.pixCenters)
        brightStarList = [
            BrightStarStamp(
                stamp_im=warp,
                archive_element=transform,
                position=self.warpOutputs.xy0s[j],
                gaiaGMag=missedStars.GMags[j],
                gaiaId=missedStars.gaiaIds[j],
                minValidAnnulusFraction=self.warper.config.minValidAnnulusFraction,
            )
            for j, (warp, transform) in enumerate(zip(self.warpOutputs.warpedStars,
                                                      self.warpOutputs.warpTransforms))
        ]
        return brightStarList

    def initAnnulusImage(self):
        """Initialize an annulus image of the given star.

        Returns
        -------
        annulusImage : `~lsst.afw.image.MaskedImageF`
            The initialized annulus image.
        """
        maskPlaneDict = self.model.mask.getMaskPlaneDict()
        annulusImage = MaskedImageF(self.modelStampSize, planeDict=maskPlaneDict)
        annulusImage.mask.array[:] = 2 ** maskPlaneDict["NO_DATA"]
        return annulusImage

    def createAnnulus(self, brightStarStamp):
        """Create a circular annulus of the given star. The annulus is set based on the inner and outer
        optimal radii. The optimal radii describe the annulus where the flux of the star is found.
        Accordingly, the aim is to crate the same annulus for the PSF model and eventually measure the modle
        flux around that annulus. An optimal radius usually differes from the radius where the PSF model is
        normalized.

        Parameters
        ----------
        brightStarStamp : `~lsst.meas.algorithms.brightStarStamps.BrightStarStamp`
            A stamp of a bright star to be subtracted.

        Returns
        -------
        annulus : `~lsst.afw.image.MaskedImageF`
            An annulus of the given star.
        """
        # Create SpanSet of annulus
        outerCircle = SpanSet.fromShape(
            brightStarStamp.optimalOuterRadius, Stencil.CIRCLE, offset=self.warper.modelCenter
        )
        innerCircle = SpanSet.fromShape(
            brightStarStamp.optimalInnerRadius, Stencil.CIRCLE, offset=self.warper.modelCenter
        )
        annulus = outerCircle.intersectNot(innerCircle)
        return annulus

    def applyStatsControl(self, annulusImage):
        """Apply statistics control to the PSF annulus image.

        Parameters
        ----------
        annulusImage : `~lsst.afw.image.MaskedImageF`
            An image containing an annulus of the given model.

        Returns
        -------
        annularFlux: float
            The annular flux of the PSF model at the radius where the flux of the given star is determined.
        """
        andMask = reduce(
            ior, (annulusImage.mask.getPlaneBitMask(bm) for bm in self.warper.config.badMaskPlanes)
        )
        self.missedStatsControl.setAndMask(andMask)
        annulusStat = makeStatistics(annulusImage, self.missedStatsFlag, self.missedStatsControl)
        return annulusStat.getValue()

    def findPsfAnnularFlux(self, brightStarStamp, maskedModel):
        """Find the annular flux of the PSF model at the radius where the flux of the given star is
        determined.

        Parameters
        ----------
        brightStarStamp : `~lsst.meas.algorithms.brightStarStamps.BrightStarStamp`
            A stamp of a bright star to be subtracted.
        maskedModel : `~lsst.afw.image.MaskedImageF`
            A masked image of the PSF model.

        Returns
        -------
        annularFlux: float (between 0 and 1)
            The annular flux of the PSF model at the radius where the flux of the given star is determined.
        """
        annulusImage = self.initAnnulusImage()
        annulus = self.createAnnulus(brightStarStamp)
        annulus.copyMaskedImage(maskedModel, annulusImage)
        annularFlux = self.applyStatsControl(annulusImage)
        return annularFlux

    def findPsfAnnularFluxes(self, brightStarStamps):
        """Find the annular fluxes of the given PSF model.

        Parameters
        ----------
        brightStarStamps : `~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`
            The stamps of stars that will be subtracted from the exposure.

        Returns
        -------
        PsfAnnularFluxes: numpy.array
            A two column numpy.array containing annular fluxes of the PSF at radii where the flux for stars
            exist (could be found).

        Notes
        -----
        While the PSF model is normalized at a certain radius, the annular flux of a star around that radius
        might be impossible to find. Therefore, we have to scale the PSF model considering a radius where the
        star has an identified flux. To do that, the flux of the model should be found and used to adjust the
        scaling step.
        """
        outerRadii = []
        annularFluxes = []
        maskedModel = MaskedImageF(self.model.image)
        # the model has wrong bbox values. Should be fixed in extended_psf.py?
        maskedModel.setXY0(0, 0)
        for star in brightStarStamps:
            if star.optimalOuterRadius not in outerRadii:
                annularFlux = self.findPsfAnnularFlux(star, maskedModel)
                outerRadii.append(star.optimalOuterRadius)
                annularFluxes.append(annularFlux)
        return np.array([outerRadii, annularFluxes]).T

    def preparePlaneModelStamp(self, brightStarStamp):
        """Prepare the PSF plane model. It is called PlaneMdoel because while it is a PSF model stamp that is
        warped and rotated to the same orientation of a chosen star, but it is not scaled to the brightness
        level of the star yet.

        Parameters
        ----------
        brightStarStamp : `~lsst.meas.algorithms.brightStarStamps.BrightStarStamp`
            The stamp of the star to which the PSF model will be scaled.

        Returns
        -------
        bbox: `~lsst.geom.Box2I`
            Contains the corner coordination and the dimensions of the model stamp.

        invImage: `~lsst.afw.image.MaskedImageF`
            The extended PSF model, shifted (and potentially warped and rotated) to match the bright star's
            positioning.

        Raises
        ------
        RuntimeError
            Raised if warping of the model failed.

        Notes
        -----
        Since detectors have different orientations, the PSF model should be rotated to match the orientation
        of the detectors in some cases. To do that, the code uses the inverse of the transform that is
        applied to the bright star stamp to match the orientation of the detector.
        """
        # Set the origin.
        self.model.setXY0(brightStarStamp.position)
        # Create an empty destination image.
        invTransform = brightStarStamp.archive_element.inverted()
        invOrigin = Point2I(invTransform.applyForward(Point2D(brightStarStamp.position)))
        bbox = Box2I(corner=invOrigin, dimensions=self.modelStampSize)
        invImage = MaskedImageF(bbox)
        # Apply inverse transform.
        goodPix = warpImage(invImage, self.model, invTransform, self.warpControl)
        if not goodPix:
            # Do we want to find another way or just subtract the non-warped scaled model?
            # Currently the code just leaves the failed ones un-subtracted.
            raise RuntimeError(
                f"Warping of a model failed for star {brightStarStamp.gaiaId}: " "no good pixel in output"
            )
        return bbox, invImage

    def addScaledModel(self, subtractor, brightStarStamp, multipleAnnuli=False):
        """Add the scaled model of the given star to the subtractor plane.

        Parameters
        ----------
        subtractor : `~lsst.afw.image.MaskedImageF`
            The full image containing the scaled model of bright stars to be subtracted from the input
            exposure.
        brightStarStamp : `~lsst.meas.algorithms.brightStarStamps.BrightStarStamp`
            The stamp of the star of which the PSF model will be scaled and added to the subtractor.
        multipleAnnuli : bool, optional
            If true, the model should be scaled based on a flux at a radius other than its normalization
            radius.

        Returns
        -------
        subtractor : `~lsst.afw.image.MaskedImageF`
            The input subtractor full image with the added scaled model at the given star's location in the
            exposure.
        invImage: `~lsst.afw.image.MaskedImageF`
            The extended PSF model, shifted (and potentially warped) to match the bright star's positioning.
        """
        bbox, invImage = self.preparePlaneModelStamp(brightStarStamp)
        bbox.clip(self.inputExpBBox)
        if bbox.getArea() > 0:
            if multipleAnnuli:
                cond = self.psf_annular_fluxes[:, 0] == brightStarStamp.optimalOuterRadius
                psf_annular_flux = self.psf_annular_fluxes[cond, 1][0]
                self.scaleModel(invImage,
                                brightStarStamp,
                                inPlace=True,
                                nb90Rots=self.inv90Rots,
                                psf_annular_flux=psf_annular_flux)
            else:
                self.scaleModel(invImage, brightStarStamp, inPlace=True, nb90Rots=self.inv90Rots)
            # Replace NaNs before subtraction (note all NaN pixels have
            # the NO_DATA flag).
            invImage.image.array[np.isnan(invImage.image.array)] = 0
            subtractor[bbox] += invImage[bbox]
        return subtractor, invImage

    def buildSubtractor(self, brightStarStamps, subtractor, invImages, multipleAnnuli=False):
        """Build an image containing potentially multiple scaled PSF models, each at the location of a given
        bright star.

        Parameters
        ----------
        brightStarStamps : `~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`
            Set of stamps centered on each bright star to be subtracted, produced by running
            `~lsst.pipe.tasks.processBrightStars.ProcessBrightStarsTask`.
        subtractor : `~lsst.afw.image.MaskedImageF`
            The Exposure that will contain the scaled model of bright stars to be subtracted from the
            exposure.
        invImages : `list`
            A list containing extended PSF models, shifted (and potentially warped) to match the bright stars
            positions.
        multipleAnnuli : bool, optional
            This will be passed to addScaledModel method, by default False.

        Returns
        -------
        subtractor : `~lsst.afw.image.MaskedImageF`
            An Exposure containing a scaled bright star model fit to every bright star profile; its image can
            then be subtracted from the input exposure.
        invImages: list
            A list containing the extended PSF models, shifted (and potentially warped) to match bright
            stars' positions.
        """
        for star in brightStarStamps:
            if star.gaiaGMag < self.config.magLimit:
                try:
                    # Adding the scaled model at the star location to the subtractor.
                    subtractor, invImage = self.addScaledModel(subtractor, star, multipleAnnuli)
                    invImages.append(invImage)
                except RuntimeError as err:
                    logger.error(err)
        return subtractor, invImages

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
        subtractor, _, badStamps = self.run(**inputs, dataId=dataId, refObjLoader=refObjLoader)
        if self.config.doWriteSubtractedExposure:
            outputExposure = inputs["inputExposure"].clone()
            outputExposure.image -= subtractor.image
        else:
            outputExposure = None
        outputBackgroundExposure = subtractor if self.config.doWriteSubtractor else None
        # in its current state, the code produces outputBadStamps which are the stamps of stars that have not
        # been subtracted from the image for any reason. If all the stars are subtracted from the calexp, the
        # output is an empty fits file.
        output = Struct(outputExposure=outputExposure,
                        outputBackgroundExposure=outputBackgroundExposure,
                        outputBadStamps=badStamps)
        butlerQC.put(output, outputRefs)

    def run(self,
            inputExposure,
            inputBrightStarStamps,
            inputExtendedPsf,
            dataId,
            skyCorr=None,
            refObjLoader=None):
        """Iterate over all bright stars in an exposure to scale the extended
        PSF model before subtracting bright stars.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The image from which bright stars should be subtracted.
        inputBrightStarStamps :
                `~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`
            Set of stamps centered on each bright star to be subtracted,
            produced by running
            `~lsst.pipe.tasks.processBrightStars.ProcessBrightStarsTask`.
        inputExtendedPsf : `~lsst.pipe.tasks.extended_psf.ExtendedPsf`
            Extended PSF model, produced by
            `~lsst.pipe.tasks.extended_psf.MeasureExtendedPsfTask`.
        dataId : `dict` or `~lsst.daf.butler.DataCoordinate`
            The dataId of the exposure (and detector) bright stars should be
            subtracted from.
        skyCorr : `~lsst.afw.math.backgroundList.BackgroundList`, optional
            Full focal plane sky correction, obtained by running
            `~lsst.pipe.tasks.skyCorrection.SkyCorrectionTask`. If
            `doApplySkyCorr` is set to `True`, `skyCorr` cannot be `None`.
        refObjLoader : `~lsst.meas.algorithms.ReferenceObjectLoader`, optional
            Loader to find objects within a reference catalog.

        Returns
        -------
        subtractorExp : `~lsst.afw.image.ExposureF`
            An Exposure containing a scaled bright star model fit to every
            bright star profile; its image can then be subtracted from the
            input exposure.
        invImages : `list` [`~lsst.afw.image.MaskedImageF`]
            A list of small images ("stamps") containing the model, each scaled
            to its corresponding input bright star.
        """
        self.inputExpBBox = inputExposure.getBBox()
        if self.config.doApplySkyCorr and (skyCorr is not None):
            self.log.info(
                "Applying sky correction to exposure %s (exposure will be modified in-place).", dataId
            )
            self.applySkyCorr(inputExposure, skyCorr)
        # Create an empty image the size of the exposure.
        # TODO: DM-31085 (set mask planes).
        subtractorExp = ExposureF(bbox=inputExposure.getBBox())
        subtractor = subtractorExp.maskedImage
        # Make a copy of the input model.
        self.model = inputExtendedPsf(dataId["detector"]).clone()
        self.modelStampSize = self.model.getDimensions()
        self.inv90Rots = 4 - inputBrightStarStamps.nb90Rots % 4
        self.model = rotateImageBy90(self.model, self.inv90Rots)

        brightStarList = self.makeBrightStarList(inputBrightStarStamps, inputExposure, refObjLoader)
        self.setMissedStarsStatsControl()
        # This might change when we use multiple categories of stars for creating PSF.
        innerRadius = inputBrightStarStamps._innerRadius
        outerRadius = inputBrightStarStamps._outerRadius
        brightStarStamps, badStamps = BrightStarStamps.initAndNormalize(
            brightStarList,
            innerRadius=innerRadius,
            outerRadius=outerRadius,
            nb90Rots=self.warpOutputs.nb90Rots,
            imCenter=self.warper.modelCenter,
            use_archive=True,
            statsControl=self.missedStatsControl,
            statsFlag=self.missedStatsFlag,
            badMaskPlanes=self.warper.config.badMaskPlanes,
            discardNanFluxObjects=False,
            forceFindFlux=True,
        )

        invImages = []
        subtractor, invImages = self.buildSubtractor(
            inputBrightStarStamps, subtractor, invImages, multipleAnnuli=False
        )
        if len(brightStarStamps) > 0:
            self.psf_annular_fluxes = self.findPsfAnnularFluxes(brightStarStamps)
            subtractor, invImages = self.buildSubtractor(
                brightStarStamps, subtractor, invImages, multipleAnnuli=True
            )
        badStamps = BrightStarStamps(badStamps)

        return subtractorExp, invImages, badStamps
