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
#
"""Retrieve extended PSF model and subtract bright stars at calexp (ie
single visit) level.
"""

__all__ = ["SubtractBrightStarsTask"]

import numpy as np
from operator import ior
from functools import reduce

from lsst.pipe import base as pipeBase
from lsst.pipe.base import connectionTypes as cT
from lsst.pex import config as pexConfig
from lsst.afw import math as afwMath
from lsst.afw import image as afwImage
from lsst import geom


class SubtractBrightStarsConnections(pipeBase.PipelineTaskConnections,
                                     dimensions=("instrument", "visit", "detector"),
                                     defaultTemplates={"outputExposureName": "brightStar_subtracted",
                                                       "outputBackgroundName": "brightStars"}):
    inputExposure = cT.Input(
        doc="Input exposure from which to subtract bright star stamps.",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("visit", "detector",),
    )
    inputBrightStarStamps = cT.Input(
        doc="Set of preprocessed postage stamps, each centered on a single bright star.",
        name="brightStarStamps",
        storageClass="BrightStarStamps",
        dimensions=("visit", "detector",),
    )
    inputExtendedPsf = cT.Input(
        doc="Extended PSF model.",
        name="extended_psf",
        storageClass="ExtendedPsf",
        dimensions=("band",),
    )
    skyCorr = cT.Input(
        doc="Input Sky Correction to be subtracted from the calexp if ``doApplySkyCorr``=True.",
        name="skyCorr",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector",),
    )
    outputExposure = cT.Output(
        doc="Exposure with bright stars subtracted.",
        name="{outputExposureName}_calexp",
        storageClass="ExposureF",
        dimensions=("visit", "detector",),
    )
    outputBackgroundExposure = cT.Output(
        doc="Exposure containing only the modelled bright stars.",
        name="{outputBackgroundName}_calexp_background",
        storageClass="ExposureF",
        dimensions=("visit", "detector",),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.doApplySkyCorr:
            self.inputs.remove("skyCorr")


class SubtractBrightStarsConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=SubtractBrightStarsConnections):
    """Configuration parameters for SubtractBrightStarsTask
    """
    doWriteSubtractor = pexConfig.Field(
        dtype=bool,
        doc="Should an exposure containing all bright star models be written to disk?",
        default=True
    )
    doWriteSubtractedExposure = pexConfig.Field(
        dtype=bool,
        doc="Should an exposure with bright stars subtracted be written to disk?",
        default=True
    )
    magLimit = pexConfig.Field(
        dtype=float,
        doc="Magnitude limit, in Gaia G; all stars brighter than this value will be subtracted",
        default=18
    )
    warpingKernelName = pexConfig.ChoiceField(
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
        }
    )
    scalingType = pexConfig.ChoiceField(
        dtype=str,
        doc="How the model should be scaled to each bright star; implemented options are "
            "`annularFlux` to reuse the annular flux of each stamp, or `leastSquare` to perform "
            "least square fitting on each pixel with no bad mask plane set.",
        default="leastSquare",
        allowed={
            "annularFlux": "reuse BrightStarStamp annular flux measurement",
            "leastSquare": "find least square scaling factor",
        }
    )
    badMaskPlanes = pexConfig.ListField(
        dtype=str,
        doc="Mask planes that, if set, lead to associated pixels not being included in the computation of "
            "the scaling factor (`BAD` should always be included). Ignored if scalingType is `annularFlux`, "
            "as the stamps are expected to already be normalized.",
            # Note that `BAD` should always be included, as secondary detected
            # sources (i.e., detected sources other than the primary source of
            # interest) also get set to `BAD`.
        default=('BAD', 'CR', 'CROSSTALK', 'EDGE', 'NO_DATA', 'SAT', 'SUSPECT', 'UNMASKEDNAN')
    )
    doApplySkyCorr = pexConfig.Field(
        dtype=bool,
        doc="Apply full focal plane sky correction before extracting stars?",
        default=True
    )


class SubtractBrightStarsTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """Use an extended PSF model to subtract bright stars from a calibrated
    exposure (i.e. at single-visit level).

    This task uses both a set of bright star stamps produced by
    `~lsst.pipe.tasks.processBrightStars.ProcessBrightStarsTask`
    and an extended PSF model produced by
    `~lsst.pipe.tasks.extended_psf.MeasureExtendedPsfTask`.
    """
    ConfigClass = SubtractBrightStarsConfig
    _DefaultName = "subtractBrightStars"

    def __init__(self, initInputs=None, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        # Placeholders to set up Statistics if scalingType is leastSquare.
        self.statsControl, self.statsFlag = None, None

    def _setUpStatistics(self, exampleMask):
        """Configure statistics control and flag, for use if ``scalingType`` is
        `leastSquare`.
        """
        if self.config.scalingType == "leastSquare":
            self.statsControl = afwMath.StatisticsControl()
            # Set the mask planes which will be ignored.
            andMask = reduce(ior, (exampleMask.getPlaneBitMask(bm) for bm in self.config.badMaskPlanes))
            self.statsControl.setAndMask(andMask)
            self.statsFlag = afwMath.stringToStatisticsProperty("SUM")

    def applySkyCorr(self, calexp, skyCorr):
        """Apply correction to the sky background level.
        Sky corrections can be generated with the 'skyCorrection.py'
        executable in pipe_drivers. Because the sky model used by that
        code extends over the entire focal plane, this can produce
        better sky subtraction.
        The calexp is updated in-place.

        Parameters
        ----------
        calexp : `~lsst.afw.image.Exposure` or `~lsst.afw.image.MaskedImage`
            Calibrated exposure.
        skyCorr : `~lsst.afw.math.backgroundList.BackgroundList`
            Full focal plane sky correction, obtained by running
            `~lsst.pipe.drivers.skyCorrection.SkyCorrectionTask`.
        """
        if isinstance(calexp, afwImage.Exposure):
            calexp = calexp.getMaskedImage()
        calexp -= skyCorr.getImage()

    def scaleModel(self, model, star, inPlace=True, nb90Rots=0):
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

        Returns
        -------
        scalingFactor : `float`
            The factor by which the model image should be multiplied for it
            to be scaled to the input bright star.
        """
        if self.config.scalingType == "annularFlux":
            scalingFactor = star.annularFlux
        elif self.config.scalingType == "leastSquare":
            if self.statsControl is None:
                self._setUpStatistics(star.stamp_im.mask)
            starIm = star.stamp_im.clone()
            # Rotate the star postage stamp.
            starIm = afwMath.rotateImageBy90(starIm, nb90Rots)
            # Reverse the prior star flux normalization ("unnormalize").
            starIm *= star.annularFlux
            # The estimator of the scalingFactor (f) that minimizes (Y-fX)^2
            # is E[XY]/E[XX].
            xy = starIm.clone()
            xy.image.array *= model.image.array
            xx = starIm.clone()
            xx.image.array = model.image.array**2
            # Compute the least squares scaling factor.
            xySum = afwMath.makeStatistics(xy, self.statsFlag, self.statsControl).getValue()
            xxSum = afwMath.makeStatistics(xx, self.statsFlag, self.statsControl).getValue()
            scalingFactor = xySum / xxSum if xxSum else 0
        else:
            raise AttributeError(f'Unknown scalingType "{self.config.scalingType}"; implemented options '
                                 'are "annularFlux" and "leastSquare".')
        if inPlace:
            model.image *= scalingFactor
        return scalingFactor

    def run(self, inputExposure, inputBrightStarStamps, inputExtendedPsf, skyCorr=None, dataId=None):
        """Iterate over all bright stars in an exposure to scale the extended
        PSF model before subtracting bright stars.

        Parameters
        ----------
        inputExposure : `afwImage.exposure.exposure.ExposureF`
            The image from which bright stars should be subtracted.
        inputBrightStarStamps :
                `~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`
            Set of stamps centered on each bright star to be subtracted,
            produced by running
            `~lsst.pipe.tasks.processBrightStars.ProcessBrightStarsTask`.
        inputExtendedPsf : `~lsst.pipe.tasks.extended_psf.ExtendedPsf`
            Extended PSF model, produced by
            `~lsst.pipe.tasks.extended_psf.MeasureExtendedPsfTask`.
        skyCorr : `~lsst.afw.math.backgroundList.BackgroundList` or `None`,
                  optional
            Full focal plane sky correction, obtained by running
            `~lsst.pipe.drivers.skyCorrection.SkyCorrectionTask`. If
            `doApplySkyCorr` is set to `True`, `skyCorr` cannot be `None`.
        dataId : `dict` or `~lsst.daf.butler.DataCoordinate`
            The dataId of the exposure (and detector) bright stars should be
            subtracted from.

        Returns
        -------
        subtractorExp : `afwImage.ExposureF`
            An Exposure containing a scaled bright star model fit to every
            bright star profile; its image can then be subtracted from the
            input exposure.
        invImages : `list` [`afwImage.MaskedImageF`]
            A list of small images ("stamps") containing the model, each scaled
            to its corresponding input bright star.
        """
        inputExpBBox = inputExposure.getBBox()
        if self.config.doApplySkyCorr and (skyCorr is not None):
            self.log.info("Applying sky correction to exposure %s (exposure will be modified in-place).",
                          dataId)
            self.applySkyCorr(inputExposure, skyCorr)
        # Create an empty image the size of the exposure.
        # TODO: DM-31085 (set mask planes).
        subtractorExp = afwImage.ExposureF(bbox=inputExposure.getBBox())
        subtractor = subtractorExp.maskedImage
        # Make a copy of the input model.
        model = inputExtendedPsf(dataId["detector"]).clone()
        modelStampSize = model.getDimensions()
        inv90Rots = 4 - inputBrightStarStamps.nb90Rots
        model = afwMath.rotateImageBy90(model, inv90Rots)
        warpCont = afwMath.WarpingControl(self.config.warpingKernelName)
        invImages = []
        # Loop over bright stars, computing the inverse transformed and scaled
        # postage stamp for each.
        for star in inputBrightStarStamps:
            if star.gaiaGMag < self.config.magLimit:
                # Set the origin.
                model.setXY0(star.position)
                # Create an empty destination image.
                invTransform = star.archive_element.inverted()
                invOrigin = geom.Point2I(invTransform.applyForward(geom.Point2D(star.position)))
                bbox = geom.Box2I(corner=invOrigin, dimensions=modelStampSize)
                invImage = afwImage.MaskedImageF(bbox)
                # Apply inverse transform.
                goodPix = afwMath.warpImage(invImage, model, invTransform, warpCont)
                if not goodPix:
                    self.log.debug(f"Warping of a model failed for star {star.gaiaId}: "
                                   "no good pixel in output")
                # Scale the model.
                self.scaleModel(invImage, star, inPlace=True, nb90Rots=inv90Rots)
                # Replace NaNs before subtraction (note all NaN pixels have
                # the NO_DATA flag).
                invImage.image.array[np.isnan(invImage.image.array)] = 0
                bbox.clip(inputExpBBox)
                if bbox.getArea() > 0:
                    subtractor[bbox] += invImage[bbox]
                invImages.append(invImage)
        return subtractorExp, invImages

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        dataId = butlerQC.quantum.dataId
        subtractor, _ = self.run(**inputs, dataId=dataId)
        if self.config.doWriteSubtractedExposure:
            outputExposure = inputs["inputExposure"].clone()
            outputExposure.image -= subtractor.image
        else:
            outputExposure = None
        outputBackgroundExposure = subtractor if self.config.doWriteSubtractor else None
        output = pipeBase.Struct(outputExposure=outputExposure,
                                 outputBackgroundExposure=outputBackgroundExposure)
        butlerQC.put(output, outputRefs)
