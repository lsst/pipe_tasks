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

"""Extract bright star cutouts; normalize and warp to the same pixel grid."""

__all__ = ["ProcessBrightStarsConnections", "ProcessBrightStarsConfig", "ProcessBrightStarsTask"]

import astropy.units as u
import numpy as np
from astropy.table import Table
from lsst.afw.cameraGeom import PIXELS, TAN_PIXELS
from lsst.afw.detection import FootprintSet, Threshold
from lsst.afw.geom.transformFactory import makeIdentityTransform, makeTransform
from lsst.afw.image import Exposure, ExposureF, MaskedImageF
from lsst.afw.math import (
    StatisticsControl,
    WarpingControl,
    rotateImageBy90,
    stringToStatisticsProperty,
    warpImage,
)
from lsst.geom import AffineTransform, Box2I, Extent2I, Point2D, Point2I, SpherePoint, radians
from lsst.meas.algorithms import LoadReferenceObjectsConfig, ReferenceObjectLoader
from lsst.meas.algorithms.brightStarStamps import BrightStarStamp, BrightStarStamps
from lsst.pex.config import Config, ChoiceField, ConfigField, Field, ListField, ConfigDictField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output, PrerequisiteInput
from lsst.utils.timer import timeMethod


class ProcessBrightStarsConnections(PipelineTaskConnections, dimensions=("instrument", "visit", "detector")):
    """Connections for ProcessBrightStarsTask."""

    inputExposure = Input(
        doc="Input exposure from which to extract bright star stamps.",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("visit", "detector"),
    )
    skyCorr = Input(
        doc="Input sky correction to be subtracted from the calexp if doApplySkyCorr=True.",
        name="skyCorr",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector"),
    )
    refCat = PrerequisiteInput(
        doc="Reference catalog that contains bright star positions",
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        multiple=True,
        deferLoad=True,
    )
    brightStarStamps = Output(
        doc="Set of preprocessed postage stamps, each centered on a single bright star.",
        name="brightStarStamps",
        storageClass="BrightStarStamps",
        dimensions=("visit", "detector"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.doApplySkyCorr:
            self.inputs.remove("skyCorr")


class BinLimits(Config):
    """Provides a list of upper and lower limits for magnitude bins."""

    limits = ListField[float](
        doc="A list containing the upper and lower limits of magnitude bins.",
        default=[],
    )


class StampSize(Config):
    """Provides a list of upper and lower limits for magnitude bins."""

    size = ListField[int](
        doc="A list containing the sized of stamps for each magnitude bin.",
        default=[],
    )


class AnnulusRadii(Config):
    """Provides a list of inner and outer radius for stamps of a magnitude bin."""

    radii = ListField[int](
        doc="A list containing the inner and outer radii of the annulus for each magnitude bin.",
        default=[],
    )


class ProcessBrightStarsConfig(PipelineTaskConfig, pipelineConnections=ProcessBrightStarsConnections):
    """Configuration parameters for ProcessBrightStarsTask."""

    magLimit = Field[float](
        doc="Magnitude limit, in Gaia G; all stars brighter than this value will be processed.",
        default=18,
    )
    magBins = ConfigDictField(
        keytype=str,
        itemtype=BinLimits,
        doc=(
            "Dictionary of magnitude limits for bins of stars to be processed. The keys are the names of "
            "the bins and the values are the magnitude limits."
        ),
        default={},
    )
    stampSize = ListField[int](
        doc="Size of the stamps to be extracted, in pixels.",
        default=(250, 250),
    )
    stampsSizes = ConfigDictField(
        keytype=str,
        itemtype=StampSize,
        doc="Sizes of the stamps to be extracted, in pixels, for each magnitude bin.",
        default={},
    )
    modelStampBuffer = Field[float](
        doc=(
            "'Buffer' factor to be applied to determine the size of the stamp the processed stars will be "
            "saved in. This will also be the size of the extended PSF model."
        ),
        default=1.1,
    )
    doRemoveDetected = Field[bool](
        doc="Whether secondary DETECTION footprints (i.e., footprints of objects other than the central "
        "primary object) should be changed to BAD.",
        default=True,
    )
    doApplyTransform = Field[bool](
        doc="Apply transform to bright star stamps to correct for optical distortions?",
        default=True,
    )
    warpingKernelName = ChoiceField[str](
        doc="Warping kernel.",
        default="lanczos5",
        allowed={
            "bilinear": "bilinear interpolation",
            "lanczos3": "Lanczos kernel of order 3",
            "lanczos4": "Lanczos kernel of order 4",
            "lanczos5": "Lanczos kernel of order 5",
        },
    )
    annularFluxRadii = ListField[int](
        doc="Inner and outer radii of the annulus used to compute AnnularFlux for normalization, in pixels.",
        default=(70, 80),
    )
    multiAnnularFluxRadii = ConfigDictField(
        keytype=str,
        itemtype=AnnulusRadii,
        doc="The inner and outer radii for the annulus of each magnitude bin.",
        default={},
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
    numSigmaClip = Field[float](
        doc="Sigma for outlier rejection; ignored if annularFluxStatistic != 'MEANCLIP'.",
        default=4,
    )
    numIter = Field[int](
        doc="Number of iterations of outlier rejection; ignored if annularFluxStatistic != 'MEANCLIP'.",
        default=3,
    )
    badMaskPlanes = ListField[str](
        doc="Mask planes that identify pixels to not include in the computation of the annular flux.",
        default=("BAD", "CR", "CROSSTALK", "EDGE", "NO_DATA", "SAT", "SUSPECT", "UNMASKEDNAN"),
    )
    minValidAnnulusFraction = Field[float](
        doc="Minumum number of valid pixels that must fall within the annulus for the bright star to be "
        "saved for subsequent generation of a PSF.",
        default=0.0,
    )
    doApplySkyCorr = Field[bool](
        doc="Apply full focal plane sky correction before extracting stars?",
        default=True,
    )
    discardNanFluxStars = Field[bool](
        doc="Should stars with NaN annular flux be discarded?",
        default=False,
    )
    refObjLoader = ConfigField[LoadReferenceObjectsConfig](
        doc="Reference object loader for astrometric calibration.",
    )


class ProcessBrightStarsTask(PipelineTask):
    """Extract bright star cutouts; normalize and warp to the same pixel grid.

    This task is used to extract, process, and store small image cut-outs
    (or "postage stamps") around bright stars. It relies on three methods,
    called in succession:

    `extractStamps`
        Find bright stars within the exposure using a reference catalog and
        extract a stamp centered on each.
    `warpStamps`
        Shift and warp each stamp to remove optical distortions and sample all
        stars on the same pixel grid.
    `measureAndNormalize`
        Compute the flux of an object in an annulus and normalize it. This is
        required to normalize each bright star stamp as their central pixels
        are likely saturated and/or contain ghosts, and cannot be used.
    """

    ConfigClass = ProcessBrightStarsConfig
    _DefaultName = "processBrightStars"

    def __init__(self, initInputs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setModelStamp()

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["dataId"] = str(butlerQC.quantum.dataId)
        refObjLoader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefs.refCat],
            refCats=inputs.pop("refCat"),
            name=self.config.connections.refCat,
            config=self.config.refObjLoader,
        )
        output = self.run(**inputs, refObjLoader=refObjLoader)
        # Only ingest stamp if it exists; prevent ingesting an empty FITS file.
        if output:
            butlerQC.put(output, outputRefs)

    @timeMethod
    def run(self, inputExposure, refObjLoader=None, dataId=None, skyCorr=None):
        """Identify bright stars within an exposure using a reference catalog,
        extract stamps around each, then preprocess them.

        Bright star preprocessing steps are: shifting, warping and potentially
        rotating them to the same pixel grid; computing their annular flux,
        and; normalizing them.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The image from which bright star stamps should be extracted.
        refObjLoader : `~lsst.meas.algorithms.ReferenceObjectLoader`, optional
            Loader to find objects within a reference catalog.
        dataId : `dict` or `~lsst.daf.butler.DataCoordinate`
            The dataId of the exposure (including detector) that bright stars
            should be extracted from.
        skyCorr : `~lsst.afw.math.backgroundList.BackgroundList`, optional
            Full focal plane sky correction obtained by `SkyCorrectionTask`.

        Returns
        -------
        brightStarResults : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``brightStarStamps``
                (`~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`)
        """
        if self.config.doApplySkyCorr:
            self.log.info("Applying sky correction to exposure %s (exposure modified in-place).", dataId)
            self.applySkyCorr(inputExposure, skyCorr)

        self.log.info("Extracting bright stars from exposure %s", dataId)
        # Extract stamps around bright stars.
        extractedStamps = self.extractStamps(inputExposure, refObjLoader=refObjLoader)
        if not extractedStamps.starStamps:
            self.log.info("No suitable bright star found.")
            return None
        # Warp (and shift, and potentially rotate) them.
        self.log.info(
            "Applying warp and/or shift to %i star stamps from exposure %s.",
            len(extractedStamps.starStamps),
            dataId,
        )
        warpOutputs = self.warpStamps(
            extractedStamps.starStamps,
            extractedStamps.pixCenters,
            extractedStamps.binTags
        )
        warpedStars = warpOutputs.warpedStars
        xy0s = warpOutputs.xy0s
        brightStarList = [
            BrightStarStamp(
                stamp_im=warp,
                archive_element=transform,
                position=xy0s[j],
                gaiaGMag=extractedStamps.gMags[j],
                gaiaId=extractedStamps.gaiaIds[j],
                binTag=warpOutputs.binTags[j],
                minValidAnnulusFraction=self.config.minValidAnnulusFraction,
            )
            for j, (warp, transform) in enumerate(zip(warpedStars, warpOutputs.warpTransforms))
        ]
        # Compute annularFlux and normalize
        self.log.info(
            "Computing annular flux and normalizing %i bright stars from exposure %s.",
            len(warpedStars),
            dataId,
        )
        # annularFlux statistic set-up, excluding mask planes
        statsControl = StatisticsControl(
            numSigmaClip=self.config.numSigmaClip,
            numIter=self.config.numIter,
        )

        # innerRadius, outerRadius = self.config.annularFluxRadii
        statsFlag = stringToStatisticsProperty(self.config.annularFluxStatistic)
        brightStarStamps = BrightStarStamps.initAndNormalize(
            brightStarList,
            # innerRadius=innerRadius,
            # outerRadius=outerRadius,
            annularFluxRadii=self.annularFluxRadii,
            nb90Rots=warpOutputs.nb90Rots,
            # imCenter=self.modelCenter,
            imsCenters=self.modelsCenters,
            modelSizes=self.modelStampsSizes,
            bins=self.bins,
            use_archive=True,
            statsControl=statsControl,
            statsFlag=statsFlag,
            badMaskPlanes=self.config.badMaskPlanes,
            discardNanFluxObjects=(self.config.discardNanFluxStars),
        )
        # Do not create empty FITS files if there aren't any normalized stamps.
        if not brightStarStamps._stamps:
            self.log.info("No normalized stamps exist for this exposure.")
            return None
        return Struct(brightStarStamps=brightStarStamps)

    def applySkyCorr(self, calexp, skyCorr):
        """Apply sky correction to the input exposure.

        Sky corrections can be generated using the
        `~lsst.pipe.tasks.skyCorrection.SkyCorrectionTask`.
        As the sky model generated via that task extends over the full focal
        plane, this should produce a more optimal sky subtraction solution.

        Parameters
        ----------
        calexp : `~lsst.afw.image.Exposure` or `~lsst.afw.image.MaskedImage`
            Calibrated exposure to correct.
        skyCorr : `~lsst.afw.math.backgroundList.BackgroundList`
            Full focal plane sky correction from ``SkyCorrectionTask``.

        Notes
        -----
        This method modifies the input ``calexp`` in-place.
        """
        if isinstance(calexp, Exposure):
            calexp = calexp.getMaskedImage()
        calexp -= skyCorr.getImage()

    def extractStamps(
        self, inputExposure, filterName="phot_g_mean", refObjLoader=None, inputBrightStarStamps=None
    ):
        """Identify the positions of bright stars within an input exposure using
        a reference catalog and extract them.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The image to extract bright star stamps from.
        filterName : `str`, optional
            Name of the camera filter to use for reference catalog filtering.
        refObjLoader : `~lsst.meas.algorithms.ReferenceObjectLoader`, optional
            Loader to find objects within a reference catalog.
        inputBrightStarStamps:
            `~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`, optional
            Provides information about the stars that have already been
            extracted from the inputExposure in other steps of the pipeline.
            For example, this is used in the `SubtractBrightStarsTask` to avoid
            extracting stars that already have been extracted when running
            `ProcessBrightStarsTask` to produce brightStarStamps.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``starStamps``
                Postage stamps (`list`).
            ``pixCenters``
                Corresponding coords to each star's center, in pixels (`list`).
            ``gMags``
                Corresponding (Gaia) G magnitudes (`list`).
            ``gaiaIds``
                Corresponding unique Gaia identifiers (`np.ndarray`).
        """
        if refObjLoader is None:
            refObjLoader = self.refObjLoader

        wcs = inputExposure.getWcs()
        inputBBox = inputExposure.getBBox()

        pixCenters = []
        starStamps = []
        badRows = []
        gMags = []
        ids = []
        binTags = []
        for i, bin in enumerate(self.bins):
            # Trim the reference catalog to only those objects within the exposure
            # bounding box dilated by half the bright star stamp size. This ensures
            # all stars that overlap the exposure are included.
            # Amir: this should contain dilatation for each bin separately. Maybe no need for central bin?
            # Amir: one dictionary for all bins or a for loope?????
            dilatationExtent = Extent2I(np.array(self.stampsSizes[bin].size) // 2)
            # Amir: this should be done for each bin separately
            withinExposure = refObjLoader.loadPixelBox(
                inputBBox.dilatedBy(dilatationExtent), wcs, filterName=filterName
            )
            refCat = withinExposure.refCat
            fluxField = withinExposure.fluxField

            # Define ref cat bright subset: objects within the mag limit.
            fluxLimit = ((self.magBins[bin].limits * u.ABmag).to(u.nJy)).to_value()  # AB magnitudes.
            fluxCond = refCat[fluxField] < fluxLimit[0]
            fluxCond &= refCat[fluxField] > fluxLimit[1]
            refCatBright = Table(
                refCat.extract("id", "coord_ra", "coord_dec", fluxField, where=fluxCond)
            )
            refCatBright["mag"] = (refCatBright[fluxField][:] * u.nJy).to(u.ABmag).to_value()  # AB magnitudes.

            # Remove input bright stars (if provided) from the bright subset.
            if inputBrightStarStamps is not None:
                # Extract the IDs of stars that have already been extracted.
                existing = np.isin(refCatBright["id"][:], inputBrightStarStamps.getGaiaIds())
                refCatBright = refCatBright[~existing]

            # Loop over each reference bright star, extract a stamp around it.
            for row, object in enumerate(refCatBright):
                coordSky = SpherePoint(object["coord_ra"], object["coord_dec"], radians)
                coordPix = wcs.skyToPixel(coordSky)
                # TODO: Replace this method with exposure getCutout after DM-40042.
                starStamp = self._getCutout(inputExposure, coordPix, self.stampsSizes[bin].size)
                if not starStamp:
                    badRows.append(row)
                    continue
                if self.config.doRemoveDetected:
                    self._replaceSecondaryFootprints(starStamp, coordPix, object["id"])
                starStamps.append(starStamp)
                pixCenters.append(coordPix)
                gMags.append(refCatBright["mag"][row])
                ids.append(refCatBright["id"][row])
                binTags.append(bin)

        return Struct(starStamps=starStamps, pixCenters=pixCenters, gMags=gMags, gaiaIds=ids, binTags=binTags)

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
        # TODO: Replace this method with exposure getCutout after DM-40042.
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
            # Restore pixels which overlap the input exposure.
            inputMI = inputExposure.maskedImage
            overlap = inputMI.Factory(inputMI, overlapBBox)
            stamp.maskedImage[overlapBBox] = overlap
            # Set detector and WCS.
            stamp.setDetector(inputExposure.getDetector())
            stamp.setWcs(inputExposure.getWcs())
        else:
            stamp = None
        return stamp

    def _replaceSecondaryFootprints(self, stamp, coordPix, objectId, find="DETECTED", replace="BAD"):
        """Replace all secondary footprints in a stamp with another mask flag.

        This method identifies all secondary footprints in a stamp as those
        whose ``find`` footprints do not overlap the given pixel coordinates.
        If then sets these secondary footprints to the ``replace`` flag.

        Parameters
        ----------
        stamp : `~lsst.afw.image.ExposureF`
            The postage stamp to modify.
        coordPix : `~lsst.geom.Point2D`
            The pixel coordinates of the central primary object.
        objectId : `int`
            The unique identifier of the central primary object.
        find : `str`, optional
            The mask plane to use to identify secondary footprints.
        replace : `str`, optional
            The mask plane to set secondary footprints to.

        Notes
        -----
        This method modifies the input ``stamp`` in-place.
        """
        # Find a FootprintSet given an Image and a threshold.
        detThreshold = Threshold(stamp.mask.getPlaneBitMask(find), Threshold.BITMASK)
        footprintSet = FootprintSet(stamp.mask, detThreshold)
        allFootprints = footprintSet.getFootprints()
        # Identify secondary objects (i.e., not the central primary object).
        secondaryFootprints = []
        for footprint in allFootprints:
            if not footprint.contains(Point2I(coordPix)):
                secondaryFootprints.append(footprint)
        # Set secondary object footprints to BAD.
        # Note: the value of numPrimaryFootprints can only be 0 or 1. If it is
        # 0, then the primary object was not found overlapping a footprint.
        # This can occur for low-S/N stars, for example. Processing can still
        # continue beyond this point in an attempt to utilize this faint flux.
        if (numPrimaryFootprints := len(allFootprints) - len(secondaryFootprints)) == 0:
            self.log.info(
                "Could not uniquely identify central %s footprint for star %s; "
                "found %d footprints instead.",
                find,
                objectId,
                numPrimaryFootprints,
            )
        footprintSet.setFootprints(secondaryFootprints)
        footprintSet.setMask(stamp.mask, replace)

    def warpStamps(self, stamps, pixCenters, binTags):
        """Warps and shifts all given stamps so they are sampled on the same
        pixel grid and centered on the central pixel. This includes rotating
        the stamp depending on detector orientation.

        Parameters
        ----------
        stamps : `Sequence` [`~lsst.afw.image.ExposureF`]
            Image cutouts centered on a single object.
        pixCenters : `Sequence` [`~lsst.geom.Point2D`]
            Positions of each object's center (from the refCat) in pixels.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``warpedStars``
                Stamps of warped stars.
                    (`list` [`~lsst.afw.image.MaskedImage`])
            ``warpTransforms``
                The corresponding Transform from the initial star stamp
                to the common model grid.
                    (`list` [`~lsst.afw.geom.TransformPoint2ToPoint2`])
            ``xy0s``
                Coordinates of the bottom-left pixels of each stamp,
                before rotation.
                    (`list` [`~lsst.geom.Point2I`])
            ``nb90Rots``
                The number of 90 degrees rotations required to compensate for
                detector orientation.
                    (`int`)
        """
        # warping control; only contains shiftingALg provided in config
        warpCont = WarpingControl(self.config.warpingKernelName)
        # Initialize detector instance (note all stars were extracted from an
        # exposure from the same detector)
        det = stamps[0].getDetector()
        # Define correction for optical distortions
        if self.config.doApplyTransform:
            pixToTan = det.getTransform(PIXELS, TAN_PIXELS)
        else:
            pixToTan = makeIdentityTransform()
        # Array of all possible rotations for detector orientation:
        possibleRots = np.array([k * np.pi / 2 for k in range(4)])
        # determine how many, if any, rotations are required
        yaw = det.getOrientation().getYaw()
        nb90Rots = np.argmin(np.abs(possibleRots - float(yaw)))

        # apply transformation to each star
        warpedStars, warpTransforms, xy0s = [], [], []
        # Compare model to star stamp sizes
        # Amir: this should be done for all bins?
        binsBufferPix = {}
        for bin in self.bins:
            bufferPix = (
                self.modelStampsSizes[bin][0] - self.stampsSizes[bin].size[0],
                self.modelStampsSizes[bin][1] - self.stampsSizes[bin].size[1],
            )
            binsBufferPix[bin] = bufferPix
        for star, cent, binTag in zip(stamps, pixCenters, binTags):
            # (re)create empty destination image
            destImage = MaskedImageF(*self.modelStampsSizes[binTag])
            bottomLeft = Point2D(star.image.getXY0())
            newBottomLeft = pixToTan.applyForward(bottomLeft)
            newBottomLeft.setX(newBottomLeft.getX() - binsBufferPix[binTag][0] / 2)
            newBottomLeft.setY(newBottomLeft.getY() - binsBufferPix[binTag][1] / 2)
            # Convert to int
            newBottomLeft = Point2I(newBottomLeft)
            # Set origin and save it
            destImage.setXY0(newBottomLeft)
            xy0s.append(newBottomLeft)

            # Define linear shifting to recenter stamps
            newCenter = pixToTan.applyForward(cent)  # center of warped star
            shift = (
                self.modelsCenters[binTag][0] + newBottomLeft[0] - newCenter[0],
                self.modelsCenters[binTag][1] + newBottomLeft[1] - newCenter[1],
            )
            affineShift = AffineTransform(shift)
            shiftTransform = makeTransform(affineShift)

            # Define full transform (warp and shift)
            starWarper = pixToTan.then(shiftTransform)

            # Apply it
            goodPix = warpImage(destImage, star.getMaskedImage(), starWarper, warpCont)
            if not goodPix:
                self.log.debug("Warping of a star failed: no good pixel in output")

            # Arbitrarily set origin of shifted star to 0
            destImage.setXY0(0, 0)

            # Apply rotation if appropriate
            if nb90Rots:
                destImage = rotateImageBy90(destImage, nb90Rots)
            warpedStars.append(destImage.clone())
            warpTransforms.append(starWarper)
        return Struct(warpedStars=warpedStars, warpTransforms=warpTransforms, xy0s=xy0s, nb90Rots=nb90Rots, binTags=binTags)

    def setModelStamp(self):
        """Compute (model) stamp size depending on provided buffer value."""
        if not self.config.magBins:
            self.setMagBins()
        else:
            self.magBins = self.config.magBins
            self.bins = list(self.magBins.keys())
        if not self.config.stampsSizes:
            self.setStampsSizes()
        else:
            self.stampsSizes = self.config.stampsSizes
        self.applyModelStampBuffer()
        # self.modelStampSize = [
        #     int(self.config.stampSize[0] * self.config.modelStampBuffer),
        #     int(self.config.stampSize[1] * self.config.modelStampBuffer),
        # ]
        # # Force stamp to be odd-sized so we have a central pixel.
        # if not self.modelStampSize[0] % 2:
        #     self.modelStampSize[0] += 1
        # if not self.modelStampSize[1] % 2:
        #     self.modelStampSize[1] += 1
        # # Central pixel.
        # self.modelCenter = self.modelStampSize[0] // 2, self.modelStampSize[1] // 2
        if not self.config.multiAnnularFluxRadii:
            self.setAnnularRadii()
        else:
            self.annularFluxRadii = self.config.multiAnnularFluxRadii

    def setMagBins(self):
        """Set default magnitude bins."""
        self.bins = ["100", "101", "102"]
        # Test continues magnitude for bins
        limits = [[19, 19.5], [16, 17], [11, 12]]
        self.magBins = {}
        for key, value in zip(self.bins, limits):
            binLimits = BinLimits()
            binLimits.limits = value
            self.magBins[key] = binLimits

    def setStampsSizes(self):
        """Set default stamps sizes."""
        sizes = [[130, 130], [200, 200], [600, 600]]
        self.stampsSizes = {}
        for key, value in zip(self.bins, sizes):
            stampSize = StampSize()
            stampSize.size = value
            self.stampsSizes[key] = stampSize

    def applyModelStampBuffer(self):
        """Apply buffer to determine the model stamps sizes."""
        self.modelStampsSizes = {}
        self.modelsCenters = {}
        for key, value in self.stampsSizes.items():
            self.modelStampsSizes[key] = [
                int(value.size[0] * self.config.modelStampBuffer),
                int(value.size[1] * self.config.modelStampBuffer),
            ]
            # Force stamp to be odd-sized so we have a central pixel.
            if not self.modelStampsSizes[key][0] % 2:
                self.modelStampsSizes[key][0] += 1
            if not self.modelStampsSizes[key][1] % 2:
                self.modelStampsSizes[key][1] += 1
            # Central pixel.
            self.modelsCenters[key] = self.modelStampsSizes[key][0] // 2, self.modelStampsSizes[key][1] // 2

    def setAnnularRadii(self):
        """Set default annular flux radii."""
        self.annularFluxRadii = {}
        radii = [[12, 20], [23, 33], [90, 105]]
        for i, key in enumerate(self.bins):
            annularFluxRadii = AnnulusRadii()
            annularFluxRadii.radii = radii[i]
            self.annularFluxRadii[key] = annularFluxRadii
