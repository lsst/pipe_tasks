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

__all__ = ["SkyCorrectionTask", "SkyCorrectionConfig"]

import warnings

import numpy as np

from lsst.afw.image import ExposureF, makeMaskedImage
from lsst.afw.math import BackgroundMI, binImage
from lsst.daf.butler import DeferredDatasetHandle
from lsst.pex.config import Config, ConfigField, ConfigurableField, Field
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output, PrerequisiteInput
from lsst.pipe.tasks.background import (
    FocalPlaneBackground,
    FocalPlaneBackgroundConfig,
    MaskObjectsTask,
    SkyMeasurementTask,
)
from lsst.pipe.tasks.visualizeVisit import VisualizeMosaicExpConfig, VisualizeMosaicExpTask


def _reorderAndPadList(inputList, inputKeys, outputKeys, padWith=None):
    """Match the order of one list to another, padding if necessary.

    Parameters
    ----------
    inputList : `list`
        List to be reordered and padded. Elements can be any type.
    inputKeys :  iterable
        Iterable of values to be compared with outputKeys.
        Length must match `inputList`.
    outputKeys : iterable
        Iterable of values to be compared with inputKeys.
    padWith :
        Any value to be inserted where one of inputKeys is not in outputKeys.

    Returns
    -------
    outputList : `list`
        Copy of inputList reordered per outputKeys and padded with `padWith`
        so that the length matches length of outputKeys.
    """
    outputList = []
    for outputKey in outputKeys:
        if outputKey in inputKeys:
            outputList.append(inputList[inputKeys.index(outputKey)])
        else:
            outputList.append(padWith)
    return outputList


class SkyCorrectionConnections(PipelineTaskConnections, dimensions=("instrument", "visit")):
    calExps = Input(
        doc="Background-subtracted calibrated exposures.",
        name="calexp",
        multiple=True,
        deferLoad=True,
        storageClass="ExposureF",
        dimensions=["instrument", "visit", "detector"],
    )
    calBkgs = Input(
        doc="Subtracted backgrounds for input calibrated exposures.",
        name="calexpBackground",
        multiple=True,
        storageClass="Background",
        dimensions=["instrument", "visit", "detector"],
    )
    backgroundToPhotometricRatioHandles = Input(
        doc="Ratio of a background-flattened image to a photometric-flattened image. "
        "Only used if doApplyFlatBackgroundRatio is True.",
        multiple=True,
        name="background_to_photometric_ratio",
        storageClass="Image",
        dimensions=["instrument", "visit", "detector"],
        deferLoad=True,
    )
    skyFrames = PrerequisiteInput(
        doc="Calibration sky frames.",
        name="sky",
        multiple=True,
        deferLoad=True,
        storageClass="ExposureF",
        dimensions=["instrument", "physical_filter", "detector"],
        isCalibration=True,
    )
    camera = PrerequisiteInput(
        doc="Input camera.",
        name="camera",
        storageClass="Camera",
        dimensions=["instrument"],
        isCalibration=True,
    )
    skyCorr = Output(
        doc="Sky correction data, to be subtracted from the calibrated exposures.",
        name="skyCorr",
        multiple=True,
        storageClass="Background",
        dimensions=["instrument", "visit", "detector"],
    )
    calExpMosaic = Output(
        doc="Full focal plane mosaicked image of the sky corrected calibrated exposures.",
        name="calexp_skyCorr_visit_mosaic",
        storageClass="ImageF",
        dimensions=["instrument", "visit"],
    )
    calBkgMosaic = Output(
        doc="Full focal plane mosaicked image of the sky corrected calibrated exposure backgrounds.",
        name="calexpBackground_skyCorr_visit_mosaic",
        storageClass="ImageF",
        dimensions=["instrument", "visit"],
    )

    def __init__(self, *, config: "SkyCorrectionConfig | None" = None):
        super().__init__(config=config)
        assert config is not None
        if not config.doSky:
            del self.skyFrames
        if not config.doApplyFlatBackgroundRatio:
            del self.backgroundToPhotometricRatioHandles


class SkyCorrectionConfig(PipelineTaskConfig, pipelineConnections=SkyCorrectionConnections):
    doApplyFlatBackgroundRatio = Field(
        dtype=bool,
        default=False,
        doc="This should be True if the input image was processed with an illumination correction.",
    )
    maskObjects = ConfigurableField(
        target=MaskObjectsTask,
        doc="Mask Objects",
    )
    doMaskObjects = Field(
        dtype=bool,
        default=True,
        doc="Iteratively mask objects to find good sky?",
    )
    bgModel1 = ConfigField(
        dtype=FocalPlaneBackgroundConfig,
        doc="Initial background model, prior to sky frame subtraction",
    )
    undoBgModel1 = Field(
        dtype=bool,
        default=False,
        doc="If True, adds back initial background model after sky and removes bgModel1 from the list",
    )
    sky = ConfigurableField(
        target=SkyMeasurementTask,
        doc="Sky measurement",
    )
    doSky = Field(
        dtype=bool,
        default=True,
        doc="Do sky frame subtraction?",
    )
    bgModel2 = ConfigField(
        dtype=FocalPlaneBackgroundConfig,
        doc="Final (cleanup) background model, after sky frame subtraction",
    )
    doBgModel2 = Field(
        dtype=bool,
        default=True,
        doc="Do final (cleanup) background model subtraction, after sky frame subtraction?",
    )
    binning = Field(
        dtype=int,
        default=8,
        doc="Binning factor for constructing full focal plane '*_camera' output datasets",
    )

    def setDefaults(self):
        Config.setDefaults(self)
        self.bgModel2.doSmooth = True
        self.bgModel2.minFrac = 0.3
        self.bgModel2.xSize = 256
        self.bgModel2.ySize = 256
        self.bgModel2.smoothScale = 1.0

    def validate(self):
        super().validate()
        if self.undoBgModel1 and not self.doSky and not self.doBgModel2:
            raise ValueError("If undoBgModel1 is True, task requires at least one of doSky or doBgModel2.")


class SkyCorrectionTask(PipelineTask):
    """Perform a full focal plane sky correction."""

    ConfigClass = SkyCorrectionConfig
    _DefaultName = "skyCorr"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("sky")
        self.makeSubtask("maskObjects")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Sort input/output connections by detector ID, padding where
        # necessary, to ensure that all detectors are processed consistently.
        # Detector IDs are defined from the intersection of calExps, calBkgs,
        # and optionally skyFrames and backgroundToPhotometricRatioHandles.
        # This resolves potential missing data issues when processing a visit
        # that contains only partial inputs.
        calExpOrder = {ref.dataId["detector"] for ref in inputRefs.calExps}
        calBkgOrder = {ref.dataId["detector"] for ref in inputRefs.calBkgs}
        detectorOrder = calExpOrder & calBkgOrder
        if self.config.doApplyFlatBackgroundRatio:
            ratioOrder = {ref.dataId["detector"] for ref in inputRefs.backgroundToPhotometricRatioHandles}
            detectorOrder &= ratioOrder
        if self.config.doSky:
            skyFrameOrder = {ref.dataId["detector"] for ref in inputRefs.skyFrames}
            detectorOrder &= skyFrameOrder
        detectorOrder = sorted(detectorOrder)
        inputRefs.calExps = _reorderAndPadList(
            inputRefs.calExps, [ref.dataId["detector"] for ref in inputRefs.calExps], detectorOrder
        )
        inputRefs.calBkgs = _reorderAndPadList(
            inputRefs.calBkgs, [ref.dataId["detector"] for ref in inputRefs.calBkgs], detectorOrder
        )
        # Only attempt to fetch sky frames if they are going to be applied.
        if self.config.doSky:
            inputRefs.skyFrames = _reorderAndPadList(
                inputRefs.skyFrames, [ref.dataId["detector"] for ref in inputRefs.skyFrames], detectorOrder
            )
        else:
            inputRefs.skyFrames = []
        # Only attempt to fetch flat ratios if they are going to be applied.
        if self.config.doApplyFlatBackgroundRatio:
            inputRefs.backgroundToPhotometricRatioHandles = _reorderAndPadList(
                inputRefs.backgroundToPhotometricRatioHandles,
                [ref.dataId["detector"] for ref in inputRefs.backgroundToPhotometricRatioHandles],
                detectorOrder,
            )
        else:
            inputRefs.backgroundToPhotometricRatioHandles = []
        outputRefs.skyCorr = _reorderAndPadList(
            outputRefs.skyCorr, [ref.dataId["detector"] for ref in outputRefs.skyCorr], detectorOrder
        )
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, calExps, calBkgs, skyFrames, camera, backgroundToPhotometricRatioHandles=[]):
        """Perform sky correction on a visit.

        The original visit-level background is first restored to the calibrated
        exposure and the existing background model is inverted in-place. If
        doMaskObjects is True, the mask map associated with this exposure will
        be iteratively updated (over nIter loops) by re-estimating the
        background each iteration and redetecting footprints.

        An initial full focal plane sky subtraction (bgModel1) will take place
        prior to scaling and subtracting the sky frame.

        If doSky is True, the sky frame will be scaled to the flux in the input
        visit.

        If doBgModel2 is True, a final full focal plane sky subtraction will
        take place after the sky frame has been subtracted.

        The first N elements of the returned skyCorr will consist of inverted
        elements of the calexpBackground model (i.e., subtractive). All
        subsequent elements appended to skyCorr thereafter will be additive
        such that, when skyCorr is subtracted from a calexp, the net result
        will be to undo the initial per-detector background solution and then
        apply the skyCorr model thereafter. Adding skyCorr to a
        calexpBackground will effectively negate the calexpBackground,
        returning only the additive background components of the skyCorr
        background model.

        Parameters
        ----------
        calExps : `list` [`lsst.afw.image.ExposureF`]
            Detector calibrated exposure images for the visit.
        calBkgs : `list` [`lsst.afw.math.BackgroundList`]
            Detector background lists matching the calibrated exposures.
        skyFrames : `list` [`lsst.afw.image.ExposureF`]
            Sky frame calibration data for the input detectors.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera matching the input data to process.
        backgroundToPhotometricRatioHandles :
                `list` [`lsst.daf.butler.DeferredDatasetHandle`], optional
            Deferred dataset handles pointing to the Background to photometric
            ratio images for the input detectors.

        Returns
        -------
        results : `Struct` containing:
            skyFrameScale : `float`
                Scale factor applied to the sky frame.
            skyCorr : `list` [`lsst.afw.math.BackgroundList`]
                Detector-level sky correction background lists.
            calExpMosaic : `lsst.afw.image.ExposureF`
                Visit-level mosaic of the sky corrected data, binned.
                Analogous to `calexp - skyCorr`.
            calBkgMosaic : `lsst.afw.image.ExposureF`
                Visit-level mosaic of the sky correction background, binned.
                Analogous to `calexpBackground + skyCorr`.
        """
        # Process each detector separately and merge into bgModel1.
        # This avoids storing every full-res image in-memory at once.
        bgModel1 = FocalPlaneBackground.fromCamera(self.config.bgModel1, camera)
        detectors = []
        masks = []
        skyCorrs = []
        bgModel1Indices = []
        if not self.config.doApplyFlatBackgroundRatio:
            backgroundToPhotometricRatioHandles = [None] * len(calExps)
        for calExpHandle, calBkg, backgroundToPhotometricRatioHandle in zip(
            calExps, calBkgs, backgroundToPhotometricRatioHandles
        ):
            calExp = self._getCalExp(
                calExpHandle, backgroundToPhotometricRatioHandle=backgroundToPhotometricRatioHandle
            )
            detectors.append(calExp.getDetector())

            # Restore original background in-place; optionally refine mask maps
            _ = self._restoreOriginalBackgroundRefineMask(calExp, calBkg)
            masks.append(calExp.mask)
            skyCorrs.append(calBkg)  # Contains only the inverted original background elements at this stage
            bgModel1Indices.append(len(calBkg))  # Index of the original background element

            # Make a background model for the image, using bgModel1 configs
            bgModel1Detector = FocalPlaneBackground.fromCamera(self.config.bgModel1, camera)
            bgModel1Detector.addCcd(calExp)
            bgModel1.merge(bgModel1Detector)
            self.log.info(
                "Detector %d: Merged %d unmasked pixels (%.1f%s of detector area) into initial BG model",
                calExp.getDetector().getId(),
                bgModel1Detector._numbers.getArray().sum(),
                100 * bgModel1Detector._numbers.getArray().sum() / calExp.getBBox().getArea(),
                "%",
            )

        # Validate bgModel1
        self._validateBgModel("bgModel1", bgModel1, self.config.bgModel1)

        # Update skyCorrs with new bgModel1 background elements
        for detector, skyCorr in zip(detectors, skyCorrs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "invalid value encountered")
                calBkgElement = bgModel1.toCcdBackground(detector, detector.getBBox())
            skyCorr.append(calBkgElement[0])

        # Fit a scaled sky frame to all input exposures
        skyFrameScale = None
        if self.config.doSky:
            skyFrameScale = self._fitSkyFrame(
                calExps, masks, skyCorrs, skyFrames, backgroundToPhotometricRatioHandles
            )

        # Remove the initial background model (bgModel1) from every skyCorr
        if self.config.undoBgModel1:
            for skyCorr, bgModel1Index in zip(skyCorrs, bgModel1Indices):
                skyCorr._backgrounds.pop(bgModel1Index)
            self.log.info(
                "Initial background models (bgModel1s) have been removed from all skyCorr background lists",
            )

        # Bin exposures, generate full-fp bg, map to CCDs and subtract in-place
        if self.config.doBgModel2:
            bgModel2 = FocalPlaneBackground.fromCamera(self.config.bgModel2, camera)
            for calExpHandle, mask, skyCorr, backgroundToPhotometricRatioHandle in zip(
                calExps, masks, skyCorrs, backgroundToPhotometricRatioHandles
            ):
                calExp = self._getCalExp(calExpHandle, mask, skyCorr, backgroundToPhotometricRatioHandle)

                # Make a background model for the image, using bgModel2 configs
                bgModel2Detector = FocalPlaneBackground.fromCamera(self.config.bgModel2, camera)
                bgModel2Detector.addCcd(calExp)
                bgModel2.merge(bgModel2Detector)
                self.log.info(
                    "Detector %d: Merged %d unmasked pixels (%.1f%s of detector area) into final BG model",
                    calExp.getDetector().getId(),
                    bgModel2Detector._numbers.getArray().sum(),
                    100 * bgModel2Detector._numbers.getArray().sum() / calExp.getBBox().getArea(),
                    "%",
                )

            # Validate bgModel2
            self._validateBgModel("bgModel2", bgModel2, self.config.bgModel2)

            # Update skyCorrs with new bgModel2 background elements
            for detector, skyCorr in zip(detectors, skyCorrs):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "invalid value encountered")
                    calBkgElement = bgModel2.toCcdBackground(detector, detector.getBBox())
                skyCorr.append(calBkgElement[0])

        # Make camera-level mosaics of bg subtracted calexps and subtracted bgs
        calExpsBinned = []
        calBkgsBinned = []
        for calExpHandle, mask, skyCorr, backgroundToPhotometricRatioHandle, bgModel1Index in zip(
            calExps, masks, skyCorrs, backgroundToPhotometricRatioHandles, bgModel1Indices
        ):
            calExp = self._getCalExp(calExpHandle, mask, skyCorr, backgroundToPhotometricRatioHandle)

            skyCorrExtra = skyCorr.clone()  # new skyCorr elements created in this task
            skyCorrExtra._backgrounds = skyCorrExtra._backgrounds[bgModel1Index:]
            skyCorrExtraMI = makeMaskedImage(skyCorrExtra.getImage())
            skyCorrExtraMI.setMask(calExp.getMask())

            calExpsBinned.append(binImage(calExp.getMaskedImage(), self.config.binning))
            calBkgsBinned.append(binImage(skyCorrExtraMI, self.config.binning))

        mosConfig = VisualizeMosaicExpConfig()
        mosConfig.binning = self.config.binning
        mosTask = VisualizeMosaicExpTask(config=mosConfig)
        detectorIds = [detector.getId() for detector in detectors]
        calExpMosaic = mosTask.run(calExpsBinned, camera, inputIds=detectorIds).outputData
        calBkgMosaic = mosTask.run(calBkgsBinned, camera, inputIds=detectorIds).outputData

        return Struct(
            skyFrameScale=skyFrameScale,
            skyCorr=skyCorrs,
            calExpMosaic=calExpMosaic,
            calBkgMosaic=calBkgMosaic,
        )

    def _getCalExp(self, calExpHandle, mask=None, skyCorr=None, backgroundToPhotometricRatioHandle=None):
        """Get a calexp from a DeferredDatasetHandle, and optionally apply an
        updated mask and skyCorr.

        Parameters
        ----------
        calExpHandle : `~lsst.afw.image.ExposureF`
                | `lsst.daf.butler.DeferredDatasetHandle`
            Either the image exposure data or a handle to the calexp dataset.
        mask : `lsst.afw.image.Mask`, optional
            Mask to apply to the calexp.
        skyCorr : `lsst.afw.math.BackgroundList`, optional
            Background list to subtract from the calexp.

        Returns
        -------
        calExp : `lsst.afw.image.ExposureF`
            The calexp with the mask and skyCorr applied.
        """
        if isinstance(calExpHandle, DeferredDatasetHandle):
            calExp: ExposureF = calExpHandle.get()
        else:
            # Here we clone the imaging data to avoid modifying data which is
            # used in downstream processing.
            calExp: ExposureF = calExpHandle.clone()

        # Convert from background-flattened to photometric-flattened images
        # Note: remember to convert back to background-flattened images
        # if science images are to be output by this task.
        if self.config.doApplyFlatBackgroundRatio:
            if not backgroundToPhotometricRatioHandle:
                raise ValueError(
                    "A list of backgroundToPhotometricRatioHandles must be supplied if "
                    "config.doApplyFlatBackgroundRatio=True.",
                )
            ratioImage = backgroundToPhotometricRatioHandle.get()
            calExp.maskedImage *= ratioImage
            self.log.info(
                "Detector %d: Converted background-flattened image to a photometric-flattened image",
                calExp.getDetector().getId(),
            )

        if mask is not None:
            calExp.setMask(mask)
        if skyCorr is not None:
            image = calExp.getMaskedImage()
            image -= skyCorr.getImage()

        return calExp

    def _getSkyFrame(self, skyFrameHandle):
        """Get a calexp from a DeferredDatasetHandle, and optionally apply an
        updated mask and skyCorr.

        Parameters
        ----------
        skyFrameHandle : `lsst.daf.butler.DeferredDatasetHandle`
            Either the sky frame data or a handle to the sky frame dataset.

        Returns
        -------
        skyFrame : `lsst.afw.image.ExposureF`
            The calexp with the mask and skyCorr applied.
        """
        if isinstance(skyFrameHandle, DeferredDatasetHandle):
            skyFrame: ExposureF = skyFrameHandle.get()
        else:
            skyFrame: ExposureF = skyFrameHandle
        return skyFrame

    def _restoreOriginalBackgroundRefineMask(self, calExp, calBkg):
        """Restore original background to a calexp and invert the related
        background model; optionally refine the mask plane.

        The original visit-level background is restored to the calibrated
        exposure and the existing background model is inverted in-place. If
        doMaskObjects is True, the mask map associated with the exposure will
        be iteratively updated (over nIter loops) by re-estimating the
        background each iteration and redetecting footprints.

        The background model modified in-place in this method will comprise the
        first N elements of the skyCorr dataset type, i.e., these N elements
        are the inverse of the calexpBackground model. All subsequent elements
        appended to skyCorr will be additive such that, when skyCorr is
        subtracted from a calexp, the net result will be to undo the initial
        per-detector background solution and then apply the skyCorr model
        thereafter. Adding skyCorr to a calexpBackground will effectively
        negate the calexpBackground, returning only the additive background
        components of the skyCorr background model.

        Parameters
        ----------
        calExp : `lsst.afw.image.ExposureF`
            Detector level calexp image.
        calBkg : `lsst.afw.math.BackgroundList`
            Detector level background lists associated with the calexp.

        Returns
        -------
        calExp : `lsst.afw.image.ExposureF`
            The calexp with the originally subtracted background restored.
        skyCorrBase : `lsst.afw.math.BackgroundList`
            The inverted original background models; the genesis for skyCorr.
        """
        image = calExp.getMaskedImage()

        # Invert all elements of the existing bg model; restore in calexp
        for calBkgElement in calBkg:
            statsImage = calBkgElement[0].getStatsImage()
            statsImage *= -1
        skyCorrBase = calBkg.getImage()
        image -= skyCorrBase

        stats = np.nanpercentile(skyCorrBase.array, [50, 75, 25])
        self.log.info(
            "Detector %d: Original background restored (BG median = %.1f counts, BG IQR = %.1f counts)",
            calExp.getDetector().getId(),
            -stats[0],
            np.subtract(*stats[1:]),
        )

        # Iteratively subtract bg, re-detect sources, and add bg back on
        if self.config.doMaskObjects:
            maskFrac0 = 1 - np.sum(calExp.mask.array == 0) / calExp.mask.array.size
            self.maskObjects.findObjects(calExp)
            maskFrac1 = 1 - np.sum(calExp.mask.array == 0) / calExp.mask.array.size

            self.log.info(
                "Detector %d: Iterative source detection and mask growth has increased masked area by %.1f%%",
                calExp.getDetector().getId(),
                (100 * (maskFrac1 - maskFrac0)),
            )

        return calExp, skyCorrBase

    def _validateBgModel(self, bgModelID, bgModel, config):
        """Check that the background model contains enough valid superpixels,
        and raise a useful error if not.

        Parameters
        ----------
        bgModelID : `str`
            Identifier for the background model.
        bgModel : `~lsst.pipe.tasks.background.FocalPlaneBackground`
            Background model to check.
        config : `~lsst.pipe.tasks.background.FocalPlaneBackgroundConfig`
            Configuration used to create the background model.
        """
        bgModelArray = bgModel._numbers.getArray()
        spArea = (config.xSize / config.pixelSize) * (config.ySize / config.pixelSize)
        self.log.info(
            "%s: FP background model constructed using %.2f x %.2f mm (%d x %d pixel) superpixels",
            bgModelID,
            config.xSize,
            config.ySize,
            int(config.xSize / config.pixelSize),
            int(config.ySize / config.pixelSize),
        )
        self.log.info(
            "%s: Pixel data exists in %d of %d superpixels; the most populated superpixel is %.1f%% filled",
            bgModelID,
            np.sum(bgModelArray > 0),
            bgModelArray.size,
            100 * np.max(bgModelArray) / spArea,
        )

        thresh = config.minFrac * spArea
        if np.all(bgModelArray < thresh):
            raise RuntimeError(
                f"No background model superpixels are more than {100*config.minFrac}% filled. "
                "Try decreasing the minFrac configuration parameter, optimizing the subset of detectors "
                "being processed, or increasing the number of detectors being processed."
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"invalid value encountered")
            stats = np.nanpercentile(bgModel.getStatsImage().array, [50, 75, 25])
        self.log.info(
            "%s: FP BG median = %.1f counts, FP BG IQR = %.1f counts",
            bgModelID,
            stats[0],
            np.subtract(*stats[1:]),
        )

    def _fitSkyFrame(self, calExps, masks, skyCorrs, skyFrames, backgroundToPhotometricRatioHandles):
        """Determine the full focal plane sky frame scale factor relative to
        an input list of calibrated exposures.

        This method measures the sky frame scale on all inputs, resulting in
        values equal to the background method solveScales().

        Input skyCorrs are updated in-place.

        Parameters
        ----------
        calExps : `list` [`lsst.afw.image.ExposureF`]
            Calibrated exposures to be background subtracted.
        masks : `list` [`lsst.afw.image.Mask`]
            Masks associated with the input calibrated exposures.
        skyCorrs : `list` [`lsst.afw.math.BackgroundList`]
            Background lists associated with the input calibrated exposures.
        skyFrames : `list` [`lsst.afw.image.ExposureF`]
            Sky frame calibration data for the input detectors.

        Returns
        -------
        scale : `float`
            Fitted scale factor applied to the sky frame.
        """
        skyBkgs = []
        scales = []
        for calExpHandle, mask, skyCorr, skyFrameHandle, backgroundToPhotometricRatioHandle in zip(
            calExps, masks, skyCorrs, skyFrames, backgroundToPhotometricRatioHandles
        ):
            calExp = self._getCalExp(calExpHandle, mask, skyCorr, backgroundToPhotometricRatioHandle)
            skyFrame = self._getSkyFrame(skyFrameHandle)
            skyBkg = self.sky.exposureToBackground(skyFrame)
            del skyFrame  # Free up memory
            skyBkgs.append(skyBkg)
            # Return a tuple of gridded image and sky frame clipped means
            samples = self.sky.measureScale(calExp.getMaskedImage(), skyBkg)
            scales.append(samples)
        scale = self.sky.solveScales(scales)
        for skyCorr, skyBkg in zip(skyCorrs, skyBkgs):
            bgData = list(skyBkg[0])
            bg = bgData[0]
            statsImage = bg.getStatsImage().clone()
            statsImage *= scale
            newBg = BackgroundMI(bg.getImageBBox(), statsImage)
            newBgData = [newBg] + bgData[1:]
            skyCorr.append(newBgData)
        self.log.info("Sky frame subtracted with a scale factor of %.5f", scale)
        return scale
