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

__all__ = ["SkyCorrectionTask", "SkyCorrectionConfig"]

import warnings

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.pex.config import Config, ConfigField, ConfigurableField, Field
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.tasks.background import (
    FocalPlaneBackground,
    FocalPlaneBackgroundConfig,
    MaskObjectsTask,
    SkyMeasurementTask,
)
from lsst.pipe.tasks.visualizeVisit import VisualizeMosaicExpConfig, VisualizeMosaicExpTask


def _skyFrameLookup(datasetType, registry, quantumDataId, collections):
    """Lookup function to identify sky frames.

    Parameters
    ----------
    datasetType : `lsst.daf.butler.DatasetType`
        Dataset to lookup.
    registry : `lsst.daf.butler.Registry`
        Butler registry to query.
    quantumDataId : `lsst.daf.butler.DataCoordinate`
        Data id to transform to find sky frames.
        The ``detector`` entry will be stripped.
    collections : `lsst.daf.butler.CollectionSearch`
        Collections to search through.

    Returns
    -------
    results : `list` [`lsst.daf.butler.DatasetRef`]
        List of datasets that will be used as sky calibration frames.
    """
    newDataId = quantumDataId.subset(registry.dimensions.conform(["instrument", "visit"]))
    skyFrames = []
    for dataId in registry.queryDataIds(["visit", "detector"], dataId=newDataId).expanded():
        skyFrame = registry.findDataset(
            datasetType, dataId, collections=collections, timespan=dataId.timespan
        )
        skyFrames.append(skyFrame)
    return skyFrames


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
    rawLinker = cT.Input(
        doc="Raw data to provide exp-visit linkage to connect calExp inputs to camera/sky calibs.",
        name="raw",
        multiple=True,
        deferLoad=True,
        storageClass="Exposure",
        dimensions=["instrument", "exposure", "detector"],
    )
    calExps = cT.Input(
        doc="Background-subtracted calibrated exposures.",
        name="calexp",
        multiple=True,
        storageClass="ExposureF",
        dimensions=["instrument", "visit", "detector"],
    )
    calBkgs = cT.Input(
        doc="Subtracted backgrounds for input calibrated exposures.",
        multiple=True,
        name="calexpBackground",
        storageClass="Background",
        dimensions=["instrument", "visit", "detector"],
    )
    backgroundToPhotometricRatioHandles = cT.Input(
        doc="Ratio of a background-flattened image to a photometric-flattened image. "
            "Only used if doApplyFlatBackgroundRatio is True.",
        multiple=True,
        name="background_to_photometric_ratio",
        storageClass="Image",
        dimensions=["instrument", "visit", "detector"],
        deferLoad=True,
    )
    skyFrames = cT.PrerequisiteInput(
        doc="Calibration sky frames.",
        name="sky",
        multiple=True,
        storageClass="ExposureF",
        dimensions=["instrument", "physical_filter", "detector"],
        isCalibration=True,
        lookupFunction=_skyFrameLookup,
    )
    camera = cT.PrerequisiteInput(
        doc="Input camera.",
        name="camera",
        storageClass="Camera",
        dimensions=["instrument"],
        isCalibration=True,
    )
    skyCorr = cT.Output(
        doc="Sky correction data, to be subtracted from the calibrated exposures.",
        name="skyCorr",
        multiple=True,
        storageClass="Background",
        dimensions=["instrument", "visit", "detector"],
    )
    calExpMosaic = cT.Output(
        doc="Full focal plane mosaicked image of the sky corrected calibrated exposures.",
        name="calexp_skyCorr_visit_mosaic",
        storageClass="ImageF",
        dimensions=["instrument", "visit"],
    )
    calBkgMosaic = cT.Output(
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
        self.bgModel2.minFrac = 0.5
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
        # Sort the calExps, calBkgs and skyFrames inputRefs and the
        # skyCorr outputRef by detector ID to ensure reproducibility.
        detectorOrder = [ref.dataId["detector"] for ref in inputRefs.calExps]
        detectorOrder.sort()
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
        inputs.pop("rawLinker", None)
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
        backgroundToPhotometricRatioHandles : `list` [`lsst.daf.butler.DeferredDatasetHandle`], optional
            Deferred dataset handles pointing to the Background to photometric ratio images
            for the input detectors.

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
        if self.config.doApplyFlatBackgroundRatio:
            if not backgroundToPhotometricRatioHandles:
                raise ValueError(
                    "A list of backgroundToPhotometricRatioHandles must be supplied if "
                    "config.doApplyFlatBackgroundRatio=True.",
                )
            # Convert from photometric flattened images to background flattened
            # images.
            for calExp, ratioHandle in zip(calExps, backgroundToPhotometricRatioHandles):
                ratioImage = ratioHandle.get()
                calExp.maskedImage *= ratioImage

        # Restore original backgrounds in-place; optionally refine mask maps
        numOrigBkgElements = [len(calBkg) for calBkg in calBkgs]
        _ = self._restoreOriginalBackgroundRefineMask(calExps, calBkgs)

        # Bin exposures, generate full-fp bg, map to CCDs and subtract in-place
        _ = self._subtractVisitBackground(calExps, calBkgs, camera, self.config.bgModel1)
        initialBackgroundIndex = len(calBkgs[0]._backgrounds) - 1

        # Subtract a scaled sky frame from all input exposures
        skyFrameScale = None
        if self.config.doSky:
            skyFrameScale = self._subtractSkyFrame(calExps, skyFrames, calBkgs)

        # Adds full-fp bg back onto exposures, removes it from list
        if self.config.undoBgModel1:
            _ = self._undoInitialBackground(calExps, calBkgs, initialBackgroundIndex)

        # Bin exposures, generate full-fp bg, map to CCDs and subtract in-place
        if self.config.doBgModel2:
            _ = self._subtractVisitBackground(calExps, calBkgs, camera, self.config.bgModel2)

        # Make camera-level images of bg subtracted calexps and subtracted bgs
        calExpIds = [exp.getDetector().getId() for exp in calExps]
        skyCorrExtras = []
        for calBkg, num in zip(calBkgs, numOrigBkgElements):
            skyCorrExtra = calBkg.clone()
            skyCorrExtra._backgrounds = skyCorrExtra._backgrounds[num:]
            skyCorrExtras.append(skyCorrExtra)
        calExpMosaic = self._binAndMosaic(calExps, camera, self.config.binning, ids=calExpIds, refExps=None)
        calBkgMosaic = self._binAndMosaic(
            skyCorrExtras, camera, self.config.binning, ids=calExpIds, refExps=calExps
        )

        if self.config.doApplyFlatBackgroundRatio:
            # Convert from background flattened images to photometric flattened
            # images.
            for calExp, ratioHandle in zip(calExps, backgroundToPhotometricRatioHandles):
                ratioImage = ratioHandle.get()
                calExp.maskedImage /= ratioImage

        return Struct(
            skyFrameScale=skyFrameScale, skyCorr=calBkgs, calExpMosaic=calExpMosaic, calBkgMosaic=calBkgMosaic
        )

    def _restoreOriginalBackgroundRefineMask(self, calExps, calBkgs):
        """Restore original background to each calexp and invert the related
        background model; optionally refine the mask plane.

        The original visit-level background is restored to each calibrated
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
        calExps : `lsst.afw.image.ExposureF`
            Detector level calexp images to process.
        calBkgs : `lsst.afw.math.BackgroundList`
            Detector level background lists associated with the calexps.

        Returns
        -------
        calExps : `lsst.afw.image.ExposureF`
            The calexps with the originally subtracted background restored.
        skyCorrBases : `lsst.afw.math.BackgroundList`
            The inverted original background models; the genesis for skyCorrs.
        """
        skyCorrBases = []
        for calExp, calBkg in zip(calExps, calBkgs):
            image = calExp.getMaskedImage()

            # Invert all elements of the existing bg model; restore in calexp
            for calBkgElement in calBkg:
                statsImage = calBkgElement[0].getStatsImage()
                statsImage *= -1
            skyCorrBase = calBkg.getImage()
            image -= skyCorrBase

            # Iteratively subtract bg, re-detect sources, and add bg back on
            if self.config.doMaskObjects:
                self.maskObjects.findObjects(calExp)

            stats = np.nanpercentile(skyCorrBase.array, [50, 75, 25])
            self.log.info(
                "Detector %d: Original background restored; BG median = %.1f counts, BG IQR = %.1f counts",
                calExp.getDetector().getId(),
                -stats[0],
                np.subtract(*stats[1:]),
            )
            skyCorrBases.append(skyCorrBase)
        return calExps, skyCorrBases

    def _undoInitialBackground(self, calExps, calBkgs, initialBackgroundIndex):
        """Undo the initial background subtraction (bgModel1) after sky frame
        subtraction.

        Parameters
        ----------
        calExps : `list` [`lsst.afw.image.ExposureF`]
            Calibrated exposures to be background subtracted.
        calBkgs : `list` [`lsst.afw.math.BackgroundList`]
            Background lists associated with the input calibrated exposures.
        initialBackgroundIndex : `int`
            Index of the initial background (bgModel1) in the background list.

        Notes
        -----
        Inputs are modified in-place.
        """
        for calExp, calBkg in zip(calExps, calBkgs):
            image = calExp.getMaskedImage()

            # Remove bgModel1 from the background list; restore in the image
            initialBackground = calBkg[initialBackgroundIndex][0].getImageF()
            image += initialBackground
            calBkg._backgrounds.pop(initialBackgroundIndex)

            self.log.info(
                "Detector %d: The initial background model prior to sky frame subtraction (bgModel1) has "
                "been removed from the background list",
                calExp.getDetector().getId(),
            )

    def _subtractVisitBackground(self, calExps, calBkgs, camera, config):
        """Perform a full focal-plane background subtraction for a visit.

        Generate a full focal plane background model, binning all masked
        detectors into bins of [bgModelN.xSize, bgModelN.ySize]. After,
        subtract the resultant background model (translated back into CCD
        coordinates) from the original detector exposure.

        Return a list of background subtracted images and a list of full focal
        plane background parameters.

        Parameters
        ----------
        calExps : `list` [`lsst.afw.image.ExposureF`]
            Calibrated exposures to be background subtracted.
        calBkgs : `list` [`lsst.afw.math.BackgroundList`]
            Background lists associated with the input calibrated exposures.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera description.
        config : `lsst.pipe.tasks.background.FocalPlaneBackgroundConfig`
            Configuration to use for background subtraction.

        Returns
        -------
        calExps : `list` [`lsst.afw.image.maskedImage.MaskedImageF`]
            Background subtracted exposures for creating a focal plane image.
        calBkgs : `list` [`lsst.afw.math.BackgroundList`]
            Updated background lists with a visit-level model appended.
        """
        # Set up empty full focal plane background model object
        bgModelBase = FocalPlaneBackground.fromCamera(config, camera)

        # Loop over each detector, bin into [xSize, ySize] bins, and update
        # summed flux (_values) and number of contributing pixels (_numbers)
        # in focal plane coordinates. Append outputs to bgModels.
        bgModels = []
        for calExp in calExps:
            bgModel = bgModelBase.clone()
            bgModel.addCcd(calExp)
            bgModels.append(bgModel)

        # Merge detector models to make a single full focal plane bg model
        for bgModel, calExp in zip(bgModels, calExps):
            msg = (
                "Detector %d: Merging %d unmasked pixels (%.1f%s of detector area) into focal plane "
                "background model"
            )
            self.log.debug(
                msg,
                calExp.getDetector().getId(),
                bgModel._numbers.getArray().sum(),
                100 * bgModel._numbers.getArray().sum() / calExp.getBBox().getArea(),
                "%",
            )
            bgModelBase.merge(bgModel)

        # Map full focal plane bg solution to detector; subtract from exposure
        calBkgElements = []
        for calExp in calExps:
            _, calBkgElement = self._subtractDetectorBackground(calExp, bgModelBase)
            calBkgElements.append(calBkgElement)

        msg = (
            "Focal plane background model constructed using %.2f x %.2f mm (%d x %d pixel) superpixels; "
            "FP BG median = %.1f counts, FP BG IQR = %.1f counts"
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"invalid value encountered")
            stats = np.nanpercentile(bgModelBase.getStatsImage().array, [50, 75, 25])
        self.log.info(
            msg,
            config.xSize,
            config.ySize,
            int(config.xSize / config.pixelSize),
            int(config.ySize / config.pixelSize),
            stats[0],
            np.subtract(*stats[1:]),
        )

        for calBkg, calBkgElement in zip(calBkgs, calBkgElements):
            calBkg.append(calBkgElement[0])
        return calExps, calBkgs

    def _subtractDetectorBackground(self, calExp, bgModel):
        """Generate CCD background model and subtract from image.

        Translate the full focal plane background into CCD coordinates and
        subtract from the original science exposure image.

        Parameters
        ----------
        calExp : `lsst.afw.image.ExposureF`
            Exposure to subtract the background model from.
        bgModel : `lsst.pipe.tasks.background.FocalPlaneBackground`
            Full focal plane camera-level background model.

        Returns
        -------
        calExp : `lsst.afw.image.ExposureF`
            Background subtracted input exposure.
        calBkgElement : `lsst.afw.math.BackgroundList`
            Detector level realization of the full focal plane bg model.
        """
        image = calExp.getMaskedImage()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"invalid value encountered")
            calBkgElement = bgModel.toCcdBackground(calExp.getDetector(), image.getBBox())
        image -= calBkgElement.getImage()
        return calExp, calBkgElement

    def _subtractSkyFrame(self, calExps, skyFrames, calBkgs):
        """Determine the full focal plane sky frame scale factor relative to
        an input list of calibrated exposures and subtract.

        This method measures the sky frame scale on all inputs, resulting in
        values equal to the background method solveScales(). The sky frame is
        then subtracted as in subtractSkyFrame() using the appropriate scale.

        Input calExps and calBkgs are updated in-place, returning sky frame
        subtracted calExps and sky frame updated calBkgs, respectively.

        Parameters
        ----------
        calExps : `list` [`lsst.afw.image.ExposureF`]
            Calibrated exposures to be background subtracted.
        skyFrames : `list` [`lsst.afw.image.ExposureF`]
            Sky frame calibration data for the input detectors.
        calBkgs : `list` [`lsst.afw.math.BackgroundList`]
            Background lists associated with the input calibrated exposures.

        Returns
        -------
        scale : `float`
            Scale factor applied to the sky frame.
        """
        skyFrameBgModels = []
        scales = []
        for calExp, skyFrame in zip(calExps, skyFrames):
            skyFrameBgModel = self.sky.exposureToBackground(skyFrame)
            skyFrameBgModels.append(skyFrameBgModel)
            # return a tuple of gridded image and sky frame clipped means
            samples = self.sky.measureScale(calExp.getMaskedImage(), skyFrameBgModel)
            scales.append(samples)
        scale = self.sky.solveScales(scales)
        for calExp, skyFrameBgModel, calBkg in zip(calExps, skyFrameBgModels, calBkgs):
            # subtract the scaled sky frame model from each calExp in-place,
            # also updating the calBkg list in-place
            self.sky.subtractSkyFrame(calExp.getMaskedImage(), skyFrameBgModel, scale, calBkg)
        self.log.info("Sky frame subtracted with a scale factor of %.5f", scale)
        return scale

    def _binAndMosaic(self, exposures, camera, binning, ids=None, refExps=None):
        """Bin input exposures and mosaic across the entire focal plane.

        Input exposures are binned and then mosaicked at the position of
        the detector in the focal plane of the camera.

        Parameters
        ----------
        exposures : `list`
            Detector level list of either calexp `ExposureF` types or
            calexpBackground `BackgroundList` types.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera matching the input data to process.
        binning : `int`
            Binning size to be applied to input images.
        ids : `list` [`int`], optional
            List of detector ids to iterate over.
        refExps : `list` [`lsst.afw.image.ExposureF`], optional
            If supplied, mask planes from these reference images will be used.
        Returns
        -------
        mosaicImage : `lsst.afw.image.ExposureF`
            Mosaicked full focal plane image.
        """
        refExps = np.resize(refExps, len(exposures))  # type: ignore
        binnedImages = []
        for exp, refExp in zip(exposures, refExps):
            try:
                nativeImage = exp.getMaskedImage()
            except AttributeError:
                nativeImage = afwImage.makeMaskedImage(exp.getImage())
                if refExp:
                    nativeImage.setMask(refExp.getMask())
            binnedImage = afwMath.binImage(nativeImage, binning)
            binnedImages.append(binnedImage)
        mosConfig = VisualizeMosaicExpConfig()
        mosConfig.binning = binning
        mosTask = VisualizeMosaicExpTask(config=mosConfig)
        imageStruct = mosTask.run(binnedImages, camera, inputIds=ids)
        mosaicImage = imageStruct.outputData
        return mosaicImage
