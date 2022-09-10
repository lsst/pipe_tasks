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

import numpy as np

import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase

from lsst.afw.cameraGeom.utils import makeImageFromCamera
from lsst.daf.butler import DimensionGraph
from lsst.pex.config import Config, Field, ConfigurableField, ConfigField
import lsst.pipe.base.connectionTypes as cT

from .background import (SkyMeasurementTask, FocalPlaneBackground,
                         FocalPlaneBackgroundConfig, MaskObjectsTask)


def reorderAndPadList(inputList, inputKeys, outputKeys, padWith=None):
    """Match the order of one list to another, padding if necessary

    Parameters
    ----------
    inputList : list
        List to be reordered and padded. Elements can be any type.
    inputKeys :  iterable
        Iterable of values to be compared with outputKeys.
        Length must match `inputList`
    outputKeys : iterable
        Iterable of values to be compared with inputKeys.
    padWith :
        Any value to be inserted where inputKey not in outputKeys

    Returns
    -------
    list
        Copy of inputList reordered per outputKeys and padded with `padWith`
        so that the length matches length of outputKeys.
    """
    outputList = []
    for d in outputKeys:
        if d in inputKeys:
            outputList.append(inputList[inputKeys.index(d)])
        else:
            outputList.append(padWith)
    return outputList


def _makeCameraImage(camera, exposures, binning):
    """Make and write an image of an entire focal plane

    Parameters
    ----------
    camera : `lsst.afw.cameraGeom.Camera`
        Camera description.
    exposures : `dict` mapping detector ID to `lsst.afw.image.Exposure`
        CCD exposures, binned by `binning`.
    binning : `int`
        Binning size that has been applied to images.
    """
    class ImageSource:
        """Source of images for makeImageFromCamera"""
        def __init__(self, exposures):
            """Constructor

            Parameters
            ----------
            exposures : `dict` mapping detector ID to `lsst.afw.image.Exposure`
                CCD exposures, already binned.
            """
            self.isTrimmed = True
            self.exposures = exposures
            self.background = np.nan

        def getCcdImage(self, detector, imageFactory, binSize):
            """Provide image of CCD to makeImageFromCamera"""
            detId = detector.getId()
            if detId not in self.exposures:
                dims = detector.getBBox().getDimensions()/binSize
                image = imageFactory(*[int(xx) for xx in dims])
                image.set(self.background)
            else:
                image = self.exposures[detector.getId()]
            if hasattr(image, "getMaskedImage"):
                image = image.getMaskedImage()
            if hasattr(image, "getMask"):
                mask = image.getMask()
                isBad = mask.getArray() & mask.getPlaneBitMask("NO_DATA") > 0
                image = image.clone()
                image.getImage().getArray()[isBad] = self.background
            if hasattr(image, "getImage"):
                image = image.getImage()

            image = afwMath.rotateImageBy90(image, detector.getOrientation().getNQuarter())

            return image, detector

    image = makeImageFromCamera(
        camera,
        imageSource=ImageSource(exposures),
        imageFactory=afwImage.ImageF,
        binSize=binning
    )
    return image


def makeCameraImage(camera, exposures, filename=None, binning=8):
    """Make and write an image of an entire focal plane

    Parameters
    ----------
    camera : `lsst.afw.cameraGeom.Camera`
        Camera description.
    exposures : `list` of `tuple` of `int` and `lsst.afw.image.Exposure`
        List of detector ID and CCD exposures (binned by `binning`).
    filename : `str`, optional
        Output filename.
    binning : `int`
        Binning size that has been applied to images.
    """
    image = _makeCameraImage(camera, dict(exp for exp in exposures if exp is not None), binning)
    if filename is not None:
        image.writeFits(filename)
    return image


def _skyLookup(datasetType, registry, quantumDataId, collections):
    """Lookup function to identify sky frames

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
        List of datasets that will be used as sky calibration frames
    """
    newDataId = quantumDataId.subset(DimensionGraph(registry.dimensions, names=["instrument", "visit"]))
    skyFrames = []
    for dataId in registry.queryDataIds(["visit", "detector"], dataId=newDataId).expanded():
        skyFrame = registry.findDataset(datasetType, dataId, collections=collections,
                                        timespan=dataId.timespan)
        skyFrames.append(skyFrame)

    return skyFrames


class SkyCorrectionConnections(pipeBase.PipelineTaskConnections, dimensions=("instrument", "visit")):
    rawLinker = cT.Input(
        doc="Raw data to provide exp-visit linkage to connect calExp inputs to camera/sky calibs.",
        name="raw",
        multiple=True,
        deferLoad=True,
        storageClass="Exposure",
        dimensions=["instrument", "exposure", "detector"],
    )
    calExpArray = cT.Input(
        doc="Input exposures to process",
        name="calexp",
        multiple=True,
        storageClass="ExposureF",
        dimensions=["instrument", "visit", "detector"],
    )
    calBkgArray = cT.Input(
        doc="Input background files to use",
        multiple=True,
        name="calexpBackground",
        storageClass="Background",
        dimensions=["instrument", "visit", "detector"],
    )
    camera = cT.PrerequisiteInput(
        doc="Input camera to use.",
        name="camera",
        storageClass="Camera",
        dimensions=["instrument"],
        isCalibration=True,
    )
    skyCalibs = cT.PrerequisiteInput(
        doc="Input sky calibrations to use.",
        name="sky",
        multiple=True,
        storageClass="ExposureF",
        dimensions=["instrument", "physical_filter", "detector"],
        isCalibration=True,
        lookupFunction=_skyLookup,
    )
    calExpCamera = cT.Output(
        doc="Output camera image.",
        name='calexp_camera',
        storageClass="ImageF",
        dimensions=["instrument", "visit"],
    )
    skyCorr = cT.Output(
        doc="Output sky corrected images.",
        name='skyCorr',
        multiple=True,
        storageClass="Background",
        dimensions=["instrument", "visit", "detector"],
    )


class SkyCorrectionConfig(pipeBase.PipelineTaskConfig, pipelineConnections=SkyCorrectionConnections):
    """Configuration for SkyCorrectionTask"""
    bgModel = ConfigField(dtype=FocalPlaneBackgroundConfig, doc="Background model")
    bgModel2 = ConfigField(dtype=FocalPlaneBackgroundConfig, doc="2nd Background model")
    sky = ConfigurableField(target=SkyMeasurementTask, doc="Sky measurement")
    maskObjects = ConfigurableField(target=MaskObjectsTask, doc="Mask Objects")
    doMaskObjects = Field(dtype=bool, default=True, doc="Mask objects to find good sky?")
    doBgModel = Field(dtype=bool, default=True, doc="Do background model subtraction?")
    doBgModel2 = Field(dtype=bool, default=True, doc="Do cleanup background model subtraction?")
    doSky = Field(dtype=bool, default=True, doc="Do sky frame subtraction?")
    binning = Field(dtype=int, default=8, doc="Binning factor for constructing focal-plane images")
    calexpType = Field(dtype=str, default="calexp",
                       doc="Should be set to fakes_calexp if you want to process calexps with fakes in.")

    def setDefaults(self):
        Config.setDefaults(self)
        self.bgModel2.doSmooth = True
        self.bgModel2.minFrac = 0.5
        self.bgModel2.xSize = 256
        self.bgModel2.ySize = 256
        self.bgModel2.smoothScale = 1.0


class SkyCorrectionTask(pipeBase.PipelineTask):
    """Correct sky over entire focal plane"""
    ConfigClass = SkyCorrectionConfig
    _DefaultName = "skyCorr"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):

        # Reorder the skyCalibs, calBkgArray, and calExpArray inputRefs and the
        # skyCorr outputRef sorted by detector id to ensure reproducibility.
        detectorOrder = [ref.dataId['detector'] for ref in inputRefs.calExpArray]
        detectorOrder.sort()
        inputRefs.calExpArray = reorderAndPadList(inputRefs.calExpArray,
                                                  [ref.dataId['detector'] for ref in inputRefs.calExpArray],
                                                  detectorOrder)
        inputRefs.skyCalibs = reorderAndPadList(inputRefs.skyCalibs,
                                                [ref.dataId['detector'] for ref in inputRefs.skyCalibs],
                                                detectorOrder)
        inputRefs.calBkgArray = reorderAndPadList(inputRefs.calBkgArray,
                                                  [ref.dataId['detector'] for ref in inputRefs.calBkgArray],
                                                  detectorOrder)
        outputRefs.skyCorr = reorderAndPadList(outputRefs.skyCorr,
                                               [ref.dataId['detector'] for ref in outputRefs.skyCorr],
                                               detectorOrder)
        inputs = butlerQC.get(inputRefs)
        inputs.pop("rawLinker", None)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        self.makeSubtask("sky")
        self.makeSubtask("maskObjects")

    def focalPlaneBackgroundRun(self, camera, cacheExposures, idList, config):
        """Perform full focal-plane background subtraction

        This method runs on the master node.

        Parameters
        ----------
        camera : `lsst.afw.cameraGeom.Camera`
            Camera description.
        cacheExposures : `list` of `lsst.afw.image.Exposures`
            List of loaded and processed input calExp.
        idList : `list` of `int`
            List of detector ids to iterate over.
        config : `lsst.pipe.drivers.background.FocalPlaneBackgroundConfig`
            Configuration to use for background subtraction.

        Returns
        -------
        exposures : `list` of `lsst.afw.image.Image`
            List of binned images, for creating focal plane image.
        newCacheBgList : `list` of `lsst.afwMath.backgroundList`
            Background lists generated.
        cacheBgModel : `FocalPlaneBackground`
            Full focal plane background model.
        """
        bgModel = FocalPlaneBackground.fromCamera(config, camera)
        data = [pipeBase.Struct(id=id, bgModel=bgModel.clone()) for id in idList]

        bgModelList = []
        for nodeData, cacheExp in zip(data, cacheExposures):
            nodeData.bgModel.addCcd(cacheExp)
            bgModelList.append(nodeData.bgModel)

        for ii, bg in enumerate(bgModelList):
            self.log.info("Background %d: %d pixels", ii, bg._numbers.getArray().sum())
            bgModel.merge(bg)

        exposures = []
        newCacheBgList = []
        cacheBgModel = []
        for cacheExp in cacheExposures:
            nodeExp, nodeBgModel, nodeBgList = self.subtractModelRun(cacheExp, bgModel)
            exposures.append(afwMath.binImage(nodeExp.getMaskedImage(), self.config.binning))
            cacheBgModel.append(nodeBgModel)
            newCacheBgList.append(nodeBgList)

        return exposures, newCacheBgList, cacheBgModel

    def run(self, calExpArray, calBkgArray, skyCalibs, camera):
        """Performa sky correction on an exposure.

        Parameters
        ----------
        calExpArray : `list` of `lsst.afw.image.Exposure`
            Array of detector input calExp images for the exposure to
            process.
        calBkgArray : `list` of `lsst.afw.math.BackgroundList`
            Array of detector input background lists matching the
            calExps to process.
        skyCalibs : `list` of `lsst.afw.image.Exposure`
            Array of SKY calibrations for the input detectors to be
            processed.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera matching the input data to process.

        Returns
        -------
        results : `pipeBase.Struct` containing
            calExpCamera : `lsst.afw.image.Exposure`
                Full camera image of the sky-corrected data.
            skyCorr : `list` of `lsst.afw.math.BackgroundList`
                Detector-level sky-corrected background lists.
        """
        # To allow SkyCorrectionTask to run in the Gen3 butler
        # environment, a new run() method was added that performs the
        # same operations in a serial environment (pipetask processing
        # does not support MPI processing as of 2019-05-03). Methods
        # used in runDataRef() are used as appropriate in run(), but
        # some have been rewritten in serial form. Please ensure that
        # any updates to runDataRef() or the methods it calls with
        # pool.mapToPrevious() are duplicated in run() and its
        # methods.
        #
        # Variable names here should match those in runDataRef() as
        # closely as possible. Variables matching data stored in the
        # pool cache have a prefix indicating this.  Variables that
        # would be local to an MPI processing client have a prefix
        # "node".
        idList = [exp.getDetector().getId() for exp in calExpArray]

        # Construct arrays that match the cache in self.runDataRef() after
        # self.loadImage() is map/reduced.
        cacheExposures = []
        cacheBgList = []
        exposures = []
        for calExp, calBgModel in zip(calExpArray, calBkgArray):
            nodeExp, nodeBgList = self.loadImageRun(calExp, calBgModel)
            cacheExposures.append(nodeExp)
            cacheBgList.append(nodeBgList)
            exposures.append(afwMath.binImage(nodeExp.getMaskedImage(), self.config.binning))

        if self.config.doBgModel:
            # Generate focal plane background, updating backgrounds in the "cache".
            exposures, newCacheBgList, cacheBgModel = self.focalPlaneBackgroundRun(
                camera, cacheExposures, idList, self.config.bgModel
            )
            for cacheBg, newBg in zip(cacheBgList, newCacheBgList):
                cacheBg.append(newBg)

        if self.config.doSky:
            # Measure the sky frame scale on all inputs.  Results in
            # values equal to self.measureSkyFrame() and
            # self.sky.solveScales() in runDataRef().
            cacheSky = []
            measScales = []
            for cacheExp, skyCalib in zip(cacheExposures, skyCalibs):
                skyExp = self.sky.exposureToBackground(skyCalib)
                cacheSky.append(skyExp)
                scale = self.sky.measureScale(cacheExp.getMaskedImage(), skyExp)
                measScales.append(scale)

            scale = self.sky.solveScales(measScales)
            self.log.info("Sky frame scale: %s" % (scale, ))

            # Subtract sky frame, as in self.subtractSkyFrame(), with
            # appropriate scale from the "cache".
            exposures = []
            newBgList = []
            for cacheExp, nodeSky, nodeBgList in zip(cacheExposures, cacheSky, cacheBgList):
                self.sky.subtractSkyFrame(cacheExp.getMaskedImage(), nodeSky, scale, nodeBgList)
                exposures.append(afwMath.binImage(cacheExp.getMaskedImage(), self.config.binning))

        if self.config.doBgModel2:
            # As above, generate a focal plane background model and
            # update the cache models.
            exposures, newBgList, cacheBgModel = self.focalPlaneBackgroundRun(
                camera, cacheExposures, idList, self.config.bgModel2
            )
            for cacheBg, newBg in zip(cacheBgList, newBgList):
                cacheBg.append(newBg)

        # Generate camera-level image of calexp and return it along
        # with the list of sky corrected background models.
        image = makeCameraImage(camera, zip(idList, exposures))

        return pipeBase.Struct(
            calExpCamera=image,
            skyCorr=cacheBgList,
        )

    def loadImageRun(self, calExp, calExpBkg):
        """Serial implementation of self.loadImage() for Gen3.

        Load and restore background to calExp and calExpBkg.

        Parameters
        ----------
        calExp : `lsst.afw.image.Exposure`
            Detector level calExp image to process.
        calExpBkg : `lsst.afw.math.BackgroundList`
            Detector level background list associated with the calExp.

        Returns
        -------
        calExp : `lsst.afw.image.Exposure`
            Background restored calExp.
        bgList : `lsst.afw.math.BackgroundList`
            New background list containing the restoration background.
        """
        image = calExp.getMaskedImage()

        for bgOld in calExpBkg:
            statsImage = bgOld[0].getStatsImage()
            statsImage *= -1

        image -= calExpBkg.getImage()
        bgList = afwMath.BackgroundList()
        for bgData in calExpBkg:
            bgList.append(bgData)

        if self.config.doMaskObjects:
            self.maskObjects.findObjects(calExp)

        return (calExp, bgList)

    def subtractModelRun(self, exposure, bgModel):
        """Serial implementation of self.subtractModel() for Gen3.

        Load and restore background to calExp and calExpBkg.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to subtract the background model from.
        bgModel : `lsst.pipe.drivers.background.FocalPlaneBackground`
            Full camera level background model.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Background subtracted input exposure.
        bgModelCcd : `lsst.afw.math.BackgroundList`
            Detector level realization of the full background model.
        bgModelMaskedImage : `lsst.afw.image.MaskedImage`
            Background model from the bgModelCcd realization.
        """
        image = exposure.getMaskedImage()
        detector = exposure.getDetector()
        bbox = image.getBBox()
        bgModelCcd = bgModel.toCcdBackground(detector, bbox)
        image -= bgModelCcd.getImage()

        return (exposure, bgModelCcd, bgModelCcd[0])
