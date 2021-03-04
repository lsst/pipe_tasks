# This file is part of pipe_drivers.
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
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase

from lsst.pipe.base import ArgumentParser, ConfigDatasetType
from lsst.daf.butler import DimensionGraph
from lsst.pex.config import Config, Field, ConfigurableField, ConfigField
from lsst.ctrl.pool.pool import Pool
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.pipe.drivers.background import (SkyMeasurementTask, FocalPlaneBackground,
                                          FocalPlaneBackgroundConfig, MaskObjectsTask)
import lsst.pipe.drivers.visualizeVisit as visualizeVisit
import lsst.pipe.base.connectionTypes as cT

__all__ = ["SkyCorrectionConfig", "SkyCorrectionTask"]

DEBUG = False  # Debugging outputs?


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
    image = visualizeVisit.makeCameraImage(camera, dict(exp for exp in exposures if exp is not None), binning)
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


class SkyCorrectionTask(pipeBase.PipelineTask, BatchPoolTask):
    """Correct sky over entire focal plane"""
    ConfigClass = SkyCorrectionConfig
    _DefaultName = "skyCorr"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs.pop("rawLinker", None)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        self.makeSubtask("sky")
        self.makeSubtask("maskObjects")

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        datasetType = ConfigDatasetType(name="calexpType")
        parser = ArgumentParser(name="skyCorr", *args, **kwargs)
        parser.add_id_argument("--id", datasetType=datasetType, level="visit",
                               help="data ID, e.g. --id visit=12345")
        return parser

    @classmethod
    def batchWallTime(cls, time, parsedCmd, numCores):
        """Return walltime request for batch job

        Subclasses should override if the walltime should be calculated
        differently (e.g., addition of some serial time).

        Parameters
        ----------
        time : `float`
            Requested time per iteration.
        parsedCmd : `argparse.Namespace`
            Results of argument parsing.
        numCores : `int`
            Number of cores.
        """
        numTargets = len(cls.RunnerClass.getTargetList(parsedCmd))
        return time*numTargets

    def runDataRef(self, expRef):
        """Perform sky correction on an exposure

        We restore the original sky, and remove it again using multiple
        algorithms. We optionally apply:

        1. A large-scale background model.
            This step removes very-large-scale sky such as moonlight.
        2. A sky frame.
        3. A medium-scale background model.
            This step removes residual sky (This is smooth on the focal plane).

        Only the master node executes this method. The data is held on
        the slave nodes, which do all the hard work.

        Parameters
        ----------
        expRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.

        See Also
        --------
        ~lsst.pipe.drivers.SkyCorrectionTask.run
        """
        if DEBUG:
            extension = "-%(visit)d.fits" % expRef.dataId

        with self.logOperation("processing %s" % (expRef.dataId,)):
            pool = Pool()
            pool.cacheClear()
            pool.storeSet(butler=expRef.getButler())
            camera = expRef.get("camera")

            dataIdList = [ccdRef.dataId for ccdRef in expRef.subItems("ccd") if
                          ccdRef.datasetExists(self.config.calexpType)]

            exposures = pool.map(self.loadImage, dataIdList)
            if DEBUG:
                makeCameraImage(camera, exposures, "restored" + extension)
                exposures = pool.mapToPrevious(self.collectOriginal, dataIdList)
                makeCameraImage(camera, exposures, "original" + extension)
                exposures = pool.mapToPrevious(self.collectMask, dataIdList)
                makeCameraImage(camera, exposures, "mask" + extension)

            if self.config.doBgModel:
                exposures = self.focalPlaneBackground(camera, pool, dataIdList, self.config.bgModel)

            if self.config.doSky:
                measScales = pool.mapToPrevious(self.measureSkyFrame, dataIdList)
                scale = self.sky.solveScales(measScales)
                self.log.info("Sky frame scale: %s" % (scale,))

                exposures = pool.mapToPrevious(self.subtractSkyFrame, dataIdList, scale)
                if DEBUG:
                    makeCameraImage(camera, exposures, "skysub" + extension)
                    calibs = pool.mapToPrevious(self.collectSky, dataIdList)
                    makeCameraImage(camera, calibs, "sky" + extension)

            if self.config.doBgModel2:
                exposures = self.focalPlaneBackground(camera, pool, dataIdList, self.config.bgModel2)

            # Persist camera-level image of calexp
            image = makeCameraImage(camera, exposures)
            expRef.put(image, "calexp_camera")

            pool.mapToPrevious(self.write, dataIdList)

    def focalPlaneBackground(self, camera, pool, dataIdList, config):
        """Perform full focal-plane background subtraction

        This method runs on the master node.

        Parameters
        ----------
        camera : `lsst.afw.cameraGeom.Camera`
            Camera description.
        pool : `lsst.ctrl.pool.Pool`
            Process pool.
        dataIdList : iterable of `dict`
            List of data identifiers for the CCDs.
        config : `lsst.pipe.drivers.background.FocalPlaneBackgroundConfig`
            Configuration to use for background subtraction.

        Returns
        -------
        exposures : `list` of `lsst.afw.image.Image`
            List of binned images, for creating focal plane image.
        """
        bgModel = FocalPlaneBackground.fromCamera(config, camera)
        data = [pipeBase.Struct(dataId=dataId, bgModel=bgModel.clone()) for dataId in dataIdList]
        bgModelList = pool.mapToPrevious(self.accumulateModel, data)
        for ii, bg in enumerate(bgModelList):
            self.log.info("Background %d: %d pixels", ii, bg._numbers.array.sum())
            bgModel.merge(bg)
        return pool.mapToPrevious(self.subtractModel, dataIdList, bgModel)

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
        """Duplicate runDataRef method without ctrl_pool for Gen3.

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

        See Also
        --------
        ~lsst.pipe.drivers.SkyCorrectionTask.runDataRef()
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

    def loadImage(self, cache, dataId):
        """Load original image and restore the sky

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Resultant exposure.
        """
        cache.dataId = dataId
        cache.exposure = cache.butler.get(self.config.calexpType, dataId, immediate=True).clone()
        bgOld = cache.butler.get("calexpBackground", dataId, immediate=True)
        image = cache.exposure.getMaskedImage()

        # We're removing the old background, so change the sense of all its components
        for bgData in bgOld:
            statsImage = bgData[0].getStatsImage()
            statsImage *= -1

        image -= bgOld.getImage()
        cache.bgList = afwMath.BackgroundList()
        for bgData in bgOld:
            cache.bgList.append(bgData)

        if self.config.doMaskObjects:
            self.maskObjects.findObjects(cache.exposure)

        return self.collect(cache)

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

    def measureSkyFrame(self, cache, dataId):
        """Measure scale for sky frame

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.

        Returns
        -------
        scale : `float`
            Scale for sky frame.
        """
        assert cache.dataId == dataId
        cache.sky = self.sky.getSkyData(cache.butler, dataId)
        scale = self.sky.measureScale(cache.exposure.getMaskedImage(), cache.sky)
        return scale

    def subtractSkyFrame(self, cache, dataId, scale):
        """Subtract sky frame

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.
        scale : `float`
            Scale for sky frame.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Resultant exposure.
        """
        assert cache.dataId == dataId
        self.sky.subtractSkyFrame(cache.exposure.getMaskedImage(), cache.sky, scale, cache.bgList)
        return self.collect(cache)

    def accumulateModel(self, cache, data):
        """Fit background model for CCD

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        data : `lsst.pipe.base.Struct`
            Data identifier, with `dataId` (data identifier) and `bgModel`
            (background model) elements.

        Returns
        -------
        bgModel : `lsst.pipe.drivers.background.FocalPlaneBackground`
            Background model.
        """
        assert cache.dataId == data.dataId
        data.bgModel.addCcd(cache.exposure)
        return data.bgModel

    def subtractModel(self, cache, dataId, bgModel):
        """Subtract background model

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.
        bgModel : `lsst.pipe.drivers.background.FocalPlaneBackround`
            Background model.

        Returns
        -------
        exposure : `lsst.afw.image.Exposure`
            Resultant exposure.
        """
        assert cache.dataId == dataId
        exposure = cache.exposure
        image = exposure.getMaskedImage()
        detector = exposure.getDetector()
        bbox = image.getBBox()
        try:
            cache.bgModel = bgModel.toCcdBackground(detector, bbox)
            image -= cache.bgModel.getImage()
        except RuntimeError:
            self.log.error(f"There was an error processing {dataId}, no calib file produced")
            return
        cache.bgList.append(cache.bgModel[0])
        return self.collect(cache)

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

    def realiseModel(self, cache, dataId, bgModel):
        """Generate an image of the background model for visualisation

        Useful for debugging.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.
        bgModel : `lsst.pipe.drivers.background.FocalPlaneBackround`
            Background model.

        Returns
        -------
        detId : `int`
            Detector identifier.
        image : `lsst.afw.image.MaskedImage`
            Binned background model image.
        """
        assert cache.dataId == dataId
        exposure = cache.exposure
        detector = exposure.getDetector()
        bbox = exposure.getMaskedImage().getBBox()
        image = bgModel.toCcdBackground(detector, bbox).getImage()
        return self.collectBinnedImage(exposure, image)

    def collectBinnedImage(self, exposure, image):
        """Return the binned image required for visualization

        This method just helps to cut down on boilerplate.

        Parameters
        ----------
        image : `lsst.afw.image.MaskedImage`
            Image to go into visualisation.

        Returns
        -------
        detId : `int`
            Detector identifier.
        image : `lsst.afw.image.MaskedImage`
            Binned image.
        """
        return (exposure.getDetector().getId(), afwMath.binImage(image, self.config.binning))

    def collect(self, cache):
        """Collect exposure for potential visualisation

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.

        Returns
        -------
        detId : `int`
            Detector identifier.
        image : `lsst.afw.image.MaskedImage`
            Binned image.
        """
        return self.collectBinnedImage(cache.exposure, cache.exposure.maskedImage)

    def collectOriginal(self, cache, dataId):
        """Collect original image for visualisation

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.

        Returns
        -------
        detId : `int`
            Detector identifier.
        image : `lsst.afw.image.MaskedImage`
            Binned image.
        """
        exposure = cache.butler.get("calexp", dataId, immediate=True)
        return self.collectBinnedImage(exposure, exposure.maskedImage)

    def collectSky(self, cache, dataId):
        """Collect original image for visualisation

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.

        Returns
        -------
        detId : `int`
            Detector identifier.
        image : `lsst.afw.image.MaskedImage`
            Binned image.
        """
        return self.collectBinnedImage(cache.exposure, cache.sky.getImage())

    def collectMask(self, cache, dataId):
        """Collect mask for visualisation

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.

        Returns
        -------
        detId : `int`
            Detector identifier.
        image : `lsst.afw.image.Image`
            Binned image.
        """
        # Convert Mask to floating-point image, because that's what's required for focal plane construction
        image = afwImage.ImageF(cache.exposure.maskedImage.getBBox())
        image.array[:] = cache.exposure.maskedImage.mask.array
        return self.collectBinnedImage(cache.exposure, image)

    def write(self, cache, dataId):
        """Write resultant background list

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.
        """
        cache.butler.put(cache.bgList, "skyCorr", dataId)

    def _getMetadataName(self):
        """There's no metadata to write out"""
        return None
