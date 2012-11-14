#!/usr/bin/env python

import numpy

from lsst.pex.config import Config, ConfigField, ConfigurableField, Field, makeRegistry
from lsst.pipe.base import Task, Struct
import lsst.afw.math as afwMath
import lsst.afw.geom as afwGeom
import lsst.afw.detection as afwDet
import lsst.afw.image as afwImage
import lsst.meas.algorithms as measAlg

class DetrendStatsConfig(Config):
    stat = Field(doc="Statistic to use to estimate background (from lsst.afw.math)", dtype=int,
                   default=afwMath.MEANCLIP)
    clip = Field(doc="Clipping threshold for background", dtype=float, default=3.0)
    iter = Field(doc="Clipping iterations for background", dtype=int, default=3)

class DetrendStatsTask(Task):
    ConfigClass = DetrendStatsConfig

    def run(self, exposureOrImage):

        stats = afwMath.StatisticsControl(self.config.clip, self.config.iter,
                                          afwImage.MaskU.getPlaneBitMask("DETECTED"))
        try:
            image = exposureOrImage.getMaskedImage()
        except:
            try:
                image = exposureOrImage.getImage()
            except:
                image = exposureOrImage

        return afwMath.makeStatistics(image, self.config.stat, stats).getValue()


class DetrendProcessConfig(Config):
    isr = makeRegistry('''
      Registry of isr tasks for reducing different kinds of detrend products''').makeField('''
      ISR task registry''')
    #isr = ConfigurableField(target=ipIsr.IsrTask, doc="Task for doing ISR")
    detection = ConfigurableField(target=measAlg.SourceDetectionTask, doc="Detection configuration")
    background = ConfigField(dtype=measAlg.BackgroundConfig, doc="Background configuration")
    stats = ConfigurableField(target=DetrendStatsTask, doc="Background statistics configuration")
    doDetection = Field(doc="Do detection on image?", dtype=bool, default=False)

class DetrendProcessTask(Task):
    ConfigClass = DetrendProcessConfig
    def __init__(self, **kwargs):
        super(DetrendProcessTask, self).__init__(**kwargs)
        self.makeSubtask("detection")
        self.makeSubtask("stats")
        self.makeSubtask("isr")

    def run(self, detrendName, isrKey, sensorRef):
        self.log.log(self.log.INFO, "Processing %s" % (sensorRef.dataId))
        exposure = self.isr.run(sensorRef).exposure
        self.display("isr", exposure=exposure, pause=True)

        background = None
        if detrendName.lower() in ("flat", "fringe", "mask"):
            if detrendName.lower() == "mask" or self.config.doDetection:
                footprintSets = self.detect(exposure)

            background = self.stats.run(exposure)

            if detrendName.lower() == "mask":
                return Struct(footprintSets=footprintSets, dim=exposure.getDimensions(), background=background)

            if detrendName.lower() == "fringe":
                # Take the background off again, ignoring detected sources
                self.subtractBackground(exposure)

        return Struct(exposure=exposure, background=background)

    def detect(self, exposure):
        background = self.subtractBackground(exposure)
        footprintSets = self.detection.detectFootprints(exposure)
        mi = exposure.getMaskedImage()
        mi += background                # Restore background
        return footprintSets

    def subtractBackground(self, exposure):
        mi = exposure.getMaskedImage()
        background = measAlg.getBackground(mi, self.config.background).getImageF()
        mi -= background
        return background


class DetrendScaleConfig(Config):
    iterations = Field(doc="Number of iterations", dtype=int, default=10)

class DetrendScaleTask(Task):
    ConfigClass = DetrendScaleConfig

    def run(self, bgMatrix):
        """Determine scaling for flat-fields

        @param bgMatrix Background values provided as a numpy matrix, bgMatrix[component][exposure]
        @return Relative scales for each component, Scales for each exposure
        """
        if numpy.any(numpy.isnan(bgMatrix)):
            raise RuntimeError("Non-finite background values: %s" % numpy.where(numpy.isnan(bgMatrix)))

        components, exposures = numpy.shape(bgMatrix)
        self.log.log(self.log.DEBUG, "Input backgrounds: %s" % bgMatrix)

        # Flat-field scaling
        bgMatrix = numpy.log(bgMatrix)      # log(Background) for each exposure/component
        compScales = numpy.zeros(components) # Initial guess at log(scale) for each component
        expScales = numpy.apply_along_axis(lambda x: numpy.average(x - compScales), 0, bgMatrix)

        for iterate in range(self.config.iterations):
            # XXX use masks for each quantity: maskedarrays
            compScales = numpy.apply_along_axis(lambda x: numpy.average(x - expScales), 1, bgMatrix)
            expScales = numpy.apply_along_axis(lambda x: numpy.average(x - compScales), 0, bgMatrix)
            avgScale = numpy.average(numpy.exp(compScales))
            compScales -= numpy.log(avgScale)
            self.log.log(self.log.DEBUG, "Iteration %d exposure scales: %s" %
                         (iterate, numpy.exp(expScales)))
            self.log.log(self.log.DEBUG, "Iteration %d component scales: %s" %
                         (iterate, numpy.exp(compScales)))

        expScales = numpy.apply_along_axis(lambda x: numpy.average(x - compScales), 0, bgMatrix)

        if numpy.any(numpy.isnan(expScales)):
            raise RuntimeError("Bad exposure scales: %s --> %s" % (bgMatrix, expScales))

        self.log.log(self.log.INFO, "Exposure scales: %s" % (numpy.exp(expScales)))
        self.log.log(self.log.INFO, "Component relative scaling: %s" % (numpy.exp(compScales)))

        return Struct(components=numpy.exp(compScales), exposures=numpy.exp(expScales))




def assertSizes(dimList):
    dim = dimList[0]
    for i, thisDim in enumerate(dimList[1:]):
        if thisDim != dim:
            raise RuntimeError("Bad dimensions for input %d: %s vs %s" % (i+1, thisDim, dim))
    return dim

class DetrendCombineConfig(Config):
    rows = Field(doc="Number of rows to read at a time", dtype=int, default=128)
    maskDetected = Field(doc="Mask pixels about the detection threshold?", dtype=bool, default=True)
    combine = Field(doc="Statistic to use for combination (from lsst.afw.math)", dtype=int,
                 default=afwMath.MEANCLIP)
    clip = Field(doc="Clipping threshold for combination", dtype=float, default=3.0)
    iter = Field(doc="Clipping iterations for combination", dtype=int, default=3)
    stats = ConfigurableField(target=DetrendStatsTask, doc="Background statistics configuration")

class DetrendCombineTask(Task):
    ConfigClass = DetrendCombineConfig

    def __init__(self, *args, **kwargs):
        super(DetrendCombineTask, self).__init__(*args, **kwargs)
        self.makeSubtask("stats")

    def run(self, sensorRefList, expScales=None, finalScale=None):
        # Get sizes
        dimList = []
        for sensorRef in sensorRefList:
            md = sensorRef.get("postISR_md")
            dimList.append(afwGeom.Extent2I(md.get("NAXIS1"), md.get("NAXIS2")))
        dim = assertSizes(dimList)

        maskVal = afwImage.MaskU.getPlaneBitMask("DETECTED") if self.config.maskDetected else 0
        stats = afwMath.StatisticsControl(self.config.clip, self.config.iter, maskVal)

        # Combine images
        combined = afwImage.MaskedImageF(dim)
        numImages = len(sensorRefList)
        imageList = afwImage.vectorMaskedImageF(numImages)
        for start in range(0, dim.getY(), self.config.rows):
            rows = min(self.config.rows, dim.getY() - start)
            box = afwGeom.Box2I(afwGeom.Point2I(0, start), afwGeom.Extent2I(dim.getX(), rows))
            subCombined = combined.Factory(combined, box)

            for i, sensorRef in enumerate(sensorRefList):
                exposure = sensorRef.get("postISR_sub", bbox=box)
                mi = exposure.getMaskedImage()
                if expScales is not None:
                    mi /= expScales[i]
                imageList[i] = mi
            afwMath.statisticsStack(subCombined, imageList, self.config.combine, stats)

        if finalScale is not None:
            background = self.stats.run(combined)
            self.log.log(self.log.INFO, "Measured background of stack is %f; adjusting to %f" %
                         (background, finalScale))
            combined *= finalScale / background

        return combined

class DetrendSimpleCombineConfig(Config):
    maskDetected = Field(doc="Mask pixels about the detection threshold?", dtype=bool, default=True)
    combine = Field(doc="Statistic to use for combination (from lsst.afw.math)", dtype=int,
                 default=afwMath.MEANCLIP)
    clip = Field(doc="Clipping threshold for combination", dtype=float, default=3.0)
    iter = Field(doc="Clipping iterations for combination", dtype=int, default=3)
    stats = ConfigurableField(target=DetrendStatsTask, doc="Background statistics configuration")

class DetrendSimpleCombineTask(Task):
    ConfigClass = DetrendSimpleCombineConfig

    def __init__(self, *args, **kwargs):
        super(DetrendSimpleCombineTask, self).__init__(*args, **kwargs)
        self.makeSubtask("stats")

    def run(self, sensorRefList, expScales=None, finalScale=None):
        # Get sizes
        dimList = []
        for sensorRef in sensorRefList:
            md = sensorRef.get("postISR_md")
            dimList.append(afwGeom.Extent2I(md.get("NAXIS1"), md.get("NAXIS2")))
        dim = assertSizes(dimList)

        maskVal = afwImage.MaskU.getPlaneBitMask("DETECTED") if self.config.maskDetected else 0
        stats = afwMath.StatisticsControl(self.config.clip, self.config.iter, maskVal)

        # Combine images
        combined = afwImage.MaskedImageF(dim)
        numImages = len(sensorRefList)
        imageList = afwImage.vectorMaskedImageF(numImages)
        for i, sensorRef in enumerate(sensorRefList):
            exposure = sensorRef.get("postISR")
            mi = exposure.getMaskedImage()
            mi *= expScales[0]/expScales[i]
            imageList[i] = mi
        afwMath.statisticsStack(combined, imageList, self.config.combine, stats)

        if finalScale is not None:
            combined /= finalScale

        return combined


class MaskCombineConfig(Config):
    maskFraction = Field(doc="Minimum fraction of images where bad pixels got flagged", dtype=float,
                         default=0.5)
    maskPlane = Field(doc="Name of mask plane to set", dtype=str, default="BAD")

class MaskCombineTask(Task):
    ConfigClass = MaskCombineConfig

    def run(footprintSetsList, dimList):
        dim = assertSizes(dimList)

        combined = afwImage.MaskU(dim)
        for footprintSets in footprintSetsList:
            mask = afwImage.MaskU(dim)
            afwImage.setMaskFromFootprintList(mask, footprintSets.positive, 1)
            afwImage.setMaskFromFootprintList(mask, footprintSets.negative, 1)
            combined += mask

        threshold = afwDet.createThreshold(int(self.config.maskFraction * len(footprintSetsList)))
        footprints = afwDet.FootprintSet(combined, threshold)
        mask = combined.addMaskPlane(self.config.maskPlane)
        combined.set(0)
        combined.setMaskFromFootprintList(footprints, mask)

        return combined


class DetrendConfig(Config):
    process = ConfigurableField(target=DetrendProcessTask, doc="Processing configuration")
    scale = ConfigurableField(target=DetrendScaleTask, doc="Scaling configuration")
    combine = ConfigurableField(target=DetrendCombineTask, doc="Detrend combination configuration")
    simpleCombine = ConfigurableField(target=DetrendSimpleCombineTask, doc="Detrend combination configuration")
    mask = ConfigurableField(target=MaskCombineTask, doc="Mask combination configuration")

class DetrendTask(Task):
    ConfigClass = DetrendConfig
    _DefaultName = "detrend"

    def __init__(self, *args, **kwargs):
        super(DetrendTask, self).__init__(**kwargs)
        self.makeSubtask("process")
        self.makeSubtask("scale")
        self.makeSubtask("combine")
        self.makeSubtask("simpleCombine")
        self.makeSubtask("mask")

    def run(self):
        raise NotImplementedError("Not implemented yet.")

