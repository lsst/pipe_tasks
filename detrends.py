#!/usr/bin/env python

from lsst.pex.config import Config, ConfigField, Field
from lsst.pipe.base import Task, Struct
import lsst.meas.algorithms as measAlg

#    detrend = ChoiceField(doc="Type of detrend", dtype=str, default="BIAS",
#                          allowed={"BIAS": "Zero second exposure to measure electronic pedestal",
#                                   "DARK": "Long exposure with no illumination to measure dark current",
#                                   "FLAT": "Exposure with uniform illumination to measure gain variation",
#                                   "FRINGE": "Exposure of night sky to measure sky response",
#                                   "MASK": "Exposures with different illumination levels to find bad pixels",
#                                   })

class DetrendStatsConfig(Config):
    stat = Field(doc="Statistic to use to estimate background (from lsst.afw.math)", dtype=int,
                   default=afwMath.MEANCLIP)
    clip = Field(doc="Clipping threshold for background", dtype=float, default=3.0)
    iter = Field(doc="Clipping iterations for background", dtype=int, default=3)
    

class DetrendProcessConfig(Config):
    isr = ConfigField(dtype=isrTask.IsrTaskConfig, doc="ISR configuration")
    detection = ConfigField(dtype=measAlg.SourceDetectionConfig, doc="Detection configuration")
    background = ConfigField(dtype=measAlg.BackgroundConfig, doc="Background configuration")
    stats = ConfigField(dtype=DetrendStatsConfig, doc="Background statistics configuration")

class DetrendScaleConfig(Config):
    iterations = Field(doc="Number of iterations", dtype=int, default=10)

class DetrendCombineConfig(Config):
    rows = Field(doc="Number of rows to read at a time", dtype=int, default=128)
    maskDetected = Field(doc="Mask pixels about the detection threshold?", dtype=bool, default=True)
    combine = Field(doc="Statistic to use for combination (from lsst.afw.math)", dtype=int,
                 default=afwMath.MEANCLIP)
    stats = ConfigField(dtype=DetrendStatsConfig, doc="Background statistics configuration")

class MaskCombineConfig(Config):
    maskFraction = Field(doc="Minimum fraction of images where bad pixels got flagged", dtype=float,
                         default=0.5)
    maskPlane = Field(doc="Name of mask plane to set", dtype=str, default="BAD")

class DetrendConfig(Config):
    process = ConfigField(dtype=DetrendProcessConfig, doc="Processing configuration")
    scale = ConfigField(dtype=DetrendScaleConfig, doc="Scaling configuration")
    combine = ConfigField(dtype=DetrendCombineConfig, doc="Detrend combination configuration")
    mask = ConfigField(dtype=MaskCombineConfig, doc="Mask combination configuration")



class DetrendStatsTask(Task):
    ConfigClass = DetrendStatsConfig

    def run(self, exposure):
        stats = afwMath.StatisticsControl(self.config.stats.clip, self.config.stats.iter,
                                          exposure.getMaskedImage().getMask().getPlaneBitMask("DETECTED"))
        return afwMath.makeStatistics(exposure, self.config.stats.stat, stats).getValue()


class DetrendProcessTask(Task):
    ConfigClass = DetrendProcessConfig
    def __init__(self, **kwargs):
        super(DetrendProcessTask, self).__init__(**kwargs)
        self.makeSubtask("isr", isrTask.IsrTask)
        self.makeSubtask("detection", measAlg.SourceDetectionTask)
        self.makeSubtask("stats", DetrendStatsTask)

    def run(self, detrend, sensorRef):
        self.log.log(self.log.INFO, "Processing %s" % (sensorRef.dataId))

        exposure = self.removeInstrumentSignature(sensorRef)

        background = None
        if detrend.lower() in ("flat", "fringe", "mask"):
            footprintSets = self.detect()

            background = self.stats.run(exposure)

            if detrend.lower() == "mask":
                return Struct(footprintSets=footprintSets, dim=exposure.getDimensions(), background=background)

            if detrend.lower() == "fringe":
                # Take the background off again, ignoring detected sources
                self.subtractBackground(exposure)

        return Struct(exposure=exposure, background=background)

    def removeInstrumentSignature(sensorRef):
        butler = sensorRef.butlerSubset.butler
        calibSet = self.isr.makeCalibDict(butler, sensorRef.dataId)
        exposure = sensorRef.get("raw")
        isrRes = self.isr.run(exposure, calibSet)
        exposure = isrRes.postIsrExposure
        self.display("isr", exposure=exposure, pause=True)
        return exposure

    def detect(exposure):
        background = self.subtractBackground(exposure)
        footprintSets = self.detection.detectFootprints(exposure)
        mi += background                # Restore background
        return footprintSets

    def subtractBackground(exposure):
        mi = exposure.getMaskedImage()
        background = measAlg.getBackground(mi, self.config.background).getImageF()
        mi -= background
        return background


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

        for iterate in range(self.config['scale']['iterate']):
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


class DetrendCombineTask(Task):
    ConfigClass = DetrendCombineConfig

    def __init__(self, *args, **kwargs):
        super(DetrendCombineTask, self).__init__(*args, **kwargs)
        self.makeSubtask("stats", DetrendStatsTask)

    def run(self, sensorRefList, expScales=None, finalScale=None):
        # Get sizes
        dimList = []
        for sensorRef in sensorRefList:
            md = sensorRef.get("calexp_md")
            dimList.append(afwGeom.Extent2I(md.get("NAXIS1"), md.get("NAXIS2")))
        dim = assertSizes(dimList)

        maskVal = afwImage.MaskU.getPlaneBitMask("DETECTED") if self.config.maskDetected else 0
        stats = afwMath.StatisticsControl(numSigmaClip, numIter, maskVal)

        # Combine images
        combined = afwImage.ImageF(dim)
        for start in range(0, dim[2], self.config.rows):
            rows = min(self.config.rows, dim[2] - start)
            box = afwGeom.Box2I(afwGeom.Point2I(0, start), afwGeom.Extent2I(dim[1], rows))
            subCombined = combined.Factory(combined, box)

            imageList = []
            for i, sensorRef in enumerate(sensorRefList):
                imageList.append(sensorRef.get("calexp_sub", llcX=0, llcY=start, width=dim[1], height=rows))
                if expScales is not None:
                    imageList[i] *= expScales[i]

            afwMath.statisticsStack(subCombined, imageList, self.config.combine, stats)

        if finalScale is not None:
            background = self.stats.run(combined)
            combined *= finalScale / background

        return combined


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

class DetrendTask(Task):
    ConfigClass = DetrendConfig

    def __init__(self, *args, **kwargs):
        super(DetrendTask, self).__init__(**kwargs)
        self.makeSubtask("process", DetrendProcessTask)
        self.makeSubtask("scale", DetrendScaleTask)
        self.makeSubtask("combine", DetrendCombineTask)
        self.makeSubtask("mask", MaskCombineTask)

    def run(self):
        raise NotImplementedError("Not implemented yet.")

