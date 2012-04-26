#!/usr/bin/env python

import gc

from lsst.pex.config import Config, Field, ConfigField
from lsst.pipe.base import Task

import lsst.daf.base as dafBase
import lsst.afw.table as afwTable
import lsst.meas.algorithms as measAlg
import lsst.pipe.tasks.calibrate as ptCal


class ForcedPhotConfig(Config):
    calibrate = ConfigField(dtype=ptCal.CalibrateConfig, doc="Configuration for calibration of stack")
    detection = ConfigField(dtype=measAlg.SourceDetectionConfig, doc="Configuration for detection on stack")
    measurement = ConfigField(dtype=measAlg.SourceMeasurementConfig,
                              doc="Configuration for measurement on CCD")

class ForcedPhotTask(Task):
    ConfigClass = ForcedPhotConfig
    _DefaultName = "forcedPhot"

    def __init__(self, *args, **kwargs):
        super(ForcedPhotTask, self).__init__(*args, **kwargs)
        self.makeSubtask("calibrate", ptCal.CalibrateTask)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("detection", measAlg.SourceDetectionTask, schema=self.schema)
        self.makeSubtask("measurement", measAlg.SourceMeasurementTask,
                         schema=self.schema, algMetadata=self.algMetadata)

    def run(self, butler, stackId, expIdList):
        stack = butler.get("stack", stackId)
        calib = self.calibrate.run(stack)
        stack = calib.exposure
        stackWcs = stack.getWcs()
        
        det = self.detect(stack)
        stackSources = det.sources
        self.measurement.run(stack, stackSources, apCorr=calib.apCorr)
        butler.put(stackSources, "stacksrc", stackId)

        del det
        del calib
        del stack
        gc.collect()

        for expId in expIdList:
            exposure = butler.get("calexp", expId)
            psf = butler.get("psf", expId)
            exposure.setPsf(psf)
            wcs = exposure.getWcs()
            apCorr = butler.get("apCorr", expId)

            sources = stackSources.copy()
            self.measurement.run(exposure, sources, apCorr=apCorr, references=stackSources, refWcs=stackWcs)
            butler.put("src", sources)
            del exposure
            del psf
            del wcs
            del apCorr
            del sources
            gc.collect()


    def detect(self, exposure):
        table = afwTable.SourceTable.make(self.schema)
        table.setMetadata(self.algMetadata)
        return self.detection.makeSourceCatalog(table, exposure)
