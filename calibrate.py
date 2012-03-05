#!/usr/bin/env python

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.calibrate as ptCalibrate
import lsst.meas.algorithms as measAlg
import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import lsst.meas.photocal as photocal
from lsst.pipe.tasks.repair import RepairTask
from lsst.pipe.tasks.measurePsf import MeasurePsfTask
import hsc.pipe.tasks.astrometry as hscAstrom


class HscCalibrateConfig(ptCalibrate.CalibrateConfig):
    astrometry = pexConfig.ConfigField(dtype = hscAstrom.HscAstrometryConfig, doc = "HSC calibration")

class HscCalibrateTask(ptCalibrate.CalibrateTask):
    ConfigClass = HscCalibrateConfig

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("repair", RepairTask)
        self.makeSubtask("detection", measAlg.SourceDetectionTask, schema=self.schema)
        self.makeSubtask("initialMeasurement", measAlg.SourceMeasurementTask,
                         schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask("measurePsf", MeasurePsfTask, schema=self.schema)
        self.makeSubtask("measurement", measAlg.SourceMeasurementTask,
                         schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask("astrometry", hscAstrom.HscAstrometryTask, schema=self.schema)
        self.makeSubtask("photocal", photocal.PhotoCalTask, schema=self.schema)
