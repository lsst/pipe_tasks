#!/usr/bin/env python

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.tasks.calibrate as ptCalibrate
from lsst.pipe.tasks.repair import RepairTask
from lsst.pipe.tasks.measurePsf import MeasurePsfTask
from lsst.pipe.tasks.photometry import PhotometryTask, RephotometryTask
import hsc.pipe.tasks.astrometry as hscAstrom


class HscCalibrateConfig(ptCalibrate.CalibrateConfig):
    astrometry = pexConfig.ConfigField(dtype = hscAstrom.HscAstrometryConfig, doc = "HSC astrometry")

class HscCalibrateTask(ptCalibrate.CalibrateTask):
    ConfigClass = HscCalibrateConfig
    def __init__(self, config=HscCalibrateConfig(), *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("repair", RepairTask, config=config.repair)
        self.makeSubtask("photometry", PhotometryTask, config=config.photometry)
        self.makeSubtask("measurePsf", MeasurePsfTask, config=config.measurePsf)
        self.makeSubtask("rephotometry", RephotometryTask, config=config.photometry)
        self.makeSubtask("astrometry", hscAstrom.HscAstrometryTask, config=config.astrometry)


