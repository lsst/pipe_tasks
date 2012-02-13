#!/usr/bin/env python

import lsst.pex.config as pexConfig
from lsst.ip.isr import IsrTask
import lsst.pipe.tasks.processCcd as ptProcessCcd
from lsst.pipe.tasks.photometry import PhotometryTask
import hsc.pipe.tasks.astrometry as hscAstrom
import hsc.pipe.tasks.suprimecam as hscSuprimeCam
import hsc.pipe.tasks.calibrate as hscCalibrate
import hsc.pipe.tasks.hscDc2 as hscDc2

class HscProcessCcdConfig(ptProcessCcd.ProcessCcdConfig):
    calibrate = pexConfig.ConfigField(dtype=hscCalibrate.HscCalibrateConfig, doc="Calibration")

class SuprimeCamProcessCcdTask(ptProcessCcd.ProcessCcdTask):
    ConfigClass = HscProcessCcdConfig
    def __init__(self, config=HscProcessCcdConfig(), *args, **kwargs):
        pipeBase.Task.__init__(self, config=config, *args, **kwargs)
        self.makeSubtask("isr", hscSuprimeCam.SuprimeCamIsrTask, config=config.isr)
        self.makeSubtask("calibrate", hscCalibrate.HscCalibrateTask, config=config.calibrate)
        self.makeSubtask("photometry", PhotometryTask, config=config.photometry)

class HscDc2ProcessCcdTask(ptProcessCcd.ProcessCcdTask):
    ConfigClass = HscProcessCcdConfig
    def __init__(self, config=HscProcessCcdConfig(), *args, **kwargs):
        pipeBase.Task.__init__(self, config=config, *args, **kwargs)
        self.makeSubtask("isr", IsrTask, config=config.isr)
        self.makeSubtask("calibrate", hscDc2.HscDc2CalibrateTask, config=config.calibrate)
        self.makeSubtask("photometry", PhotometryTask, config=config.photometry)

