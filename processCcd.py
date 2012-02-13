#!/usr/bin/env python

from lsst.ip.isr import IsrTask
from lsst.pipe.tasks.photometry import PhotometryTask

import hsc.pipe.tasks.astrometry as hscAstrom
import hsc.pipe.tasks.suprimecam as hscSuprimeCam
import hsc.pipe.tasks.calibrate as hscCalibrate
import hsc.pipe.tasks.hscDc2 as hscDc2

class HscProcessCcdConfig(ptProcessCcd.ProcessCcdConfig):
    calibrate = pexConfig.ConfigField(dtype=HscCalibrateTask.ConfigClass, doc="Calibration")

class SuprimeCamProcessCcd(ptProcessCcd.ProcessCcd):
    ConfigClass = HscProcessCcdConfig
    def __init__(self, config=HscProcessCcdConfig(), *args, **kwargs):
        pipeBase.Task.__init__(self, config=config, *args, **kwargs)
        self.makeSubtask("isr", hscSuprimeCam.SuprimeCamIsrTask, config=config.isr)
        self.makeSubtask("calibrate", hscCalibrate.HscCalibrateTask, config=config.calibrate)
        self.makeSubtask("photometry", PhotometryTask, config=config.photometry)

class HscDc2ProcessCcd(ptProcessCcd.ProcessCcd):
    ConfigClass = HscProcessCcdConfig
    def __init__(self, config=HscProcessCcdConfig(), *args, **kwargs):
        pipeBase.Task.__init__(self, config=config, *args, **kwargs)
        self.makeSubtask("isr", IsrTask, config=config.isr)
        self.makeSubtask("calibrate", hscDc2.HscDc2CalibrateTask, config=config.calibrate)
        self.makeSubtask("photometry", PhotometryTask, config=config.photometry)

