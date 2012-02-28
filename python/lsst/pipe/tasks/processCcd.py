#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 2008, 2009, 2010 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.detection as afwDet
import lsst.meas.algorithms as measAlg

from lsst.ip.isr import IsrTask
from lsst.pipe.tasks.calibrate import CalibrateTask
from lsst.pipe.tasks.photometry import PhotometryTask

class ProcessCcdConfig(pexConfig.Config):
    """Config for ProcessCcd"""
    doIsr = pexConfig.Field(dtype=bool, default=True, doc = "Perform ISR?")
    doCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Perform calibration?")
    doPhotometry = pexConfig.Field(dtype=bool, default=True, doc = "Perform photometry?")
    doWriteIsr = pexConfig.Field(dtype=bool, default=True, doc = "Write ISR results?")
    doWriteCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Write calibration results?")
    isr = pexConfig.ConfigField(dtype=IsrTask.ConfigClass, doc="Instrumental Signature Removal")
    calibrate = pexConfig.ConfigField(dtype=CalibrateTask.ConfigClass,
                                      doc="Calibration (inc. high-threshold detection and measurement)")
    detection = pexConfig.ConfigField(dtype=measAlg.SourceDetectionTask.ConfigConfig,
                                      doc="Low-threshold detection for final measurement")
    measurement = pexConfig.ConfigField(dtype=measAlg.SourceMeasurementTask.ConfigClass,
                                        doc="Final source measurement on low-threshold detections")

    def __init__(self):
        pexConfig.Config.__init__(self)
        self.doWriteIsr = False
        self.isr.methodList = ["doConversionForIsr", "doSaturationDetection",
                               "doOverscanCorrection", "doVariance", "doFlatCorrection"]
        self.isr.doWrite = False

        # Astrometry; subaru specific, shouldn't end up here once merged with processCcdLsstSim
        self.calibrate.astrometry.distortion.name = "radial"
        self.calibrate.astrometry.distortion["radial"].coefficients \
            = [0.0, 1.0, 7.16417e-08, 3.03146e-10, 5.69338e-14, -6.61572e-18]
        self.calibrate.astrometry.distortion["radial"].observedToCorrected = True

class ProcessCcdTask(pipeBase.Task):
    """Process a CCD"""
    ConfigClass = ProcessCcdConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("isr", IsrTask)
        self.makeSubtask("calibrate", CalibrateTask)
        self.makeSubtask("photometry", PhotometryTask)

    @pipeBase.timeMethod
    def run(self, sensorRef):
        self.log.log(self.log.INFO, "Processing %s" % (sensorRef.dataId))
        if self.config.doIsr:
            butler = sensorRef.butlerSubset.butler
            calibSet = self.isr.makeCalibDict(butler, sensorRef.dataId)
            rawExposure = sensorRef.get("raw")
            isrRes = self.isr.run(rawExposure, calibSet)
            self.display("isr", exposure=isrRes.postIsrExposure, pause=True)
            ccdExposure = self.isr.doCcdAssembly([isrRes.postIsrExposure])
            self.display("ccdAssembly", exposure=ccdExposure)
            if self.config.doWriteIsr:
                sensorRef.put(ccdExposure, 'postISRCCD')
        else:
            ccdExposure = None

        if self.config.doCalibrate:
            if ccdExposure is None:
                ccdExposure = sensorRef.get('postISRCCD')
            calib = self.calibrate.run(ccdExposure)
            ccdExposure = calib.exposure
            if self.config.doWriteCalibrate:
                sensorRef.put(ccdExposure, 'calexp')
                sensorRef.put(afwDet.PersistableSourceVector(calib.sources), 'icSrc')
                if calib.psf is not None:
                    sensorRef.put(calib.psf, 'psf')
                if calib.apCorr is not None:
                    #sensorRef.put(calib.apCorr, 'apcorr')
                    pass
                if calib.matches is not None:
                    sensorRef.put(afwDet.PersistableSourceMatchVector(calib.matches, calib.matchMeta),
                               'icMatch')
        else:
            calib = None

        if self.config.doPhotometry:
            if ccdExposure is None:
                ccdExposure = sensorRef.get('calexp')
            if calib is None:
                psf = sensorRef.get('psf')
                apCorr = None # sensorRef.get('apcorr')
            else:
                psf = calib.psf
                apCorr = calib.apCorr
            phot = self.photometry.run(ccdExposure, psf, apcorr=apCorr)
            if self.config.doWritePhotometry:
                sensorRef.put(afwDet.PersistableSourceVector(phot.sources), 'src')
        else:
            phot = None

        return pipeBase.Struct(
            ccdExposure = isrRes.postIsrExposure if self.config.doIsr else None,
            exposure = ccdExposure,
            psf = psf,
            apCorr = apCorr,
            sources = phot.sources if phot else None,
            matches = calib.matches if calib else None,
            matchMeta = calib.matchMeta if calib else None,
        )
