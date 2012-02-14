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

from lsst.ip.isr import IsrTask
from lsst.pipe.tasks.calibrate import CalibrateTask
from lsst.pipe.tasks.photometry import PhotometryTask


class ProcessCcdLsstSimConfig(pexConfig.Config):
    """Config for ProcessCcdLsstSim"""
    doIsr = pexConfig.Field(dtype=bool, default=True, doc = "Perform ISR?")
    doCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Perform calibration?")
    doPhotometry = pexConfig.Field(dtype=bool, default=True, doc = "Perform photometry?")
    doWriteCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Write calibration results?")
    doWritePhotometry = pexConfig.Field(dtype=bool, default=True, doc = "Write photometry results?")
    isr = pexConfig.ConfigField(dtype=IsrTask.ConfigClass, doc="Amp-level instrumental signature removal")
    ccdIsr = pexConfig.ConfigField(dtype=IsrTask.ConfigClass, doc="CCD level instrumental signature removal")
    calibrate = pexConfig.ConfigField(dtype=CalibrateTask.ConfigClass, doc="Calibration")
    photometry = pexConfig.ConfigField(dtype=PhotometryTask.ConfigClass, doc="Photometry")

    def __init__(self, *args, **kwargs):
        pexConfig.Config.__init__(self, *args, **kwargs)
        self.isr.doWrite = False # don't persist data until until CCD ISR is run; ignored anyway
        self.ccdIsr.methodList = ['doSaturationInterpolation', 'doMaskAndInterpDefect', 'doMaskAndInterpNan']
        self.ccdIsr.doWrite = False # ProcessCcdLsstSimTask, not IsrTask, persists the data; ignored anyway


class ProcessCcdLsstSimTask(pipeBase.Task):
    """Process a CCD for LSSTSim
    
    @todo: this variant of ProcessCcdTask can be eliminated once IsrTask is unified.
    """
    ConfigClass = ProcessCcdLsstSimConfig

    def __init__(self, config=ProcessCcdLsstSimConfig(), *args, **kwargs):
        pipeBase.Task.__init__(self, config=config, *args, **kwargs)
        self.makeSubtask("isr", IsrTask, config=config.isr)
        self.makeSubtask("ccdIsr", IsrTask, config=config.ccdIsr)
        self.makeSubtask("calibrate", CalibrateTask, config=config.calibrate)
        self.makeSubtask("photometry", PhotometryTask, config=config.photometry)

    @pipeBase.timeMethod
    def run(self, sensorRef):
        """Run task; do not persist results.
        """
        self.log.log(self.log.INFO, "Processing %s" % (sensorRef.dataId))
        butler = sensorRef.butlerSubset.butler
        if self.config.doIsr:
            # perform amp-level ISR
            exposureList = list()
            for ampRef in sensorRef.subItems():
                calibSet = self.isr.makeCalibDict(butler, ampRef.dataId)
                ampExposure = ampRef.get("raw")
                isrRes = self.isr.run(ampExposure, calibSet)
                exposureList.append(isrRes.postIsrExposure)
#                self.display("isr", exposure=isrRes.postIsrExposure, pause=True)
            # assemble amps into a CCD
            ccdExposure = self.isr.doCcdAssembly(exposureList)
            del exposureList
            # perform CCD-level ISR
            ccdCalibSet = self.ccdIsr.makeCalibDict(butler, sensorRef.dataId)
            ccdIsrRes = self.ccdIsr.run(ccdExposure, ccdCalibSet)
            ccdExposure = ccdIsrRes.postIsrExposure
            
#            self.display("ccdAssembly", exposure=ccdExposure)
        else:
            ccdExposure = None

        if self.config.doCalibrate:
            if ccdExposure is None:
                ccdExposure = sensorRef.get("postISRCCD")
            calib = self.calibrate.run(ccdExposure)
            ccdExposure = calib.exposure
        else:
            calib = None

        if self.config.doPhotometry:
            if ccdExposure is None:
                ccdExposure = sensorRef.get("calexp")
            if calib is None:
                psf = sensorRef.get("psf")
                apCorr = None # sensorRef.get('apcorr')
            else:
                psf = calib.psf
                apCorr = calib.apCorr
            phot = self.photometry.run(ccdExposure, psf, apcorr=apCorr)
        else:
            phot = None

        return pipeBase.Struct(
            postIsrExposure = postIsrExposure if self.config.doIsr else None,
            exposure = ccdExposure,
            psf = psf,
            apCorr = apCorr,
            sources = phot.sources if phot else None,
            matches = calib.matches if calib else None,
            matchMeta = calib.matchMeta if calib else None,
        )

    def persist(self, struct, sensorRef):
        """Persist results of run method
        """
        if self.config.doIsr:
            sensorRef.put(res.postIsrExposure, "postISRCCD")
        if self.config.doCalibrate:
            sensorRef.put(res.exposure, "calexp")
            sensorRef.put(afwDet.PersistableSourceVector(calib.sources), 'icSrc')
            sensorRef(res.psf, "psf")
            if calib.matches is not None:
                sensorRef.put(afwDet.PersistableSourceMatchVector(calib.matches, calib.matchMeta), 'icMatch')
        if self.config.doPhotometry and self.config.doWritePhotometry:
            sensorRef.put(afwDet.PersistableSourceVector(phot.sources), 'src')

    
    def runButler(self, sensorRef):
        """Run and persist results; log errors
        """
        try:
            res = self.run(sensorRef)
            self.persist(res, sensorRef)
        except Exception, e:
            self.log.log(self.log.FATAL, "Failed on dataId=%s: %s" % (sensorRef.dataId, e))
            raise
