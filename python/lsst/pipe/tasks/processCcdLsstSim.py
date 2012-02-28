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
    doWriteIsr = pexConfig.Field(dtype=bool, default=True, doc = "Write ISR results?")
    doWriteCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Write calibration results?")
    doWritePhotometry = pexConfig.Field(dtype=bool, default=True, doc = "Write photometry results?")
    isr = pexConfig.ConfigField(dtype=IsrTask.ConfigClass, doc="Amp-level instrumental signature removal")
    ccdIsr = pexConfig.ConfigField(dtype=IsrTask.ConfigClass, doc="CCD level instrumental signature removal")
    calibrate = pexConfig.ConfigField(dtype=CalibrateTask.ConfigClass, doc="Calibration")
    photometry = pexConfig.ConfigField(dtype=PhotometryTask.ConfigClass, doc="Photometry")

    def __init__(self):
        pexConfig.Config.__init__(self)
        self.doWriteIsr = False
        self.isr.methodList = ['doSaturationInterpolation', 'doMaskAndInterpDefect', 'doMaskAndInterpNan']
        self.isr.doWrite = False


class ProcessCcdLsstSimTask(pipeBase.Task):
    """Process a CCD for LSSTSim
    
    @todo: this variant of ProcessCcdTask can be eliminated once IsrTask is unified.
    """
    ConfigClass = ProcessCcdLsstSimConfig

    def __init__(self, *args, **kwargs):
        pipeBase.Task.__init__(self, *args, **kwargs)
        self.makeSubtask("isr", IsrTask)
        self.makeSubtask("ccdIsr", IsrTask)
        self.makeSubtask("calibrate", CalibrateTask)
        self.makeSubtask("photometry", PhotometryTask)

    @pipeBase.timeMethod
    def run(self, sensorRef):
        """Process a CCD: including ISR, source detection, photometry and WCS determination
        
        @param sensorRef: sensor-level butler data reference
        @return pipe_base Struct containing these fields:
        - postIsrExposure: exposure after ISR performed if calib.doIsr, else None
        - exposure: calibrated exposure (calexp)
        - psf: the PSF determined for the exposure
        - apCorr: aperture correction
        - sources: detected source if calib.doPhotometry run, else None
        - matches: ? if doCalibrate, else None
        - matchMeta: ? if config.doCalibrate, else None
        """
        self.log.log(self.log.INFO, "Processing %s" % (sensorRef.dataId))
        if self.config.doIsr:
            butler = sensorRef.butlerSubset.butler
            for snapRef in sensorRef.subItems(level="snap"):
                self.log.log(self.log.INFO, "Performing ISR on snap %s" % (snapRef.dataId))
                # perform amp-level ISR
                exposureList = list()
                for ampRef in snapRef.subItems(level="channel"):
                    self.log.log(self.log.INFO, "Performing ISR on channel %s" % (ampRef.dataId))
                    calibSet = self.isr.makeCalibDict(butler, ampRef.dataId)
                    ampExposure = ampRef.get("raw")
                    isrRes = self.isr.run(ampExposure, calibSet)
                    exposureList.append(isrRes.postIsrExposure)
                    self.display("isr", exposure=isrRes.postIsrExposure, pause=True)
                # assemble amps into a CCD
                tempExposure = self.isr.doCcdAssembly(exposureList)
                del exposureList
                # perform CCD-level ISR
                ccdCalibSet = self.ccdIsr.makeCalibDict(butler, snapRef.dataId)
                ccdIsrRes = self.ccdIsr.run(tempExposure, ccdCalibSet)
                del tempExposure
                postIsrExposure = ccdIsrRes.postIsrExposure
                
                self.display("ccdAssembly", exposure=postIsrExposure)
                if self.config.doWriteIsr:
                    snapRef.put(postIsrExposure, 'postISRCCD')
        else:
            postIsrExposure = None

        if self.config.doCalibrate:
            if postIsrExposure is None:
                postIsrExposure = sensorRef.get('postISRCCD')
            calib = self.calibrate.run(postIsrExposure)
            calExposure = calib.exposure
            if self.config.doWriteCalibrate:
                sensorRef.put(calExposure, 'calexp')
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
            if calExposure is None:
                calExposure = sensorRef.get('calexp')
            if calib is None:
                psf = sensorRef.get('psf')
                apCorr = None # sensorRef.get('apcorr')
            else:
                psf = calib.psf
                apCorr = calib.apCorr
            phot = self.photometry.run(calExposure, psf, apcorr=apCorr)
            if self.config.doWritePhotometry:
                sensorRef.put(afwDet.PersistableSourceVector(phot.sources), 'src')
        else:
            phot = None

        return pipeBase.Struct(
            postIsrExposure = postIsrExposure if self.config.doIsr else None,
            exposure = calExposure,
            psf = psf,
            apCorr = apCorr,
            sources = phot.sources if phot else None,
            matches = calib.matches if calib else None,
            matchMeta = calib.matchMeta if calib else None,
        )
