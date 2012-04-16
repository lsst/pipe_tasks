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
import lsst.daf.base as dafBase
import lsst.afw.table as afwTable
from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask
from lsst.ip.isr import IsrTask
from lsst.pipe.tasks.calibrate import CalibrateTask
from lsst.pipe.tasks.snapCombine import SnapCombineTask

class ProcessCcdLsstSimConfig(pexConfig.Config):
    """Config for ProcessCcdLsstSim"""
    doIsr = pexConfig.Field(dtype=bool, default=True, doc = "Perform ISR?")
    doSnapCombine = pexConfig.Field(dtype=bool, default=True, doc = "Combine Snaps?")
    doCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Perform calibration?")
    doDetection = pexConfig.Field(dtype=bool, default=True, doc = "Detect sources?")
    doMeasurement = pexConfig.Field(dtype=bool, default=True, doc = "Measure sources?")
    doWriteIsr = pexConfig.Field(dtype=bool, default=True, doc = "Write ISR results?")
    doWriteSnapCombine = pexConfig.Field(dtype=bool, default=True, doc = "Write snapCombine results?")  
    doWriteCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Write calibration results?")
    doWriteSources = pexConfig.Field(dtype=bool, default=True, doc = "Write sources?")
    isr = pexConfig.ConfigurableField(
        target = IsrTask,
        doc = "Amp-level instrumental signature removal",
    )
    ccdIsr = pexConfig.ConfigurableField(
        target = IsrTask,
        doc = "CCD level instrumental signature removal (deprecated; isr will soon do it all)",
    )
    snapCombine = pexConfig.ConfigurableField(
        target = SnapCombineTask,
        doc = "Combine snaps",
    )
    calibrate = pexConfig.ConfigurableField(
        target = CalibrateTask,
        doc = "Calibration (inc. high-threshold detection and measurement)",
    )
    detection = pexConfig.ConfigurableField(
        target = SourceDetectionTask,
        doc = "Low-threshold detection for final measurement",
    )
    measurement = pexConfig.ConfigurableField(
        target = SourceMeasurementTask,
        doc = "Final source measurement on low-threshold detections",
    )

    def validate(self):
        pexConfig.Config.validate(self)
        if self.doMeasurement and not self.doDetection:
            raise ValueError("Cannot run source measurement without source detection.")

    def setDefaults(self):
        self.isr.doWrite = False
        self.ccdIsr.methodList = ['doSaturationInterpolation', 'doMaskAndInterpDefect', 'doMaskAndInterpNan']
        self.ccdIsr.doWrite = False

        # FIXME: unless these defaults need to be different from the subtask defaults,
        #        don't repeat them here
        self.snapCombine.doPsfMatch = True
        self.snapCombine.repair.doInterpolate = True
        self.snapCombine.diffim.kernel.name = "DF"
        self.snapCombine.diffim.kernel.active.spatialKernelOrder = 1
        self.snapCombine.coadd.badMaskPlanes = ["EDGE"]
        self.snapCombine.detection.thresholdValue = 5.0


class ProcessCcdLsstSimTask(pipeBase.CmdLineTask):
    """Process a CCD for LSSTSim
    
    @todo: this variant of ProcessCcdTask can be eliminated once IsrTask is unified.
    """
    ConfigClass = ProcessCcdLsstSimConfig
    _DefaultName = "processCcd"

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("ccdIsr")
        self.makeSubtask("snapCombine")
        self.makeSubtask("calibrate")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

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
        snap0 = None
        snap1 = None
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
                    snapRef.put(postIsrExposure, "postISRCCD")

                if snapRef.dataId['snap'] == 0:
                    snap0 = postIsrExposure
                elif snapRef.dataId['snap'] == 1:
                    snap1 = postIsrExposure

        if self.config.doSnapCombine:
            if snap0 is None or snap1 is None:
                snap0 = sensorRef.get("postISRCCD", snap=0)
                snap1 = sensorRef.get("postISRCCD", snap=1)

            combineRes = self.snapCombine.run(snap0, snap1)
            visitExposure = combineRes.visitExposure
            self.display("snapCombine", exposure=visitExposure)
            if self.config.doWriteSnapCombine:
                sensorRef.put(visitExposure, "visitCCD", snap=0)
        else:
            visitExposure = sensorRef.get("visitCCD", snap=0)

        if self.config.doCalibrate:
            if visitExposure is None:
                if self.config.doSnapCombine:
                    visitExposure = sensorRef.get('visitCCD')
                else:
                    visitExposure = sensorRef.get('postISRCCD')
            calib = self.calibrate.run(visitExposure)
            calExposure = calib.exposure

            if self.config.doWriteCalibrate:
                sensorRef.put(calExposure, 'calexp')
                sensorRef.put(calib.sources, 'icSrc')
                if calib.psf is not None:
                    sensorRef.put(calib.psf, 'psf')
                if calib.apCorr is not None:
                    sensorRef.put(calib.apCorr, 'apCorr')
                if calib.matches is not None:
                    normalizedMatches = afwTable.packMatches(calib.matches)
                    normalizedMatches.table.setMetadata(calib.matchMeta)
                    sensorRef.put(normalizedMatches, 'icMatch')
        else:
            calib = None
            calExposure = None

        if self.config.doDetection:
            if calExposure is None:
                calExposure = sensorRef.get('calexp')
            if calib is None:
                psf = sensorRef.get('psf')
                calExposure.setPsf(sensorRef.get('psf'))
            table = afwTable.SourceTable.make(self.schema)
            table.setMetadata(self.algMetadata)
            detRet = self.detection.makeSourceCatalog(table, calExposure)
            sources = detRet.sources
        else:
            sources = None

        if self.config.doMeasurement:
            assert(sources)
            assert(calExposure)
            if calib is None:
                apCorr = sensorRef.get("apCorr")
            else:
                apCorr = calib.apCorr
            self.measurement.run(calExposure, sources, apCorr)

        if self.config.doWriteSources:
            sensorRef.put(sources, 'src')

        return pipeBase.Struct(
            postIsrExposure = postIsrExposure if self.config.doIsr else None,
            visitExposure = visitExposure if self.config.doSnapCombine else None,
            calExposure = calExposure,
            calib = calib,
            apCorr = apCorr,
            sources = sources,
            matches = calib.matches if calib else None,
            matchMeta = calib.matchMeta if calib else None,
        )
