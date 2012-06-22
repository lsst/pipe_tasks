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

class ProcessCcdConfig(pexConfig.Config):
    """Config for ProcessCcd"""
    doIsr = pexConfig.Field(dtype=bool, default=True, doc = "Perform ISR?")
    doCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Perform calibration?")
    doDetection = pexConfig.Field(dtype=bool, default=True, doc = "Detect sources?")
    doMeasurement = pexConfig.Field(dtype=bool, default=True, doc = "Measure sources?")
    doWriteCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Write calibration results?")
    doWriteSources = pexConfig.Field(dtype=bool, default=True, doc = "Write sources?")
    isr = pexConfig.ConfigurableField(
        target = IsrTask,
        doc = "Instrumental Signature Removal",
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

class ProcessCcdTask(pipeBase.CmdLineTask):
    """Process a CCD
    
    Available steps include:
    - instrument signature removal (ISR)
    - calibrate
    - detect sources
    - measure sources
    """
    ConfigClass = ProcessCcdConfig
    _DefaultName = "processCcd"

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("calibrate")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

    @pipeBase.timeMethod
    def run(self, sensorRef):
        """Process one CCD
        
        @param sensorRef: sensor-level butler data reference
        @return pipe_base Struct containing these fields:
        - postIsrExposure: exposure after ISR performed if calib.doIsr or config.doCalibrate, else None
        - exposure: calibrated exposure (calexp): as computed if config.doCalibrate,
            else as upersisted and updated if config.doDetection, else None
        - calib: object returned by calibration process if config.doCalibrate, else None
        - apCorr: aperture correction: as computed config.doCalibrate, else as unpersisted
            if config.doMeasure, else None
        - sources: detected source if config.doPhotometry, else None
        """
        self.log.log(self.log.INFO, "Processing %s" % (sensorRef.dataId))

        # We make one IdFactory that will be used by both icSrc and src datasets;
        # I don't know if this is the way we ultimately want to do things, but at least
        # this ensures the source IDs are fully unique.
        expBits = sensorRef.get("ccdExposureId_bits")
        expId = long(sensorRef.get("ccdExposureId"))
        idFactory = afwTable.IdFactory.makeSource(expId, 64 - expBits)

        # initialize outputs
        postIsrExposure = None
        calExposure = None
        calib = None
        apCorr = None
        sources = None
        
        if self.config.doIsr:
            postIsrExposure = self.isr.run(sensorRef).exposure

        if self.config.doCalibrate:
            if postIsrExposure is None:
                postIsrExposure = sensorRef.get("postISRCCD")
            calib = self.calibrate.run(postIsrExposure, idFactory=idFactory)
            calExposure = calib.exposure
            apCorr = calib.apCorr
            if self.config.doWriteCalibrate:
                sensorRef.put(calib.sources, "icSrc")
                if calib.psf is not None:
                    sensorRef.put(calib.psf, "psf")
                if calib.apCorr is not None:
                    sensorRef.put(calib.apCorr, "apCorr")
                if calib.matches is not None:
                    normalizedMatches = afwTable.packMatches(calib.matches)
                    normalizedMatches.table.setMetadata(calib.matchMeta)
                    sensorRef.put(normalizedMatches, "icMatch")

        if self.config.doDetection:
            if calExposure is None:
                if not sensorRef.datasetExists("calexp"):
                    raise pipeBase.TaskError("doCalibrate false, doDetection true and calexp does not exist")
                calExposure = sensorRef.get("calexp")
            if calib is None or calib.psf is None:
                psf = sensorRef.get("psf")
                calExposure.setPsf(psf)
            table = afwTable.SourceTable.make(self.schema, idFactory)
            table.setMetadata(self.algMetadata)
            sources = self.detection.makeSourceCatalog(table, calExposure).sources

        if self.config.doWriteCalibrate:
            # wait until after detection, since that sets detected mask bits and may tweak the background;
            # note that this overwrites an existing calexp if doCalibrate false
            if calExposure is None:
                self.log.log(self.log.WARN, "calibrated exposure is None; cannot save it")
            else:
                sensorRef.put(calExposure, "calexp")

        if self.config.doMeasurement:
            if apCorr is None:
                apCorr = sensorRef.get("apCorr")
            self.measurement.run(calExposure, sources, apCorr)

        if sources is not None and self.config.doWriteSources:
            sensorRef.put(sources, "src")

        return pipeBase.Struct(
            postIsrExposure = postIsrExposure,
            exposure = calExposure,
            calib = calib,
            apCorr = apCorr,
            sources = sources,
        )
