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
from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask, SourceDeblendTask
from lsst.ip.isr import IsrTask
from lsst.pipe.tasks.calibrate import CalibrateTask

class ProcessCcdConfig(pexConfig.Config):
    """Config for ProcessCcd"""
    doIsr = pexConfig.Field(dtype=bool, default=True, doc = "Perform ISR?")
    doCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Perform calibration?")
    doDetection = pexConfig.Field(dtype=bool, default=True, doc = "Detect sources?")
    ## NOTE, default this to False until it is fully vetted
    doDeblend = pexConfig.Field(dtype=bool, default=False, doc = "Deblend sources?")
    doMeasurement = pexConfig.Field(dtype=bool, default=True, doc = "Measure sources?")
    doWriteIsr = pexConfig.Field(dtype=bool, default=True, doc = "Write ISR results?")
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
    deblend = pexConfig.ConfigurableField(
        target = SourceDeblendTask,
        doc = "Split blended sources into their components",
    )
    measurement = pexConfig.ConfigurableField(
        target = SourceMeasurementTask,
        doc = "Final source measurement on low-threshold detections",
    )

    def validate(self):
        pexConfig.Config.validate(self)
        if self.doMeasurement and not self.doDetection:
            raise ValueError("Cannot run source measurement without source detection.")
        #if self.doDeblend and not self.doDetection:
        #    raise ValueError("Cannot run source deblending without source detection.")

    def setDefaults(self):
        self.doWriteIsr = False
        self.isr.methodList = ["doConversionForIsr", "doSaturationDetection",
                               "doOverscanCorrection", "doVariance", "doFlatCorrection"]
        self.isr.doWrite = False

class ProcessCcdTask(pipeBase.Task):
    """Process a CCD"""
    ConfigClass = ProcessCcdConfig
    _DefaultName = "processCcd"

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("calibrate")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", schema=self.schema)
        if self.config.doDeblend:
            self.makeSubtask("deblend", schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

    @pipeBase.timeMethod
    def run(self, sensorRef):
        self.log.info("Processing %s" % (sensorRef.dataId))
        psf = None
        if self.config.doIsr:
            butler = sensorRef.butlerSubset.butler
            calibSet = self.isr.makeCalibDict(butler, sensorRef.dataId)
            rawExposure = sensorRef.get("raw")
            isrRes = self.isr.run(rawExposure, calibSet)
            self.display("isr", exposure=isrRes.postIsrExposure, pause=True)
            exposure = self.isr.doCcdAssembly([isrRes.postIsrExposure])
            self.display("ccdAssembly", exposure=exposure)
            if self.config.doWriteIsr:
                sensorRef.put(exposure, 'postISRCCD')
        else:
            exposure = None

        if self.config.doCalibrate:
            if exposure is None:
                exposure = sensorRef.get('postISRCCD')
            calib = self.calibrate.run(exposure)
            exposure = calib.exposure
            psf = calib.psf
            if self.config.doWriteCalibrate:
                sensorRef.put(exposure, 'calexp')
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

        if self.config.doDetection:
            if exposure is None:
                exposure = sensorRef.get('calexp')
            if calib is None:
                psf = sensorRef.get('psf')
                exposure.setPsf(sensorRef.get('psf'))
            table = afwTable.SourceTable.make(self.schema)
            table.setMetadata(self.algMetadata)
            detRet = self.detection.makeSourceCatalog(table, exposure)
            sources = detRet.sources
        else:
            sources = None

        if self.config.doDeblend:
            if exposure is None:
                exposure = sensorRef.get('calexp')
            if psf is None:
                psf = sensorRef.get('psf')
            if sources is None:
                # This is kind of a strange situation: we're re-reading a 'src'
                # data item, which may sources that were previously deblended!
                sources = sensorRef.get('src')
                self.log.info("Reading 'src' data for deblending: got %i sources" % len(sources))
                # Make sure the IdFactory exists and doesn't duplicate IDs
                # (until JimB finishes #2083)
                f = sources.getTable().getIdFactory()
                if f is None:
                    f = afwTable.IdFactory.makeSimple()
                    sources.getTable().setIdFactory(f)
                f.notify(max([src.getId() for src in sources]))
                #
                # Remove children from the Catalog (don't re-deblend)
                n0 = len(sources)
                i=0
                while i < len(sources):
                    if sources[i].getParent():
                        del sources[i]
                    else:
                        i += 1
                n1 = len(sources)
                if n1 != n0:
                    self.log.info("Dropped %i of %i 'child' sources (%i remaining)" %
                                  ((n0-n1), n0, n1))
                
            assert(exposure)
            assert(psf)
            assert(sources)
            self.deblend.run(exposure, sources, psf)

        if self.config.doMeasurement:
            assert(sources)
            assert(exposure)
            if calib is None:
                apCorr = sensorRef.get("apCorr")
            else:
                apCorr = calib.apCorr
            self.measurement.run(exposure, sources, apCorr)
 
        if self.config.doWriteSources:
            sensorRef.put(sources, 'src')

        if self.config.doWriteCalibrate:
            sensorRef.put(exposure, 'calexp')
            
        return pipeBase.Struct(
            exposure = exposure,
            calib = calib,
            sources = sources,
        )
