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
import lsst.afw.image as afwImage
from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask
from lsst.pipe.tasks.calibrate import CalibrateTask

class ProcessCcdSdssConfig(pexConfig.Config):
    """Config for ProcessCcdSdss"""
    doCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Perform calibration?")
    doDetection = pexConfig.Field(dtype=bool, default=True, doc = "Detect sources?")
    doMeasurement = pexConfig.Field(dtype=bool, default=True, doc = "Measure sources?")
    doWriteCalibrate = pexConfig.Field(dtype=bool, default=True, doc = "Write calibration results?")
    doWriteSources = pexConfig.Field(dtype=bool, default=True, doc = "Write sources?")
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
        # OPTIMIZE FOR SDSS
        #import pdb; pdb.set_trace()
        self.calibrate.repair.doInterpolate = False
        self.calibrate.repair.doCosmicRay = False

        self.calibrate.background.binSize = 512
        
        self.calibrate.doAstrometry = False     # RuntimeError: Can't find Ccd from detector.
        self.calibrate.doPhotoCal = False       # ValueError: Cannot do photometric calibration without doing astrometric matching
        #self.doWriteCalibrate = False # TypeError: in method 'Persistence_persist', argument 2 of type 'lsst::daf::base::Persistable const &'
        #self.doWriteSources = False # TypeError: in method 'Persistence_persist', argument 2 of type 'lsst::daf::base::Persistable const &'



class ProcessCcdSdssTask(pipeBase.CmdLineTask):
    """Process a CCD for SDSS
    
    """
    ConfigClass = ProcessCcdSdssConfig
    _DefaultName = "processCcd"

    def __init__(self, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)
        self.makeSubtask("calibrate")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

    @classmethod
    def _makeArgumentParser(cls):
        return pipeBase.ArgumentParser(name=cls._DefaultName, datasetType="fpC")        

    @pipeBase.timeMethod
    def makeExp(self, frameRef, gain = 1.0):
        image = frameRef.get("fpC").convertF()
        mask  = frameRef.get("fpM")
        wcs   = frameRef.get("asTrans")
        var   = afwImage.ImageF(image, True)
        var  /= gain

        mi    = afwImage.MaskedImageF(image, mask, var)
        exp   = afwImage.ExposureF(mi, wcs)
        return exp


    @pipeBase.timeMethod
    def run(self, sensorRef):
        """Process a CCD: including source detection, photometry and WCS determination
        
        @param sensorRef: sensor-level butler data reference
        @return pipe_base Struct containing these fields:
        - exposure: calibrated exposure (calexp)
        - psf: the PSF determined for the exposure
        - apCorr: aperture correction
        - sources: detected source if calib.doPhotometry run, else None
        - matches: ? if doCalibrate, else None
        - matchMeta: ? if config.doCalibrate, else None
        """
        self.log.log(self.log.INFO, "Processing %s" % (sensorRef.dataId))

        if self.config.doCalibrate:
            for frameRef in sensorRef.subItems(level="frame"):
                self.log.log(self.log.INFO, "Performing Calibrate on fpC %s" % (frameRef.dataId))
                exp = self.makeExp(frameRef)
                calib = self.calibrate.run(exp)
                calExposure = calib.exposure
                frameRef.put(calExposure, "calexp")

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
            calExposure = calExposure,
            calib = calib,
            apCorr = apCorr,
            sources = sources,
            matches = calib.matches if calib else None,
            matchMeta = calib.matchMeta if calib else None,
        )
