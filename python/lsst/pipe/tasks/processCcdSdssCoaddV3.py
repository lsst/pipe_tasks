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
import lsst.afw.cameraGeom as afwCameraGeom
from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask
from lsst.pipe.tasks.calibrate import CalibrateTask

class ProcessCcdSdssCoaddV3Config(pexConfig.Config):
    """Config for ProcessCcdSdssCoaddV3"""
    doScaleVariance = pexConfig.Field(dtype=bool, default=True, doc = "Scale the variance plane, which are incorrect in V3 coadds?")
    varScaleFactor  = pexConfig.Field(dtype=float, default=10.0, doc = "Value to scale the variance by")

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
        self.calibrate.repair.doInterpolate = False
        self.calibrate.repair.doCosmicRay = False

        self.calibrate.background.binSize = 512 # Message: nySample has too few points for requested interpolation style.
        self.calibrate.initialPsf.fwhm = 2.5    # Degraded the seeing for coadd to 2.5 arcseconds

class ProcessCcdSdssCoaddV3Task(pipeBase.CmdLineTask):
    """Process a CCD for SDSS Coadd (V3)
    
    """
    ConfigClass = ProcessCcdSdssCoaddV3Config
    _DefaultName = "processCcd"

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("calibrate")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

    @classmethod
    def _makeArgumentParser(cls):
        return pipeBase.ArgumentParser(name=cls._DefaultName, datasetType="coadd")        

    @pipeBase.timeMethod
    def makeExp(self, frameRef):
        exp = frameRef.get("coadd")
        if self.config.doScaleVariance:
            var  = exp.getMaskedImage().getVariance()
            var *= self.config.varScaleFactor

        det = afwCameraGeom.Detector(afwCameraGeom.Id("%s%d" %
                                                      (frameRef.dataId["filter"], frameRef.dataId["camcol"])))
        exp.setDetector(det)
        exp.setFilter(afwImage.Filter(frameRef.dataId['filter']))

        return exp

    @pipeBase.timeMethod
    def run(self, frameRef):
        """Process a CCD: including source detection, photometry and WCS determination
        
        @param frameRef: frame-level butler data reference
        @return pipe_base Struct containing these fields:
        - exposure: calibrated exposure (calexp)
        - psf: the PSF determined for the exposure
        - apCorr: aperture correction
        - sources: detected source if calib.doPhotometry run, else None
        - matches: ? if doCalibrate, else None
        - matchMeta: ? if config.doCalibrate, else None
        """
        self.log.log(self.log.INFO, "Processing %s" % (frameRef.dataId))

        if self.config.doCalibrate:
            self.log.log(self.log.INFO, "Performing Calibrate on coadd %s" % (frameRef.dataId))
            exp = self.makeExp(frameRef)
            calib = self.calibrate.run(exp)
            calExposure = calib.exposure

            if self.config.doWriteCalibrate:
                frameRef.put(calExposure, 'coadd_calexp')
                frameRef.put(calib.sources, 'coadd_icSrc')
                if calib.psf is not None:
                    frameRef.put(calib.psf, 'coadd_psf')
                if calib.apCorr is not None:
                    frameRef.put(calib.apCorr, 'coadd_apCorr')
                if calib.matches is not None:
                    normalizedMatches = afwTable.packMatches(calib.matches)
                    normalizedMatches.table.setMetadata(calib.matchMeta)
                    frameRef.put(normalizedMatches, 'coadd_icMatch')
        else:
            calib = None
            calExposure = None

        if self.config.doDetection:
            if calExposure is None:
                calExposure = frameRef.get('coadd_calexp')
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
                apCorr = frameRef.get("coadd_apCorr")
            else:
                apCorr = calib.apCorr
            self.measurement.run(calExposure, sources, apCorr)

        if self.config.doWriteSources:
            frameRef.put(sources, 'coadd_src')

        return pipeBase.Struct(
            calExposure = calExposure,
            calib = calib,
            apCorr = apCorr,
            sources = sources,
            matches = calib.matches if calib else None,
            matchMeta = calib.matchMeta if calib else None,
        )
