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
import numpy as num
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.daf.base as dafBase
import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.afw.cameraGeom as afwCameraGeom
import lsst.afw.math as afwMath

from lsst.meas.algorithms import SourceDetectionTask, SourceMeasurementTask
from .calibrate import CalibrateTask

class ProcessCcdSdssCoaddConfig(pexConfig.Config):
    """Config for ProcessCcdSdssCoadd"""
    coaddName = pexConfig.Field(dtype=str, default="goodSeeingCoadd", doc = "Type of coadd")
    doScaleVariance = pexConfig.Field(dtype=bool, default=True, doc = "Scale variance plane using empirical noise") 

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

        self.calibrate.measurePsf.psfDeterminer["pca"].spatialOrder    = 1  # Should be spatially invariant
        self.calibrate.measurePsf.psfDeterminer["pca"].kernelSizeMin   = 31 # Larger Psfs
        self.calibrate.measurePsf.starSelector["secondMoment"].fluxLim = 3000.0

        self.calibrate.doBackground = False
        self.calibrate.detection.reEstimateBackground = False
        self.detection.reEstimateBackground = False
        self.detection.thresholdType = "pixel_stdev"

        self.calibrate.photocal.badFlags=['flags.pixel.edge','flags.pixel.saturated.any'] # Remove flags.pixel.interpolated.any

class ProcessCcdSdssCoaddTask(pipeBase.CmdLineTask):
    """Process a CCD for SDSS Coadd
    
    """
    ConfigClass = ProcessCcdSdssCoaddConfig
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
        return pipeBase.ArgumentParser(name=cls._DefaultName, datasetType=cls.ConfigClass().coaddName)        

    @pipeBase.timeMethod
    def scaleVariance(self, exposure):
        ctrl = afwMath.StatisticsControl()
        ctrl.setAndMask(~0x0)
        var    = exposure.getMaskedImage().getVariance()
        mask   = exposure.getMaskedImage().getMask()
        dstats = afwMath.makeStatistics(exposure.getMaskedImage(), afwMath.VARIANCECLIP, ctrl).getValue(afwMath.VARIANCECLIP)
        vstats = afwMath.makeStatistics(var, mask, afwMath.MEANCLIP, ctrl).getValue(afwMath.MEANCLIP)
        vrat   = dstats / vstats
        self.log.info("Renormalising variance by %f" % (vrat))
        var   *= vrat

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
        self.log.info("Processing %s" % (frameRef.dataId))
        if self.config.doCalibrate:
            self.log.info("Performing Calibrate on coadd %s" % (frameRef.dataId))
            coadd = frameRef.get(self.config.coaddName)
            if self.config.doScaleVariance:
                self.scaleVariance(coadd)

            calib = self.calibrate.run(coadd)
            calExposure = calib.exposure

            if self.config.doWriteCalibrate:
                frameRef.put(calExposure, self.config.coaddName+"_calexp")
                frameRef.put(calib.sources, self.config.coaddName+"_icSrc")
                if calib.psf is not None:
                    frameRef.put(calib.psf, self.config.coaddName+"_psf")
                if calib.apCorr is not None:
                    frameRef.put(calib.apCorr, self.config.coaddName+"_apCorr")
                if calib.matches is not None:
                    normalizedMatches = afwTable.packMatches(calib.matches)
                    normalizedMatches.table.setMetadata(calib.matchMeta)
                    frameRef.put(normalizedMatches, self.config.coaddName+"_icMatch")
        else:
            calib = None
            calExposure = None

        if self.config.doDetection:
            if calExposure is None:
                calExposure = frameRef.get(self.config.coaddName+"_calexp")
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
                apCorr = frameRef.get(self.config.coaddName+"_apCorr")
            else:
                apCorr = calib.apCorr
            self.measurement.run(calExposure, sources, apCorr)

        if self.config.doWriteSources:
            frameRef.put(sources, self.config.coaddName+"_src")

        return pipeBase.Struct(
            calExposure = calExposure,
            calib = calib,
            apCorr = apCorr,
            sources = sources,
            matches = calib.matches if calib else None,
            matchMeta = calib.matchMeta if calib else None,
        )
