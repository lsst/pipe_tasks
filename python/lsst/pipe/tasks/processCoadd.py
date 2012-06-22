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
from lsst.pipe.tasks.calibrate import CalibrateTask
from .coadd import CoaddArgumentParser

class ProcessCoaddConfig(pexConfig.Config):
    """Config for ProcessCoadd"""
    coaddName = pexConfig.Field(
        doc = "coadd name: typically one of deep or goodSeeing",
        dtype = str,
        default = "deep",
    )
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

        self.calibrate.computeApCorr.badFlags = ("flags.pixel.edge", "flags.pixel.saturated.any") # Remove flags.pixel.interpolated.any
        self.calibrate.photocal.badFlags=['flags.pixel.edge','flags.pixel.saturated.any'] # Remove flags.pixel.interpolated.any

class ProcessCoaddTask(pipeBase.CmdLineTask):
    """Process a CCD for SDSS Coadd
    
    """
    ConfigClass = ProcessCoaddConfig
    _DefaultName = "processCoadd"

    def __init__(self, **kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.makeSubtask("calibrate")
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.algMetadata = dafBase.PropertyList()
        if self.config.doDetection:
            self.makeSubtask("detection", schema=self.schema)
        if self.config.doMeasurement:
            self.makeSubtask("measurement", schema=self.schema, algMetadata=self.algMetadata)

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
    def run(self, sensorRef):
        """Process a CCD: including source detection, photometry and WCS determination
        
        @param sensorRef: sensor-level butler data reference to SDSS coadd patch
        @return pipe_base Struct containing these fields:
        - exposure: calibrated exposure (calexp): as computed if config.doCalibrate,
            else as upersisted and updated if config.doDetection, else None
        - calib: object returned by calibration process if config.doCalibrate, else None
        - apCorr: aperture correction: as computed config.doCalibrate, else as unpersisted
            if config.doMeasure, else None
        - sources: detected source if config.doPhotometry, else None
        """
        self.log.log(self.log.INFO, "Processing %s" % (sensorRef.dataId))
        outPrefix = self.config.coaddName + "Coadd_"

        # We make one IdFactory that will be used by both icSrc and src datasets;
        # I don't know if this is the way we ultimately want to do things, but at least
        # this ensures the source IDs are fully unique.
        expBits = sensorRef.get(self.config.coaddName + "CoaddId_bits")
        expId = long(sensorRef.get(self.config.coaddName + "CoaddId"))
        idFactory = afwTable.IdFactory.makeSource(expId, 64 - expBits)

        # initialize outputs
        calExposure = None
        calib = None
        apCorr = None
        sources = None

        if self.config.doCalibrate:
            self.log.log(self.log.INFO, "Performing Calibrate on coadd %s" % (sensorRef.dataId))
            coadd = sensorRef.get(self.config.coaddName+"Coadd")
            if self.config.doScaleVariance:
                self.scaleVariance(coadd)

            calib = self.calibrate.run(coadd, idFactory=idFactory)
            calExposure = calib.exposure
            apCorr = calib.apCorr
            if self.config.doWriteCalibrate:
                sensorRef.put(calib.sources, outPrefix+"icSrc")
                if calib.psf is not None:
                    sensorRef.put(calib.psf, outPrefix+"psf")
                if calib.apCorr is not None:
                    sensorRef.put(calib.apCorr, outPrefix+"apCorr")
                if calib.matches is not None:
                    normalizedMatches = afwTable.packMatches(calib.matches)
                    normalizedMatches.table.setMetadata(calib.matchMeta)
                    sensorRef.put(normalizedMatches, outPrefix+"icMatch")

        if self.config.doDetection:
            if calExposure is None:
                calexpName = outPrefix+"calexp"
                if not sensorRef.datasetExists(calexpName):
                    raise pipeBase.TaskError("doCalibrate false, doDetection true and %s does not exist" % \
                        (calexpName,))
                calExposure = sensorRef.get(calexpName)
            table = afwTable.SourceTable.make(self.schema, idFactory)
            table.setMetadata(self.algMetadata)
            sources = self.detection.makeSourceCatalog(table, calExposure).sources

        if self.config.doWriteCalibrate:
            # wait until after detection, since that sets detected mask bits and may tweak the background;
            # note that this overwrites an existing calexp if doCalibrate false
            if calExposure is None:
                self.log.log(self.log.WARN, "calibrated exposure is None; cannot save it")
            else:
                sensorRef.put(calExposure, outPrefix+"calexp")

        if self.config.doMeasurement:
            if calib is None:
                apCorr = sensorRef.get(outPrefix+"apCorr")
            self.measurement.run(calExposure, sources, apCorr)

        if self.config.doWriteSources:
            sensorRef.put(sources, outPrefix+"src")

        return pipeBase.Struct(
            exposure = calExposure,
            calib = calib,
            apCorr = apCorr,
            sources = sources,
        )

    @classmethod
    def _makeArgumentParser(cls):
        return CoaddArgumentParser(name=cls._DefaultName, datasetType="deepCoadd")

    def _getConfigName(self):
        """Return the name of the config dataset
        """
        return "%s_processCoadd_config" % (self.config.coaddName,)
    
    def _getMetadataName(self):
        """Return the name of the metadata dataset
        """
        return "%s_processCoadd_metadata" % (self.config.coaddName,)
